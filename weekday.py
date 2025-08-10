#!/usr/bin/env python3
"""
ministry_scraper.py (revised)
=============================
Scrapes a directory of **Christian‑based ministries in the USA** (state → county → ministry)
with paged DuckDuckGo search, Playwright MCP browsing, and AI enrichment.

This revision focuses on **correctly separating organization_name vs Title** using an AI resolver
plus final validation to avoid aggregator names (MapQuest, Facebook, Yelp, etc.). It also adds a
`--test-url` flag for quick debugging of a single page.

Output Schema (columns)
-----------------------
organization_name, Ministry Categories, Title, link_to_ministry, phone, Content, email,
address, city, state, zip, country, source_site
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urljoin, urlparse

import aiohttp
import pandas as pd
import us
import usaddress
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright
from langchain_community.agent_toolkits.playwright import PlayWrightBrowserToolkit

load_dotenv()

# ---------- Files & constants ---------------------------------------------------------------
OUTPUT_DIR = Path("out"); OUTPUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUTPUT_DIR / "ministry_directory.csv"
CATEGORIES_PATH = Path("categories.csv")
COUNTIES_PATH = Path("counties.json")

BATCH_SIZE_DEFAULT = 100
MAX_PAGES_DEFAULT = 5
CONCURRENCY_DEFAULT = 5

CSV_HEADERS = [
    "organization_name",
    "Ministry Categories",
    "Title",
    "link_to_ministry",
    "phone",
    "Content",
    "email",
    "address",
    "city",
    "state",
    "zip",
    "country",
    "source_site",
]

# ---------- Models --------------------------------------------------------------------------
class MinistryRecord(BaseModel):
    org_name: str = Field(alias="organization_name")
    ministry_category: list[str] | str = Field(alias="Ministry Categories")
    title: str | None = Field(alias="Title")
    link_to_ministry: str
    phone: str | None = None
    content_summary: str | None = Field(alias="Content")
    email: str | None = None
    address: str | None = None
    city: str | None = None
    state: str
    zip: str | None = None
    country: str = "USA"
    source_site: str

    class Config:
        populate_by_name = True

# ---------- Regex & small utils -------------------------------------------------------------
MINISTRY_REGEX = re.compile(r"church|ministr(?:y|ies)|fellowship|christian|gospel|missions", re.I)
EMAIL_REGEX = re.compile(r"[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})")
SKIP_MIME_PREFIXES = ("application/pdf", "application/msword", "application/vnd", "image/")
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
TITLE_SEPS = (" | ", " – ", " — ", " · ", " : ", " :: ")

ADDRESS_RE = re.compile(
    r"\d{1,5}\s+[A-Za-z0-9.,\-\s']+?,\s*([A-Za-z\s]+),\s*([A-Z]{2})\s+(\d{5})(?:-\d{4})?"
)

AGGREGATOR_DOMAINS = {
    "mapquest.com", "yelp.com", "facebook.com", "m.facebook.com", "business.facebook.com",
    "instagram.com", "twitter.com", "x.com", "linkedin.com", "goo.gl", "google.com", "g.page",
    "bing.com", "duckduckgo.com", "eventbrite.com", "meetup.com", "constantcontact.com",
    "blackbaudhosting.com", "donorbox.org", "pushpay.com", "paypal.com", "square.site", "my.canva.site",
    "linktr.ee", "linktree.com"
}
AGGREGATOR_NAMES = {
    "mapquest","facebook","yelp","google","bing","duckduckgo","eventbrite","meetup",
    "paypal","pushpay","donorbox","square","linktree","instagram","twitter","x","linkedin"
}

def _clean_name_from_titles(page_title: str | None, result_title: str | None) -> str | None:
    def _clean(t: str) -> str:
        t = re.sub(r"\s*[-–—|:·]+\s*", " ", t).strip()
        t = re.sub(r"\bon\s+(MapQuest|Facebook|Yelp|Google)\b.*$", "", t, flags=re.I).strip()
        t = re.sub(r"(MapQuest|Facebook|Yelp|Google)\b.*$", "", t, flags=re.I).strip()
        return t
    candidates: list[str] = []
    if page_title: candidates.append(_clean(page_title))
    if result_title: candidates.append(_clean(result_title))
    candidates = [c for c in candidates if c and len(c) > 2]
    return min(candidates, key=len) if candidates else None

# ---------- County data ---------------------------------------------------------------------
COUNTY_MAP = json.loads(COUNTIES_PATH.read_text()) if COUNTIES_PATH.exists() else {}

def get_state_abbrev(state_name: str) -> str:
    st = us.states.lookup(state_name)
    if not st:
        raise ValueError(f"Unknown state: {state_name}")
    return st.abbr  # type: ignore

# ---------- HTTP helpers --------------------------------------------------------------------
_async_session: aiohttp.ClientSession | None = None
async def _session() -> aiohttp.ClientSession:
    global _async_session
    if _async_session is None:
        _async_session = aiohttp.ClientSession()
    return _async_session

async def is_download(url: str) -> bool:
    s = await _session()
    try:
        async with s.get(url, timeout=15, allow_redirects=True, headers={"Range":"bytes=0-0"}) as r:
            ctype = (r.headers.get("content-type") or "").lower()
            return ctype.startswith(SKIP_MIME_PREFIXES)
    except Exception:
        return True

async def fetch_html(url: str) -> str | None:
    s = await _session()
    try:
        async with s.get(url, allow_redirects=True, headers={"User-Agent": UA}) as r:
            if r.status != 200:
                return None
            ctype = (r.headers.get("content-type") or "").lower()
            if not ctype.startswith("text/html"):
                return None
            raw = await r.read()
            return raw.decode(errors="ignore")
    except Exception:
        return None

def visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "form"]):
        tag.decompose()
    return "\n".join(soup.stripped_strings)

# ---------- Org / title / address extraction ------------------------------------------------

def extract_org_name(soup: BeautifulSoup, url: str, result_title: str | None = None) -> str:
    host = (urlparse(url).hostname or "").lower()

    # 1) JSON-LD Organization / LocalBusiness / Church
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue
        obj = data if isinstance(data, dict) else (data[0] if isinstance(data, list) and data else None)
        if isinstance(obj, dict) and obj.get("@type") in ("Organization", "LocalBusiness", "Church"):
            name = obj.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()

    # 2) Site brand/text in header
    brand = soup.find(attrs={"class": re.compile(r"(site-)?brand|logo|site-title", re.I)})
    if brand and brand.get_text(strip=True):
        return brand.get_text(strip=True)

    # 3) Aggregator handling: parse from the titles if available
    def _shave(txt: str) -> str:
        txt = re.sub(r"\s*[-–—|:·]+\s*", " ", txt)
        txt = re.sub(r"\bon\s+(MapQuest|Facebook|Yelp)\b.*$", "", txt, flags=re.I).strip()
        txt = re.sub(r"(MapQuest|Facebook|Yelp)\b.*$", "", txt, flags=re.I).strip()
        return txt

    page_title = (soup.title.string or "").strip() if soup.title else ""
    if host in AGGREGATOR_DOMAINS and (result_title or page_title):
        pool = [p for p in (_shave(page_title), _shave(result_title or "")) if p]
        if pool:
            return sorted(pool, key=len)[0]

    # 4) Domain fallback (last resort)
    core_parts = host.split(":")[0].split(".")
    core = core_parts[-2] if len(core_parts) >= 2 else core_parts[0]
    return core.replace("-", " ").title() if core else (result_title or url)


def extract_ministry_title(soup: BeautifulSoup, url: str, org_name: str, result_title: str | None = None) -> str:
    def cleaned(t: str) -> str:
        t = t.strip()
        # avoid returning branding only
        if org_name and org_name.lower() in t.lower() and len(t) - len(org_name) < 10:
            return ""
        for sep in ("|", "–", "—", "·", ":", "::"):
            t = " ".join([p.strip() for p in t.split(sep) if p.strip()])
        t = re.sub(r"\bon\s+(MapQuest|Facebook|Yelp)\b.*$", "", t, flags=re.I).strip()
        return t

    # Prefer H1
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        t = cleaned(h1.get_text(strip=True))
        if t:
            return t

    # Then page <title>
    page_title = (soup.title.string or "").strip() if soup.title else ""
    if page_title:
        t = cleaned(page_title)
        if t:
            return t

    # Then SERP title
    if result_title:
        t = cleaned(result_title)
        if t:
            return t

    # Final fallback
    return org_name


def _addr_from_json_ld(soup: BeautifulSoup):
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue
        obj = data if isinstance(data, dict) else (data[0] if isinstance(data, list) and data else None)
        if isinstance(obj, dict) and isinstance(obj.get("address"), dict):
            a = obj["address"]
            return {
                "address": a.get("streetAddress"),
                "city": a.get("addressLocality"),
                "state": a.get("addressRegion"),
                "zip": a.get("postalCode"),
            }


def _addr_from_html_tag(soup: BeautifulSoup):
    tag = soup.find("address")
    if tag:
        return _addr_from_regex(tag.get_text(" ", strip=True))


def _addr_from_regex(text: str):
    m = ADDRESS_RE.search(text)
    if not m:
        return None
    try:
        parsed, _ = usaddress.tag(m.group(0))
    except usaddress.RepeatedLabelError:
        return None
    return {
        "address": (f"{parsed.get('AddressNumber','')} {parsed.get('StreetName','')} "
                    f"{parsed.get('StreetNamePostType','')}").strip(),
        "city": parsed.get("PlaceName"),
        "state": parsed.get("StateName"),
        "zip": parsed.get("ZipCode"),
    }


def extract_address_fields(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    return (
        _addr_from_json_ld(soup)
        or _addr_from_html_tag(soup)
        or _addr_from_regex(visible_text(html))
        or {}
    )

# ---------- Category classifier & summariser -------------------------------------------------
class CategoryClassifier:
    """Maps description → one or more categories from categories.csv.

    Accepts:
    1) two-column [category, keywords]
    2) single-column with any of ", ; \t /" as separators
    """
    def __init__(self, csv_path: Path):
        raw = pd.read_csv(csv_path, sep=None, engine="python")
        splitter = re.compile(r"[\t,;]")
        first_col = raw.iloc[:, 0].dropna().astype(str)
        cats: list[str] = []
        for cell in first_col:
            cats.extend([c.strip() for c in splitter.split(cell) if c.strip()])
        self.categories = sorted(set(cats))

        self.keyword_map: dict[str, set[str]] = {}
        if raw.shape[1] >= 2:
            for cat, keys in raw.iloc[:, :2].itertuples(index=False):
                keyset = {k.strip().lower() for k in splitter.split(str(keys)) if k and k.strip()}
                if keyset:
                    self.keyword_map[str(cat)] = keyset
        
        self._canon = {c.lower(): c for c in self.categories}
        self._slash_alias: dict[str, str] = {}
        for c in self.categories:
            if "/" in c:
                for part in c.split("/"):
                    p = part.strip().lower()
                    if p:
                        self._slash_alias[p] = c

        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=api_key)

    async def classify(self, description: str) -> list[str]:
        allowed = ", ".join(self.categories)
        prompt = (
            "Pick all matching ministry categories (comma-separated). "
            "Use the exact strings from the allowed list; do not invent new ones and do not split multi-word or slash categories. "
            "If a slash category applies (e.g., 'Funeral/Burial'), return that single token. "
            f"Allowed: {allowed}. If none fit, return 'Other'.\n"
            "Description:\n" + (description or "")[:800]
        )
        try:
            resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
            raw = [c.strip() for c in (resp.content or "").split(",")]
            out: set[str] = set()
            for token in raw:
                if not token:
                    continue
                # exact/ci match first
                ci = self._canon.get(token.lower())
                if ci:
                    out.add(ci); continue
                # map partials like "funeral" -> "Funeral/Burial"
                alias = self._slash_alias.get(token.lower())
                if alias:
                    out.add(alias); continue
            if out:
                return sorted(out)
            return [self._keyword_fallback(description)]
        except Exception:
            return [self._keyword_fallback(description)]

    def _keyword_fallback(self, description: str) -> str:
        text = (description or "").lower()
        for cat, keys in self.keyword_map.items():
            if keys and any(k in text for k in keys):
                return cat
        return "Other"


class Summariser:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=api_key)

    async def summary(self, text: str) -> str:
        prompt = (
            "Summarise the ministry in 1–2 sentences highlighting mission and target community.\n"
            "Text:\n" + (text or "")[:4000]
        )
        try:
            resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return (resp.content or "").strip()
        except Exception:
            return ""

    async def page_type(self, text: str, url: str) -> str:
        prompt = (
            "Classify the webpage into one of: 'ministry', 'listing', 'career', "
            "'donation', or 'other'. A ministry page describes a single "
            "ministry or program. A listing page mainly links to multiple "
            "ministries. Return only the label.\n"
            f"URL: {url}\nText:\n" + (text or "")[:2000]
        )
        try:
            resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
            label = (resp.content or "").strip().lower()
            if label not in {"ministry", "listing", "career", "donation"}:
                return "other"
            return label
        except Exception:
            return "other"

    async def resolve_org_and_title(
        self,
        page_text: str,
        url: str,
        default_org: str | None,
        default_title: str | None,
    ) -> dict:
        """
        Return {'organization_name': str, 'title': str, 'confidence': float, 'reason': str}
        - organization_name: umbrella org/church (never an aggregator like MapQuest)
        - title: specific ministry/program on THIS page (or page-level ministry if applicable)
        """
        prompt = f"""
You are extracting names for a directory of Christian ministries.

TASK
- Decide the true organization/church name (umbrella org).
- Decide the specific ministry title on THIS page (not a generic "Ministries" hub), if applicable.
- If the site is an aggregator (MapQuest, Facebook, Google Maps, Yelp, Linktree, etc.), DO NOT use the aggregator as the organization. Infer the real org from text/context.
- If only one ministry exists and it’s the same as the org, let title == organization.
- Prefer on-page names (headers/About/JSON-LD) over domain names. Avoid support/donate/job pages as titles.

INPUT
URL: {url}
Default org guess: {default_org or ""}
Default title guess: {default_title or ""}
PAGE TEXT (truncated):
{(page_text or "")[:2500]}

OUTPUT (JSON ONLY):
{{
  "organization_name": "<string>",
  "title": "<string>",
  "confidence": <float between 0 and 1>,
  "reason": "<short rationale>"
}}
"""
        try:
            import json as _json
            resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
            data = _json.loads((resp.content or "").strip())
            if not isinstance(data, dict):
                raise ValueError("Bad JSON from model")
            # Safe defaults
            data.setdefault("organization_name", default_org or default_title or "")
            data.setdefault("title", default_title or default_org or "")
            if not str(data.get("organization_name", "")).strip():
                data["organization_name"] = default_org or default_title or ""
            if not str(data.get("title", "")).strip():
                data["title"] = default_title or default_org or ""
            if not isinstance(data.get("confidence"), (int, float)):
                data["confidence"] = 0.0
            return data
        except Exception:
            return {
                "organization_name": default_org or default_title or "",
                "title": default_title or default_org or "",
                "confidence": 0.0,
                "reason": "fallback",
            }

# ---------- Search --------------------------------------------------------------------------
async def search_ministries(county: str, state_abbrev: str, limit: int) -> list[dict[str,str]]:
    query = f"site:.org \"{county} County\" {state_abbrev} Christian ministry"
    out: list[dict[str,str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=limit):
            out.append({"title": r["title"], "url": r["href"], "source": "ddg"})
    return out

# ---------- Toolkit (Playwright MCP / local) ------------------------------------------------
class Browser:
    def __init__(self):
        self.ws_url = os.getenv("PLAYWRIGHT_MCP_URL")
        self._toolkit: PlayWrightBrowserToolkit | None = None
        self._playwright = None
        self._browser = None
        self._task: asyncio.Task | None = asyncio.create_task(self._open())

    async def _open(self):
        self._playwright = await async_playwright().start()
        if self.ws_url:
            self._browser = await self._playwright.chromium.connect(self.ws_url)
        else:
            self._browser = await self._playwright.chromium.launch(headless=True)
        return self._browser

    async def toolkit(self) -> PlayWrightBrowserToolkit:
        if self._toolkit is None:
            browser = await self._task
            self._toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        return self._toolkit

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

# ---------- Ministry extraction --------------------------------------------------------------
MINISTRY_WORDS = ("ministry", "ministries", "program", "group", "recovery", "care", "support", "outreach")

def looks_like_ministry(title: str, url: str) -> bool:
    return bool(MINISTRY_REGEX.search(title) or MINISTRY_REGEX.search(url))


def extract_ministry_links(soup: BeautifulSoup, base: str) -> list[tuple[str,str]]:
    links: list[tuple[str,str]] = []
    for a in soup.find_all("a", href=True):
        text = (a.get_text(" ", strip=True) or "").lower()
        href = urljoin(base, a["href"]) if a["href"] else None
        if not href:
            continue
        same_site = urlparse(href).netloc == urlparse(base).netloc
        if same_site and any(w in text for w in MINISTRY_WORDS):
            links.append((text.title()[:120], href))
    # de-dup by URL
    seen: set[str] = set()
    out: list[tuple[str,str]] = []
    for t,u in links:
        if u not in seen:
            out.append((t,u)); seen.add(u)
    return out

# ---------- Scraper -------------------------------------------------------------------------
class MinistryScraper:
    def __init__(self, classifier: CategoryClassifier, summariser: Summariser, batch: int, concurr: int, max_pages: int):
        self.classifier, self.summariser = classifier, summariser
        self.batch_size, self.max_pages = batch, max_pages
        self.sema = asyncio.Semaphore(concurr)
        self.browser = Browser()
        self._seen_links: set[str] = set()

        if not CSV_PATH.exists():
            with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_HEADERS, quoting=csv.QUOTE_ALL).writeheader()

    async def scrape_ministry(self, meta: dict[str,str], state_abbrev: str) -> list[MinistryRecord]:
        async with self.sema:
            url, result_title = meta["url"], meta["title"]

            # Try MCP first; fall back to HTTP
            page_html: str | None = None
            try:
                tk = await self.browser.toolkit()
                tools = {t.name: t for t in tk.get_tools()}
                await tools["navigate_browser"].ainvoke({"url": url, "headers": {"User-Agent": UA}, "timeout": 15000})
                page_text = await tools["extract_text"].ainvoke({"selector": "body"})
                page_html = await fetch_html(url) or ""
                if not page_text.strip() and page_html:
                    page_text = visible_text(page_html)
            except Exception:
                page_html = await fetch_html(url)
                if not page_html:
                    return []
                page_text = visible_text(page_html)

            if not page_text.strip():
                return []

            page_kind = await self.summariser.page_type(page_text, url)
            if page_kind in {"career", "donation", "other"}:
                return []

            soup = BeautifulSoup(page_html or "", "html.parser")
            home_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            host = (urlparse(url).hostname or "").lower()
            is_agg = any(host.endswith(d) for d in AGGREGATOR_DOMAINS)
            page_title = (soup.title.string or "").strip() if soup.title else ""

            # Defaults for AI
            org_name_guess  = extract_org_name(soup, url, result_title)
            title_guess     = extract_ministry_title(soup, url, org_name_guess, result_title)

            # Context-rich input to the resolver
            page_text_for_ai = (
                f"SEARCH_RESULT_TITLE: {result_title or ''}\n"
                f"PAGE_TITLE: {page_title}\n"
                f"HOST: {host}\n"
                f"IS_AGGREGATOR: {is_agg}\n" + page_text
            )

            resolved = await self.summariser.resolve_org_and_title(
                page_text=page_text_for_ai,
                url=url,
                default_org=org_name_guess,
                default_title=title_guess,
            )
            org_name = (resolved.get("organization_name") or org_name_guess).strip()
            page_level_title = (resolved.get("title") or title_guess).strip()

            # Final safety: if aggregator slipped through, correct it
            core = (host.split(":")[0].split(".")[-2] if "." in host else host).lower()
            looks_like_agg = (
                core in AGGREGATOR_NAMES or
                any(n in org_name.lower() for n in AGGREGATOR_NAMES)
            )
            if is_agg and looks_like_agg:
                fixed = _clean_name_from_titles(page_title, result_title)
                if fixed:
                    org_name = fixed
                else:
                    org_name = org_name_guess if org_name_guess and org_name_guess.lower() not in AGGREGATOR_NAMES else (page_title or result_title or org_name)

            # Use AI to detect ministry listing pages and expand them
            sublinks = extract_ministry_links(soup, url) if page_kind == "listing" else []
            if page_kind == "listing" and not sublinks:
                return []

            records: list[MinistryRecord] = []
            if page_kind == "listing" and sublinks:
                for link_title, link_url in sublinks[:30]:  # cap fan-out per page
                    html2 = await fetch_html(link_url)
                    if not html2:
                        continue
                    text2 = visible_text(html2)
                    kind2 = await self.summariser.page_type(text2, link_url)
                    if kind2 != "ministry":
                        continue
                    soup2 = BeautifulSoup(html2, "html.parser")

                    # Page-specific guesses
                    org2_guess = org_name  # usually same umbrella org
                    title2_guess = extract_ministry_title(soup2, link_url, org2_guess, link_title) or link_title

                    # Resolve again for each sub-ministry page
                    page_title2 = (soup2.title.string or "").strip() if soup2.title else ""
                    host2 = (urlparse(link_url).hostname or "").lower()
                    is_agg2 = any(host2.endswith(d) for d in AGGREGATOR_DOMAINS)
                    page_text2_for_ai = (
                        f"SEARCH_RESULT_TITLE: {link_title or ''}\n"
                        f"PAGE_TITLE: {page_title2}\n"
                        f"HOST: {host2}\n"
                        f"IS_AGGREGATOR: {is_agg2}\n" + text2
                    )

                    resolved2 = await self.summariser.resolve_org_and_title(
                        page_text=page_text2_for_ai,
                        url=link_url,
                        default_org=org2_guess,
                        default_title=title2_guess,
                    )
                    org2 = (resolved2.get("organization_name") or org2_guess).strip()
                    title2 = (resolved2.get("title") or title2_guess).strip()

                    # Aggregator safety for subpages
                    core2 = (host2.split(":")[0].split(".")[-2] if "." in host2 else host2).lower()
                    looks_like_agg2 = (core2 in AGGREGATOR_NAMES or any(n in org2.lower() for n in AGGREGATOR_NAMES))
                    if is_agg2 and looks_like_agg2:
                        fixed2 = _clean_name_from_titles(page_title2, link_title)
                        if fixed2:
                            org2 = fixed2
                        else:
                            org2 = org_name

                    summary2 = await self.summariser.summary(text2)
                    cats2 = await self.classifier.classify(summary2 or text2[:600])
                    addr2 = extract_address_fields(html2)
                    phone2 = PHONE_REGEX.search(text2)
                    email2 = EMAIL_REGEX.search(text2)

                    rec = MinistryRecord(
                        organization_name=org2,
                        **{"Ministry Categories": cats2},
                        Title=title2,
                        link_to_ministry=link_url,
                        phone=(phone2.group(0) if phone2 else None),
                        Content=summary2,
                        email=(email2.group(0) if email2 else None),
                        address=addr2.get("address"),
                        city=addr2.get("city"),
                        state=addr2.get("state") or state_abbrev,
                        zip=addr2.get("zip"),
                        country="USA",
                        source_site=home_url,
                    )
                    records.append(rec)
                return records

            # Single-ministry page
            summary = await self.summariser.summary(page_text)
            cats = await self.classifier.classify(summary or page_text[:600])
            addr = extract_address_fields(page_html or "")
            phone = PHONE_REGEX.search(page_text)
            email = EMAIL_REGEX.search(page_text)

            rec = MinistryRecord(
                organization_name=org_name,
                **{"Ministry Categories": cats},
                Title=page_level_title,
                link_to_ministry=url,
                phone=(phone.group(0) if phone else None),
                Content=summary,
                email=(email.group(0) if email else None),
                address=addr.get("address"),
                city=addr.get("city"),
                state=addr.get("state") or state_abbrev,
                zip=addr.get("zip"),
                country="USA",
                source_site=home_url,
            )
            return [rec]

    async def scrape_county(self, county: str, state_abbrev: str) -> AsyncIterator[MinistryRecord]:
        seen: set[str] = set()
        for page in range(self.max_pages):
            batch = await search_ministries(county, state_abbrev, limit=self.batch_size)
            new = [b for b in batch
                   if b["url"] not in seen
                   and looks_like_ministry(b["title"], b["url"])
                   and not b["url"].lower().endswith((".pdf",".doc",".docx",".ppt",".pptx"))
                   and not await is_download(b["url"])]
            if not new:
                break
            seen.update(b["url"] for b in new)
            for m in new:
                print(f"        ↳ queueing {m['url']}")
            tasks = [self.scrape_ministry(meta, state_abbrev) for meta in new]
            for coro in asyncio.as_completed(tasks):
                recs = await coro
                for rec in recs:
                    yield rec

    def _write_record(self, record: MinistryRecord):
        # Deduplicate by link_to_ministry
        if record.link_to_ministry in self._seen_links:
            return
        self._seen_links.add(record.link_to_ministry)

        with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=CSV_HEADERS,
                quoting=csv.QUOTE_ALL,
                extrasaction="ignore",
                restval="",
            )
            row = record.model_dump(by_alias=True)
            cats = row.get("Ministry Categories")
            if isinstance(cats, (list, tuple, set)):
                row["Ministry Categories"] = "| ".join(cats)
            writer.writerow(row)
        print(f"            ✔ wrote {record.org_name[:60]}…")

    async def scrape_state(self, state_name: str):
        state_abbrev = get_state_abbrev(state_name)
        counties = COUNTY_MAP.get(state_abbrev, [])
        for county in counties:
            print(f"    … {county}")
            async for rec in self.scrape_county(county, state_abbrev):
                self._write_record(rec)

    async def close(self):
        await self.browser.close()
        if _async_session and not _async_session.closed:
            await _async_session.close()

# ---------- CLI / main ----------------------------------------------------------------------
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Christian ministry directory scraper")
    p.add_argument("--states", nargs="*", help="List of states to scrape (default all)")
    p.add_argument("--start-state", help="Resume at this state abbreviation (e.g. AL)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--concurrency", type=int, default=CONCURRENCY_DEFAULT)
    p.add_argument("--max-pages", type=int, default=MAX_PAGES_DEFAULT)
    p.add_argument("--test-url", help="Debug a single URL and print resolved names")
    return p.parse_args()

async def main():
    args = parse_args()

    # Quick single-URL debug path
    if args.test_url:
        classifier = CategoryClassifier(CATEGORIES_PATH)
        summariser = Summariser()
        scraper = MinistryScraper(classifier, summariser, 1, 1, 1)
        meta = {"url": args.test_url, "title": "(manual)"}
        recs = await scraper.scrape_ministry(meta, state_abbrev="AL")
        for r in recs:
            print("DEBUG →", r.model_dump(by_alias=True))
        await scraper.close()
        return

    all_states = [s.name for s in us.states.STATES]
    state_list = args.states or all_states
    if args.start_state:
        started = False
        filtered: list[str] = []
        for s in state_list:
            if not started and get_state_abbrev(s) != args.start_state:
                continue
            started = True
            filtered.append(s)
        state_list = filtered

    classifier = CategoryClassifier(CATEGORIES_PATH)
    summariser = Summariser()
    scraper = MinistryScraper(classifier, summariser, args.batch_size, args.concurrency, args.max_pages)

    try:
        for state in state_list:
            print(f"\n=== Processing {state} ===")
            await scraper.scrape_state(state)
    finally:
        await scraper.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user – exiting gracefully.")
