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
from urllib.parse import urljoin, urlparse, parse_qs, unquote_plus

import aiohttp
import pandas as pd
import us
import usaddress
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
import time
from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright
from langchain_community.agent_toolkits.playwright import PlayWrightBrowserToolkit
from langchain.agents import AgentExecutor, create_openai_functions_agent

load_dotenv()

# ---------- OpenAI API usage tracking -------------------------------------------------------
OPENAI_MAX_RPM = int(os.getenv("OPENAI_MAX_RPM", "0")) or None
OPENAI_MAX_TPM = int(os.getenv("OPENAI_MAX_TPM", "0")) or None


class OpenAIUsageTracker:
    def __init__(self, max_rpm: int | None, max_tpm: int | None):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self._lock = asyncio.Lock()
        self._reset()

    def _reset(self) -> None:
        self._start = time.monotonic()
        self._requests = 0
        self._tokens = 0

    async def _maybe_sleep(self, now: float) -> None:
        elapsed = now - self._start
        if (
            (self.max_rpm and self._requests >= self.max_rpm)
            or (self.max_tpm and self._tokens >= self.max_tpm)
        ):
            wait = 60 - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._reset()

    async def before(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now - self._start >= 60:
                self._reset()
            await self._maybe_sleep(now)

    async def after(self, tokens: int) -> None:
        async with self._lock:
            now = time.monotonic()
            if now - self._start >= 60:
                self._reset()
            self._requests += 1
            self._tokens += tokens
            await self._maybe_sleep(now)


OPENAI_USAGE = OpenAIUsageTracker(OPENAI_MAX_RPM, OPENAI_MAX_TPM)


class UsageCallbackHandler(AsyncCallbackHandler):
    def __init__(self, tracker: OpenAIUsageTracker):
        self.tracker = tracker

    async def on_llm_start(self, *args, **kwargs):  # type: ignore[override]
        await self.tracker.before()

    async def on_llm_end(self, response, **kwargs):  # type: ignore[override]
        usage = getattr(response, "llm_output", {}) or {}
        usage = usage.get("token_usage", {})
        total = usage.get("total_tokens") or (
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        )
        await self.tracker.after(int(total or 0))


USAGE_HANDLER = UsageCallbackHandler(OPENAI_USAGE)

# ---------- Files & constants ---------------------------------------------------------------
OUTPUT_DIR = Path("out"); OUTPUT_DIR.mkdir(exist_ok=True)
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
    "linktr.ee", "linktree.com", "wikipedia.org", "wikipedia.com"
}
AGGREGATOR_NAMES = {
    "mapquest","facebook","yelp","google","bing","duckduckgo","eventbrite","meetup",
    "paypal","pushpay","donorbox","square","linktree","instagram","twitter","x","linkedin", "wikipedia"
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

# ---------- Address extraction (AI-boosted v2) ----------------------------------------------

ADDRESS_BLOCK_RE = re.compile(r"(?:^|[\n\r])\s*(?:address|visit (?:us|our)|location|contact)\b[:\s]*", re.I)

STREET_PATTERN = r"""
(?P<street>
  (?:
    \d{1,6}\s+[A-Za-z0-9.\-']+(?:\s+[A-Za-z0-9.\-']+)*
    \s+(?:Ave(?:nue)?|St(?:reet)?|Rd(?:oad)?|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Hwy|Highway|Way|Pkwy|Parkway|Pl|Place|Terrace|Ter|Cir|Circle)
    (?:\s+(?:N|S|E|W|NE|NW|SE|SW))?
    (?:\s*(?:Unit|Ste\.?|Suite|\#)\s*[A-Za-z0-9\-]+)?
  )
)
"""
CITY_STATE_ZIP_PATTERN = r"(?P<city>[A-Za-z .'\-]+)\s*,\s*(?P<state>[A-Z]{2})\s+(?P<zip>\d{5}(?:-\d{4})?)"

STREET_CITYZIP_RE = re.compile(rf"{STREET_PATTERN}\s*,\s*{CITY_STATE_ZIP_PATTERN}", re.I | re.X)
POBOX_CITYZIP_RE = re.compile(r"""
(?P<street>P(?:\.|\s*)O(?:\.|\s*)\s*Box\s+\d{1,8})\s*,?\s*
(?P<city>[A-Za-z .'\-]+)\s*,\s*
(?P<state>[A-Z]{2})\s+
(?P<zip>\d{5}(?:-\d{4})?)
""", re.I | re.X)

def _iter_address_matches(text: str):
    for m in STREET_CITYZIP_RE.finditer(text):
        yield m.groupdict()
    for m in POBOX_CITYZIP_RE.finditer(text):
        yield m.groupdict()

def _us_norm(state: str | None) -> str | None:
    if not state: return None
    st = us.states.lookup(state.strip())
    return st.abbr if st else None

def _mk_addr(street, city, state, zip_):
    return {
        "address": (street or "").strip() or None,
        "city": (city or "").strip() or None,
        "state": _us_norm(state),
        "zip": (zip_ or "").strip() or None,
    }

def _is_good(c: dict) -> bool:
    # good enough if at least city+state; better if full
    return bool(c.get("city") and c.get("state"))

def _find_postal_dicts(obj) -> list[dict]:
    out = []
    if isinstance(obj, dict):
        if obj.get("@type") in ("PostalAddress","schema:PostalAddress") or \
           {"streetAddress","addressLocality","addressRegion","postalCode"} & set(obj.keys()):
            out.append(obj)
        for v in obj.values():
            out.extend(_find_postal_dicts(v))
    elif isinstance(obj, list):
        for it in obj:
            out.extend(_find_postal_dicts(it))
    return out

def _candidates_from_jsonld(soup: BeautifulSoup) -> list[dict]:
    cands = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue
        for a in _find_postal_dicts(data):
            cands.append(_mk_addr(a.get("streetAddress"), a.get("addressLocality"), a.get("addressRegion"), a.get("postalCode")))
    return cands

def _candidates_from_vcard_microdata(soup: BeautifulSoup) -> list[dict]:
    cands = []
    for block in soup.select(".adr, [itemtype*='PostalAddress'], [itemscope][itemtype*='schema.org/PostalAddress']"):
        street_el = block.select_one(".street-address,[itemprop='streetAddress']")
        city_el   = block.select_one(".locality,[itemprop='addressLocality']")
        state_el  = block.select_one(".region,[itemprop='addressRegion']")
        zip_el    = block.select_one(".postal-code,[itemprop='postalCode']")
        street = street_el.get_text(strip=True) if street_el else None
        city   = city_el.get_text(strip=True) if city_el else None
        state  = state_el.get_text(strip=True) if state_el else None
        zip_   = zip_el.get_text(strip=True) if zip_el else None
        cands.append(_mk_addr(street, city, state, zip_))
    return cands

def _candidates_from_address_tags(soup: BeautifulSoup) -> list[dict]:
    cands = []
    for tag in soup.find_all("address"):
        txt = tag.get_text(" ", strip=True)
        try:
            parsed, _ = usaddress.tag(txt)
            cand = _mk_addr(
                f"{parsed.get('AddressNumber','')} {parsed.get('StreetName','')} {parsed.get('StreetNamePostType','')}".strip(),
                parsed.get("PlaceName"), parsed.get("StateName"), parsed.get("ZipCode"),
            )
            if _is_good(cand):
                cands.append(cand)
        except usaddress.RepeatedLabelError:
            pass
        for gd in _iter_address_matches(txt):
            cands.append(_mk_addr(gd.get("street"), gd.get("city"), gd.get("state"), gd.get("zip")))
    return cands

def _candidates_from_visible_text(text: str) -> list[dict]:
    cands = []
    for gd in _iter_address_matches(text):
        cands.append(_mk_addr(gd.get("street"), gd.get("city"), gd.get("state"), gd.get("zip")))
    return cands

def _candidates_from_map_links(soup: BeautifulSoup) -> list[dict]:
    """Extract address from Google/Apple Maps links (q=...)."""
    cands = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "google.com/maps" in href or "maps.google.com" in href or "maps.apple.com" in href:
            try:
                q = urlparse(href).query
                params = parse_qs(q)
                raw = None
                if "q" in params and params["q"]:
                    raw = unquote_plus(params["q"][0])
                elif "destination" in params and params["destination"]:
                    raw = unquote_plus(params["destination"][0])
                elif "daddr" in params and params["daddr"]:
                    raw = unquote_plus(params["daddr"][0])
                if not raw:
                    continue
                raw = re.sub(r"\s+", " ", raw).strip()
                try:
                    parsed, _ = usaddress.tag(raw)
                    cand = _mk_addr(
                        " ".join([parsed.get("AddressNumber",""), parsed.get("StreetName",""), parsed.get("StreetNamePostType","")]).strip(),
                        parsed.get("PlaceName"), parsed.get("StateName"), parsed.get("ZipCode"),
                    )
                    if _is_good(cand):
                        cands.append(cand)
                except usaddress.RepeatedLabelError:
                    for gd in _iter_address_matches(raw):
                        cands.append(_mk_addr(gd.get("street"), gd.get("city"), gd.get("state"), gd.get("zip")))
            except Exception:
                continue
    return cands

def _candidates_from_line_windows(text: str, max_width: int = 3) -> list[dict]:
    """Slide windows of up to 3 lines; run usaddress on joined text. Captures line-broken addresses."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cands = []
    n = len(lines)
    for w in (2, 3):
        if w > max_width: break
        for i in range(n - w + 1):
            chunk = " ".join(lines[i:i+w])
            if not re.search(r"\b[A-Z]{2}\b\s+\d{5}", chunk) and not re.search(r"\b[A-Z]{2}\b", chunk):
                continue
            try:
                parsed, _ = usaddress.tag(chunk)
                cand = _mk_addr(
                    " ".join([parsed.get("AddressNumber",""), parsed.get("StreetName",""), parsed.get("StreetNamePostType","")]).strip(),
                    parsed.get("PlaceName"), parsed.get("StateName"), parsed.get("ZipCode"),
                )
                if _is_good(cand):
                    cands.append(cand)
            except usaddress.RepeatedLabelError:
                pass
    return cands

def _score_candidate(c: dict, text: str, state_hint: str | None) -> float:
    score = 0.0
    if c.get("zip"): score += 2.0
    if c.get("state"): score += 2.0
    if c.get("city"): score += 1.0
    if c.get("address"): score += 1.0
    if state_hint and c.get("state") == state_hint:
        score += 2.5
    if c.get("address"):
        addr = c["address"]
        for m in ADDRESS_BLOCK_RE.finditer(text):
            start = m.start()
            window = text[max(0, start-300): start+800]
            if addr.lower()[:10] in window.lower():
                score += 2.0
                break
    if c.get("address") and "box" in c["address"].lower():
        score -= 1.5
    return score

# Lazy-inited shared LLM for addresses
_ADDRESS_LLM: ChatOpenAI | None = None

def _address_llm() -> ChatOpenAI:
    global _ADDRESS_LLM
    if _ADDRESS_LLM is None:
        _ADDRESS_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"), callbacks=[USAGE_HANDLER],)
    return _ADDRESS_LLM

async def _ai_choose_best_address(candidates: list[dict], page_text: str, state_hint: str | None) -> dict | None:
    filtered = [c for c in candidates if c.get("city") and (not state_hint or not c.get("state") or c.get("state") == state_hint or len(c.get("state",""))==2)]
    if not filtered:
        return None
    seen = set(); uniq = []
    for c in filtered:
        key = (c.get("address"), c.get("city"), c.get("state"), c.get("zip"))
        if key in seen: continue
        uniq.append(c); seen.add(key)
    uniq = uniq[:12]
    prompt = (
        "Select the single best PHYSICAL street address for the ministry on this page.\n"
        "Rules:\n"
        "- Prefer a full street address over a PO Box; use PO Box only if no street address.\n"
        "- Prefer addresses near Contact/Visit/Location sections or footer.\n"
        "- Prefer an address matching the state hint if present.\n"
        "- Ignore unrelated venues or map platform addresses.\n"
        "- Output JSON ONLY with keys: address, city, state, zip. State must be 2-letter if available.\n"
        f"State hint: {state_hint or ''}\n"
        f"\nCANDIDATES:\n{json.dumps(uniq, ensure_ascii=False)}\n"
        f"\nPAGE TEXT (truncated):\n{page_text[:2500]}"
    )
    try:
        resp = await _address_llm().ainvoke([HumanMessage(content=prompt)])
        data = json.loads((resp.content or "").strip())
        if isinstance(data, dict):
            out = _mk_addr(data.get("address"), data.get("city"), data.get("state"), data.get("zip"))
            if _is_good(out):
                return out
    except Exception:
        pass
    return None

async def _ai_extract_from_text(page_text: str, state_hint: str | None) -> dict | None:
    prompt = (
        "From the following text, extract the best PHYSICAL mailing address for the ministry/church.\n"
        "Prefer street addresses; use PO Box only if no street address exists. Prefer matches to the state hint.\n"
        "Output JSON ONLY with keys: address, city, state, zip. If zip is missing, you may omit it.\n"
        f"State hint: {state_hint or ''}\n\nTEXT:\n{page_text[:4000]}"
    )
    try:
        resp = await _address_llm().ainvoke([HumanMessage(content=prompt)])
        data = json.loads((resp.content or "").strip())
        if isinstance(data, dict):
            out = _mk_addr(data.get("address"), data.get("city"), data.get("state"), data.get("zip"))
            if _is_good(out):
                return out
    except Exception:
        pass
    return None

async def extract_address_fields(html: str, page_text: str | None = None, state_hint: str | None = None) -> dict:
    """
    AI-boosted extractor with multiple harvesters + AI arbitration.
    Returns {address, city, state, zip} (zip may be None) or {}.
    """
    if not html and not page_text:
        return {}
    soup = BeautifulSoup(html or "", "html.parser")
    text = (page_text or visible_text(html or ""))[:16000]

    candidates = []
    candidates += _candidates_from_jsonld(soup)
    candidates += _candidates_from_vcard_microdata(soup)
    candidates += _candidates_from_address_tags(soup)
    candidates += _candidates_from_map_links(soup)
    candidates += _candidates_from_visible_text(text)
    candidates += _candidates_from_line_windows(text)

    # normalize & dedup
    normed, seen = [], set()
    for c in candidates:
        n = _mk_addr(c.get("address"), c.get("city"), c.get("state"), c.get("zip"))
        if not any(v for v in n.values()): 
            continue
        key = (n.get("address"), n.get("city"), n.get("state"), n.get("zip"))
        if key in seen: 
            continue
        seen.add(key); normed.append(n)

    if os.getenv("ADDR_DEBUG") == "1":
        print("ADDR_CANDIDATES_LOCAL", normed[:8])

    if not normed:
        # last-ditch: direct AI from text
        direct = await _ai_extract_from_text(text, state_hint)
        return direct or {}

    ranked = sorted(normed, key=lambda c: _score_candidate(c, text, state_hint), reverse=True)

    # If clear winner by score, return it
    def sc(i): return _score_candidate(ranked[i], text, state_hint)
    if len(ranked) == 1 or sc(0) - sc(1) >= 1.5:
        return ranked[0]

    # Otherwise ask AI to pick the best
    chosen = await _ai_choose_best_address(ranked[:12], text, state_hint)
    if chosen:
        return chosen
    return ranked[0]

# ---------- Search-based Address Enrichment (DDG + AI) -------------------------------------
SEARCH_ADDR_QUERIES = [
    "site:{domain} contact",
    "site:{domain} visit",
    "site:{domain} location",
    "site:{domain} directions",
    "site:{domain} find us",
    '"{org}" address {state}',
    '{org} address {state}',
    '"{org}" address',
]

ADDR_AGGREGATORS = (
    "google.com/maps", "maps.google.com", "maps.apple.com",
    "bing.com/maps", "yelp.com", "mapquest.com",
)

CONTACT_SLUGS = [
    "/contact", "/contact-us", "/contactus", "/visit", "/visit-us",
    "/plan-a-visit", "/im-new", "/location", "/locations",
    "/find-us", "/directions", "/about", "/about-us",
]

async def _ddg_candidate_urls(org_name: str, home_url: str, state_hint: str | None, max_urls: int = 8) -> list[str]:
    base = f"{urlparse(home_url).scheme}://{urlparse(home_url).netloc}"
    host = urlparse(home_url).netloc
    domain = host
    queries = [q.format(domain=domain, org=org_name, state=(state_hint or "")) for q in SEARCH_ADDR_QUERIES]

    seen, out = set(), []

    # Proactively try common contact slugs on the same domain
    for slug in CONTACT_SLUGS:
        u = urljoin(base, slug)
        if u not in seen:
            seen.add(u); out.append(u)

    # DuckDuckGo search for on-site pages and map/dir pages
    try:
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=5):
                    u = r.get("href")
                    if not u: continue
                    netloc = urlparse(u).netloc
                    if (netloc == host) or any(agg in u for agg in ADDR_AGGREGATORS):
                        if u not in seen:
                            seen.add(u); out.append(u)
                    if len(out) >= max_urls:
                        break
                if len(out) >= max_urls:
                    break
    except Exception:
        pass

    return out[:max_urls]

async def enrich_address_via_duckduckgo(org_name: str, home_url: str, state_hint: str | None) -> dict:
    """
    Returns {address, city, state, zip} or {}.
    Strategy:
      1) discover candidate pages via contact slugs + DDG
      2) harvest address candidates (JSON-LD, <address>, map links, line windows)
      3) rank heuristically; if ambiguous, ask AI to choose best
    """
    candidate_urls = await _ddg_candidate_urls(org_name, home_url, state_hint, max_urls=8)

    # Collect candidates across pages
    all_candidates: list[dict] = []
    texts_for_ai: list[str] = []
    for u in candidate_urls:
        html = await fetch_html(u)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        text = visible_text(html)[:6000]
        texts_for_ai.append(f"[{u}]\n{text[:3000]}")

        all_candidates += _candidates_from_jsonld(soup)
        all_candidates += _candidates_from_address_tags(soup)
        all_candidates += _candidates_from_map_links(soup)
        all_candidates += _candidates_from_visible_text(text)
        all_candidates += _candidates_from_line_windows(text)

    # Normalize + dedupe
    normed, seen = [], set()
    for c in all_candidates:
        n = _mk_addr(c.get("address"), c.get("city"), c.get("state"), c.get("zip"))
        if not any(v for v in n.values()): 
            continue
        key = (n.get("address"), n.get("city"), n.get("state"), n.get("zip"))
        if key in seen: 
            continue
        seen.add(key); normed.append(n)

    if os.getenv("ADDR_DEBUG") == "1":
        print("ADDR_DDG_URLS", candidate_urls)
        print("ADDR_DDG_CANDIDATES", normed[:8])

    if not normed:
        # last-ditch: direct AI over concatenated texts
        joined = "\n\n".join(texts_for_ai)[:12000]
        direct = await _ai_extract_from_text(joined, state_hint)
        return direct or {}

    joined_text = "\n\n".join(texts_for_ai)[:12000]
    ranked = sorted(normed, key=lambda c: _score_candidate(c, joined_text, state_hint), reverse=True)

    def sc(i): return _score_candidate(ranked[i], joined_text, state_hint)
    if len(ranked) == 1 or sc(0) - sc(1) >= 1.5:
        return ranked[0]

    chosen = await _ai_choose_best_address(ranked[:12], joined_text, state_hint)
    if chosen:
        return chosen
    return ranked[0]

async def search_google_maps_address(ministry_name: str, county: str, state_abbrev: str) -> dict:
    """Search Google Maps (via DDG) for a ministry's address.

    Uses query pattern "(Ministry) (county) (state abbreviation) address" and
    extracts structured components from Google Maps URLs or general snippets.

    Returns a dict with keys ``address``, ``city``, ``state``, ``zip`` or an empty
    dict when no match is found.
    """
    if not ministry_name or not isinstance(ministry_name, str):
        print("Warning: Invalid ministry name for address search")
        return {}

    clean_name = ministry_name.strip()
    if len(clean_name) > 100:
        clean_name = clean_name[:100].rsplit(" ", 1)[0]

    query = f'"{clean_name}" {county} County {state_abbrev} address'

    try:
        with DDGS() as ddgs:
            maps_query = (
                f'site:google.com/maps OR site:maps.google.com "{clean_name}" '
                f'{county} County {state_abbrev}'
            )
            results = ddgs.text(maps_query, max_results=3)
            for result in results:
                url = result.get("href", "")
                if "google.com/maps" in url or "maps.google.com" in url:
                    parsed = urlparse(url)
                    params = parse_qs(parsed.query)
                    address_text = None
                    if "q" in params:
                        address_text = unquote_plus(params["q"][0])
                    elif "daddr" in params:
                        address_text = unquote_plus(params["daddr"][0])
                    elif "destination" in params:
                        address_text = unquote_plus(params["destination"][0])
                    if address_text:
                        pattern = (
                            r"(\d+\s+[^,]+),\s*([^,]+),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)"
                        )
                        match = re.search(pattern, address_text)
                        if match:
                            return {
                                "address": match.group(1).strip(),
                                "city": match.group(2).strip(),
                                "state": match.group(3),
                                "zip": match.group(4),
                            }

            general_results = ddgs.text(query, max_results=5)
            for result in general_results:
                snippet = result.get("body", "")
                pattern = (
                    r"(\d+\s+[^,]+),\s*([^,]+),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)"
                )
                match = re.search(pattern, snippet)
                if match:
                    street = match.group(1).strip()
                    if re.match(r'^\d+\s+', street) and len(street) < 100:
                        return {
                            "address": street,
                            "city": match.group(2).strip(),
                            "state": match.group(3),
                            "zip": match.group(4),
                        }

                city_state_zip = r"([A-Za-z\s]+),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)"
                match2 = re.search(city_state_zip, snippet)
                if match2:
                    street_pattern = (
                        r"(\d+\s+[A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|"
                        r"Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl|Circle|Cir|Parkway|Pkwy))"
                    )
                    street_match = re.search(
                        street_pattern, snippet[: match2.start()], re.I
                    )
                    if street_match:
                        return {
                            "address": street_match.group(1).strip(),
                            "city": match2.group(1).strip(),
                            "state": match2.group(2),
                            "zip": match2.group(3),
                        }

    except Exception as e:
        print(f"Error searching Google Maps for '{clean_name}': {str(e)[:200]}")

    return {}

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
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=api_key, callbacks=[USAGE_HANDLER],)

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
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=api_key, callbacks=[USAGE_HANDLER],)

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
        self._task: asyncio.Task | None = asyncio.create_task(self._open())
        self._toolkit: PlayWrightBrowserToolkit | None = None

    async def _open(self):
        pw = await async_playwright().start()
        if self.ws_url:
            return await pw.chromium.connect(self.ws_url)
        return await pw.chromium.launch(headless=True)

    async def toolkit(self) -> PlayWrightBrowserToolkit:
        if self._toolkit is None:
            browser = await self._task
            self._toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        return self._toolkit

async def agentic_address_search(url: str, state_hint: str | None, browser: Browser) -> dict:
    """
    Use an LLM with Playwright tools to locate a ministry's mailing address.
    Returns {address, city, state, zip} or {}.
    """
    try:
        tk = await browser.toolkit()
        tools = tk.get_tools()
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            callbacks=[USAGE_HANDLER],
        )
        agent = create_openai_functions_agent(llm, tools)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        prompt = (
            f"Navigate to {url} and find the ministry's physical mailing address. "
            f"State hint: {state_hint or ''}. "
            "Return JSON ONLY with keys: address, city, state, zip. "
            "If not found, return an empty JSON object."
        )
        res = await executor.ainvoke({"input": prompt})
        data = json.loads((res.get("output") or "").strip())
        if isinstance(data, dict):
            out = _mk_addr(data.get("address"), data.get("city"), data.get("state"), data.get("zip"))
            if _is_good(out):
                return out
    except Exception:
        pass
    return {}

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
    def __init__(self, classifier: CategoryClassifier, summariser: Summariser, batch: int, concurr: int, max_pages: int, csv_path: Path,):
        self.classifier, self.summariser = classifier, summariser
        self.batch_size, self.max_pages = batch, max_pages
        self.sema = asyncio.Semaphore(concurr)
        self.browser = Browser()
        self.csv_path = csv_path
        self._seen_links: set[str] = set()

        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_HEADERS, quoting=csv.QUOTE_ALL).writeheader()

    async def scrape_ministry(self, meta: dict[str,str], state_abbrev: str, county: str = "") -> list[MinistryRecord]:
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

            # Detect index of ministries; explode to individual entries
            sublinks = extract_ministry_links(soup, url)
            is_index_like = ("/ministr" in (urlparse(url).path.lower())) or (len(sublinks) >= 3)

            records: list[MinistryRecord] = []
            if is_index_like and sublinks:
                for link_title, link_url in sublinks[:30]:  # cap fan-out per page
                    html2 = await fetch_html(link_url)
                    if not html2:
                        continue
                    text2 = visible_text(html2)
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

                    summary2 = await self.summariser.summary(text2)
                    cats2 = await self.classifier.classify(summary2 or text2[:600])
                    addr2 = await agentic_address_search(link_url, state_abbrev, self.browser)
                    if not (addr2.get("city") and addr2.get("state")):
                        addr2 = await search_google_maps_address(org2, county, state_abbrev)
                    if not (addr2.get("city") and addr2.get("state")):
                        addr2 = await extract_address_fields(html2, text2, state_abbrev)
                    if not (addr2.get("city") and addr2.get("state")):
                        addr2 = await enrich_address_via_duckduckgo(org2, home_url, state_abbrev)

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
            addr = await agentic_address_search(url, state_abbrev, self.browser)
            if not (addr.get("city") and addr.get("state")):
                addr = await search_google_maps_address(org_name, county, state_abbrev)
            if not (addr.get("city") and addr.get("state")):
                addr = await extract_address_fields(page_html or "", page_text, state_abbrev)
            if not (addr.get("city") and addr.get("state")):
                addr = await enrich_address_via_duckduckgo(org_name, home_url, state_abbrev)
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

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
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
        if _async_session and not _async_session.closed:
            await _async_session.close()

# ---------- CLI / main ----------------------------------------------------------------------
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Christian ministry directory scraper")
    p.add_argument("--state", required=True, help="State to scrape (name or abbreviation)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--concurrency", type=int, default=CONCURRENCY_DEFAULT)
    p.add_argument("--max-pages", type=int, default=MAX_PAGES_DEFAULT)
    p.add_argument("--test-url", help="Debug a single URL and print resolved names")
    return p.parse_args()

async def main():
    args = parse_args()

    state = args.state
    state_abbrev = get_state_abbrev(state)
    csv_path = OUTPUT_DIR / f"ministry_directory_{state_abbrev}.csv"
    # Quick single-URL debug path
    if args.test_url:
        classifier = CategoryClassifier(CATEGORIES_PATH)
        summariser = Summariser()
        scraper = MinistryScraper(classifier, summariser, 1, 1, 1, csv_path)
        meta = {"url": args.test_url, "title": "(manual)"}
        recs = await scraper.scrape_ministry(meta, state_abbrev=state_abbrev)
        for r in recs:
            print("DEBUG →", r.model_dump(by_alias=True))
        await scraper.close()
        return


    classifier = CategoryClassifier(CATEGORIES_PATH)
    summariser = Summariser()
    scraper = MinistryScraper(classifier, summariser, args.batch_size, args.concurrency, args.max_pages, csv_path)

    try:
        print(f"\n=== Processing {state} ===")
        await scraper.scrape_state(state)
    finally:
        await scraper.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user – exiting gracefully.")
