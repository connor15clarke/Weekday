#!/usr/bin/env python3
"""
weekdayfinal4.py
================

Fully revised Christian ministry scraper with:
- Date-pattern URL filtering (+ --allow-dated-urls)
- Single-hub Ministries follow
- Relaxed banned-cue heuristic
- SERP negative domain/keyword filter
- .html/.htm allowed
- Detailed skip logging
- Continuous mode across states (+ --continuous)
- Optional post-run cleaning skip (--no-clean)
- NEW: **Population-scaled pages per county** (--scale-pages-by-pop)

How population scaling works
----------------------------
By default, `--max-pages` is a per-county cap. With `--scale-pages-by-pop`, we
*override* the per-state `max_pages` using this formula:

    scaled = base_max_pages
             * (state_population / pop_baseline)
             * (ref_counties / num_counties_in_state)

Then clamp:  scaled ∈ [min_pages_per_county, max_pages_cap].

Defaults:
- base_max_pages = --max-pages (your current default per county, e.g. 5)
- pop_baseline   = median population across states in STATE_POP (you can override)
- ref_counties   = 50 (typical scale reference)
- min_pages_per_county = 3
- max_pages_cap  = 30

This makes low-county, high-pop states (e.g., AZ) crawl deeper per county, while
high-county, low-pop states (e.g., AR) crawl less per county—bringing totals closer
to a population-proportional coverage.

Usage examples
--------------
# AZ: scale by population, cap per county pages at 30
python weekdayfinal4.py --state AZ --scale-pages-by-pop --max-pages 5 --max-pages-cap 30

# Continuous across states, no cleaning, auto population scaling
python weekdayfinal4.py --continuous --scale-pages-by-pop --no-clean

# Custom baseline & reference counties
python weekdayfinal4.py --state AZ --scale-pages-by-pop --pop-baseline 4500000 --ref-counties 50

# Allow dated URLs if you want to capture program posts
python weekdayfinal4.py --state AZ --scale-pages-by-pop --allow-dated-urls
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import AsyncIterator, Iterable, Optional
from urllib.parse import urljoin, urlparse
from statistics import median

import aiohttp
import pandas as pd
import us
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright
from langchain_community.agent_toolkits.playwright import PlayWrightBrowserToolkit

# Project helpers
from weekday_scraper.config import load_settings
from weekday_scraper.rate import RateManager
from weekday_scraper.store import Store
from weekday_scraper.types import OrgResolution
from weekday_scraper.search.google_search import google_search_ministries
from weekday_scraper.places.geocoding import (
    resolve_org_address as g_resolve_org_address,
    parse_ministry_page_address as g_parse_ministry_page_address,
)

load_dotenv()

# ---------- OpenAI usage tracking ------------------------------------------------------------
OPENAI_MAX_RPM = int(os.getenv("OPENAI_MAX_RPM", "0")) or None
OPENAI_MAX_TPM = int(os.getenv("OPENAI_MAX_TPM", "0")) or None

class OpenAIUsageTracker:
    def __init__(self, max_rpm: int | None, max_tpm: int | None):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self._lock = asyncio.Lock()
        self._reset()

    def _reset(self):
        self._start = time.monotonic()
        self._requests = 0
        self._tokens = 0

    async def _maybe_sleep(self, now: float):
        elapsed = now - self._start
        if ((self.max_rpm and self._requests >= self.max_rpm) or
            (self.max_tpm and self._tokens >= self.max_tpm)):
            wait = 60 - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._reset()

    async def before(self):
        async with self._lock:
            now = time.monotonic()
            if now - self._start >= 60:
                self._reset()
            await self._maybe_sleep(now)

    async def after(self, tokens: int):
        async with self._lock:
            now = time.monotonic()
            if now - self._start >= 60:
                self._reset()
            self._requests += 1
            self._tokens += tokens
            await self._maybe_sleep(now)

USAGE = OpenAIUsageTracker(OPENAI_MAX_RPM, OPENAI_MAX_TPM)

class UsageCallbackHandler(AsyncCallbackHandler):
    def __init__(self, tracker: OpenAIUsageTracker):
        self.tracker = tracker
    async def on_llm_start(self, *_, **__):
        await self.tracker.before()
    async def on_llm_end(self, response, **__):
        usage = getattr(response, "llm_output", {}) or {}
        usage = usage.get("token_usage", {})
        total = usage.get("total_tokens") or (
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        )
        await self.tracker.after(int(total or 0))

CALLBACK = UsageCallbackHandler(USAGE)

# ---------- Files & constants ----------------------------------------------------------------
OUTPUT_DIR = Path("out"); OUTPUT_DIR.mkdir(exist_ok=True)
CATEGORIES_PATH = Path("categories.csv")
COUNTIES_PATH = Path("counties.json")

BATCH_SIZE_DEFAULT = 1000
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

# ---------- State population (approx. 2020 Census baseline) ---------------------------------
STATE_POP = {
    "AL": 5024279, "AK": 733391,  "AZ": 7151502, "AR": 3011524, "CA": 39538223,
    "CO": 5773714, "CT": 3605944, "DE": 989948,  "DC": 689545,  "FL": 21538187,
    "GA": 10711908,"HI": 1455271, "ID": 1839106, "IL": 12812508,"IN": 6785528,
    "IA": 3190369, "KS": 2937880, "KY": 4505836, "LA": 4657757, "ME": 1362359,
    "MD": 6177224, "MA": 7029917, "MI": 10077331,"MN": 5706494, "MS": 2961279,
    "MO": 6154913, "MT": 1084225, "NE": 1961504, "NV": 3104614, "NH": 1377529,
    "NJ": 9288994, "NM": 2117522, "NY": 20201249,"NC": 10439388,"ND": 779094,
    "OH": 11799448,"OK": 3959353, "OR": 4237256, "PA": 13002700,"RI": 1097379,
    "SC": 5118425, "SD": 886667,  "TN": 6910840, "TX": 29145505,"UT": 3271616,
    "VT": 643077,  "VA": 8631393, "WA": 7705281, "WV": 1793716, "WI": 5893718,
    "WY": 576851,
}

# ---------- Models ---------------------------------------------------------------------------
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

# ---------- Normalizers ----------------------------------------------------------------------
_WS = re.compile(r"\s+")
def _to_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)

def _norm(s: str) -> str:
    s = _to_str(s)
    s = unicodedata.normalize("NFKC", s).replace("\u00A0", " ")
    return _WS.sub(" ", s).strip()

def _normalize_phone(s: str) -> str:
    s = _norm(s); digits = re.sub(r"\D+", "", s)
    if not digits: return ""
    if len(digits) == 11 and digits.startswith("1"): digits = digits[1:]
    return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}" if len(digits)==10 else digits

def _normalize_zip(z: str) -> str:
    z = _norm(z); m = re.search(r"\b(\d{5})(?:-\d{4})?\b", z)
    return m.group(1) if m else ""

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CSV_HEADERS:
        if c not in out.columns: out[c] = ""
    return out[CSV_HEADERS].copy()

# ---------- Regex & utils --------------------------------------------------------------------
MINISTRY_REGEX = re.compile(
    r"church|parish|ministr(?:y|ies)|ministerio(?:s)?|iglesia|fellowship|christian|gospel|missions|cathedral|temple",
    re.I
)
EMAIL_REGEX = re.compile(r"[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})")
SKIP_MIME_PREFIXES = ("application/pdf", "application/msword", "application/vnd", "image/")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

AGGREGATOR_DOMAINS = {
    "mapquest.com","yelp.com","facebook.com","m.facebook.com","business.facebook.com",
    "instagram.com","twitter.com","x.com","linkedin.com","goo.gl","google.com","g.page",
    "bing.com","duckduckgo.com","eventbrite.com","meetup.com","constantcontact.com",
    "blackbaudhosting.com","donorbox.org","pushpay.com","paypal.com","square.site","my.canva.site",
    "linktr.ee","linktree.com","wikipedia.org","wikipedia.com"
}
DIRECTORY_DOMAINS = {
    "freefood.org","causeiq.com","charitynavigator.org","greatnonprofits.org","guidestar.org",
    "churchfinder.com","faithstreet.com","churchangel.com","yellowpages.com",
    "superpages.com","opengovus.com","directmap.us","privateschoolreview.com",
    "mapquest.com","yelp.com","facebook.com","m.facebook.com","instagram.com"
}
SOCIAL_DOMAINS = {"facebook.com","m.facebook.com","instagram.com","twitter.com","x.com","linkedin.com","youtube.com"}
AGGREGATOR_NAMES = {"mapquest","facebook","yelp","google","bing","duckduckgo","eventbrite","meetup",
                    "paypal","pushpay","donorbox","square","linktree","instagram","twitter","x","linkedin","wikipedia"}

EXCLUDED_URL_KEYWORDS = (
    "donate","donation","tithe","cemetery",
    "careers","jobs","employment","apply","support-us","support-our-ministry",
    "events","event","calendar","sermon","watch","live","livestream","bulletin","newsletter",
    "facebook","instagram","twitter","x","linkedin","youtube",
    "about-us","about","who-we-are","our-team","leadership","contact-us","contact","get-in-touch",
    "news","blog","media","gallery","photo","photo-gallery","video","video gallery","zillow",
    "weddings","memorials","obituaries","funeral-home","obituary","causeiq","article","story","findagrave","record"
)

PHONE_RE = re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
ZIP_RE = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")

NAV_TRASH_RE = re.compile(
    r"\b(skip to content|menu|home|who we are|our team|leadership|give|donate|login|subscribe|newsletter|blog|news|wedding|obituary|media|photos|photo gallery|videos|video gallery|watch live|watch online|livestream|sermon archive|sermons|bulletin|events|event calendar|calendar|newsletters?|staff|meet the pastor|meet our pastor|meet the team|meet our team|about us|about|breaking|memorial|privacy policy|terms of service|terms of use|cookie policy|sitemap|faq|faqs|help|support us|support our ministry|support this ministry|support this church|give online|online giving|make a donation|make a tithe|careers?|jobs?|employment|apply for a job|apply now|join our team|join the team|volunteer application|watch sermons?|watch service online?|watch service live?|watch live service?|watch live stream?|watch livestream?|live stream archive|live stream sermons?|service times?|mass times?|office hours?|office hours|find us on facebook|find us on instagram|find us on twitter|find us on x.com|find us on linkedin|find us on youtube|follow us on facebook|follow us on instagram|follow us on twitter|follow us on x.com|follow us on linkedin|follow us on youtube|service(?:s)?|service times|office hours|plan a visit|contact|cemetery|cemeteries|visit|directions)\b",
    re.I,
)

# ---------- Negative keywords loader ---------------------------------------------------------
def _split_keywords_cell(cell: str) -> list[str]:
    cell = _to_str(cell)
    if not cell: return []
    parts = re.split(r"[,\n;]+", cell)
    return [_norm(p).lower() for p in parts if _norm(p)]

def load_negative_keywords(sources: Optional[list[str]] = None) -> list[str]:
    if sources is None:
        sources = [str(CATEGORIES_PATH), "negative_keywords.json", "negative_keywords.txt"]
    found: list[str] = []
    for src in sources:
        if not src or not os.path.exists(src): continue
        try:
            low = src.lower()
            if low.endswith(".json"):
                with open(src,"r",encoding="utf-8") as f: data = json.load(f)
                items = data.get("negative_keywords", data) if isinstance(data, dict) else data
                if isinstance(items, list):
                    for x in items:
                        k = _norm(_to_str(x)).lower(); 
                        if k: found.append(k)
            elif low.endswith(".txt"):
                with open(src,"r",encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        if line and not line.startswith("#"):
                            k=_norm(line).lower()
                            if k: found.append(k)
            elif low.endswith(".csv"):
                try: df = pd.read_csv(src)
                except Exception: df = pd.read_csv(src, encoding="utf-8-sig")
                for col in ("negative_keywords","Negative Keywords","exclude","Exclude","exclude_keywords"):
                    if col in df.columns:
                        for cell in df[col].dropna().astype(str).tolist():
                            found.extend(_split_keywords_cell(cell))
        except Exception:
            continue
    # unique-preserve order
    seen=set(); out=[]
    for k in found:
        if k and k not in seen:
            seen.add(k); out.append(k)
    return out

def _compile_kw(kw: str) -> re.Pattern:
    kw = kw.strip().lower()
    if not kw: return re.compile(r"$^")
    safe = re.escape(kw)
    if re.fullmatch(r"[a-z0-9]+", kw) and len(kw) <= 3:
        return re.compile(rf"(?i)\b{safe}\b")
    return re.compile(rf"(?i){safe}")

def build_negative_patterns(negatives: Iterable[str]) -> list[re.Pattern]:
    return [_compile_kw(k) for k in set(k.strip().lower() for k in negatives if k)]

def contains_negative(text: str, patterns: list[re.Pattern]) -> bool:
    if not patterns: return False
    t = _norm(text).lower()
    return bool(t and any(p.search(t) for p in patterns))

def candidate_has_negative(cand: dict, patterns: list[re.Pattern]) -> bool:
    hay = " | ".join([_to_str(cand.get("title","")),_to_str(cand.get("url","")),_to_str(cand.get("snippet","")),_to_str(cand.get("site",""))])
    return contains_negative(hay, patterns)

def record_has_negative(row_like: dict, patterns: list[re.Pattern]) -> bool:
    hay = " | ".join([_to_str(row_like.get("Title","")),_to_str(row_like.get("organization_name","")),
                      _to_str(row_like.get("Content","")),_to_str(row_like.get("link_to_ministry","")),
                      _to_str(row_like.get("source_site",""))])
    return contains_negative(hay, patterns)

NEGATIVE_KEYWORDS = load_negative_keywords()
NEG_PATTERNS = build_negative_patterns(NEGATIVE_KEYWORDS)

# ---------- County map ----------------------------------------------------------------------
COUNTY_MAP = json.loads(COUNTIES_PATH.read_text()) if COUNTIES_PATH.exists() else {}

def get_state_abbrev(state_name_or_abbr: str) -> str:
    st = us.states.lookup(state_name_or_abbr)
    if not st: raise ValueError(f"Unknown state: {state_name_or_abbr}")
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
            if r.status != 200: return None
            ctype = (r.headers.get("content-type") or "").lower()
            if not ctype.startswith("text/html"): return None
            raw = await r.read()
            return raw.decode(errors="ignore")
    except Exception:
        return None

def visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript","svg","header","footer","nav","form"]):
        tag.decompose()
    return "\n".join(soup.stripped_strings)

# ---------- Org/title extraction ------------------------------------------------------------
def _clean_org_text(t: str) -> str:
    if not t: return ""
    t = re.sub(r"\s+", " ", t).strip()
    t = re.split(r"\s[|»–—·:]\s", t)[0].strip()
    if NAV_TRASH_RE.search(t) or PHONE_RE.search(t) or ZIP_RE.search(t): return ""
    if len(t) > 120 or len(t.split()) > 10: return ""
    return t

def _jsonld_org_names(soup: BeautifulSoup) -> list[str]:
    out = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try: data = json.loads(tag.string or "")
        except Exception: continue
        items = data if isinstance(data, list) else [data]
        flat=[]
        for obj in items:
            if isinstance(obj, dict):
                flat.append(obj)
                g = obj.get("@graph")
                if isinstance(g, list):
                    flat.extend([n for n in g if isinstance(n, dict)])
        for obj in flat:
            if not isinstance(obj, dict): continue
            t = obj.get("@type")
            types = {str(t).lower()} if isinstance(t, str) else {str(x).lower() for x in (t or [])}
            if {"organization","localbusiness","church","place"} & types:
                nm = obj.get("name") or obj.get("legalName")
                if isinstance(nm, str):
                    c = _clean_org_text(nm)
                    if c: out.append(c)
    return out

def extract_org_name(soup: BeautifulSoup, url: str, result_title: str | None = None) -> str:
    host = (urlparse(url).hostname or "").lower()
    jl = _jsonld_org_names(soup)
    if jl: return jl[0]
    og = soup.find("meta", attrs={"property":"og:site_name"}) or soup.find("meta", attrs={"name":"og:site_name"})
    if og and og.get("content"):
        c = _clean_org_text(og["content"])
        if c: return c
    candidates=[]
    for sel in ["header a[rel='home']", ".site-title a", ".site-title",
                "a.custom-logo-link", ".logo a", "a.logo", ".navbar-brand", ".brand a", "a.brand"]:
        for el in soup.select(sel):
            txt = el.get_text(" ", strip=True) if el.name != "img" else ""
            if not txt:
                img = el.find("img", alt=True)
                if img and img.get("alt"): txt = img["alt"].strip()
            c = _clean_org_text(txt)
            if c: candidates.append(c)
    if candidates:
        candidates.sort(key=len); return candidates[0]
    page_title = (soup.title.string or "").strip() if soup.title else ""
    nm = page_title or result_title or ""
    c = _clean_org_text(nm)
    if c: return c
    core_parts = host.split(":")[0].split(".")
    core = core_parts[-2] if len(core_parts) >= 2 else core_parts[0]
    return core.replace("-", " ").title() if core else (result_title or url)

def extract_ministry_title(soup: BeautifulSoup, url: str, org_name: str, result_title: str | None = None) -> str:
    def cleaned(t: str) -> str:
        t = t.strip()
        if org_name and org_name.lower() in t.lower() and len(t) - len(org_name) < 10: return ""
        for sep in ("|","–","—","·",":","::"):
            t = " ".join([p.strip() for p in t.split(sep) if p.strip()])
        t = re.sub(r"\bon\s+(MapQuest|Facebook|Yelp)\b.*$", "", t, flags=re.I).strip()
        return t
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        t = cleaned(h1.get_text(strip=True))
        if t: return t
    page_title = (soup.title.string or "").strip() if soup.title else ""
    if page_title:
        t = cleaned(page_title)
        if t: return t
    if result_title:
        t = cleaned(result_title)
        if t: return t
    return org_name

# ---------- Category classifier & summariser -------------------------------------------------
class CategoryClassifier:
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
                if keyset: self.keyword_map[str(cat)] = keyset

        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=api_key, callbacks=[CALLBACK])

    async def classify(self, description: str) -> list[str]:
        allowed = ", ".join(self.categories)
        prompt = (
            "Pick all matching ministry categories (comma-separated). "
            "Use the exact strings from the allowed list; do not invent new ones and do not split multi-word or slash categories. "
            f"Allowed: {allowed}. If none fit, return 'Other'.\n"
            "Description:\n" + (description or "")[:800]
        )
        try:
            resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
            raw = [c.strip() for c in (resp.content or "").split(",")]
            out: set[str] = set(); canon = {c.lower(): c for c in self.categories}
            for token in raw:
                if not token: continue
                ci = canon.get(token.lower())
                if ci: out.add(ci)
            return sorted(out) if out else [self._keyword_fallback(description)]
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
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=api_key, callbacks=[CALLBACK])
    async def summary(self, text: str) -> str:
        prompt = "Summarise the ministry in 1–2 sentences highlighting mission and target community.\nText:\n" + (text or "")[:4000]
        try:
            resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return (resp.content or "").strip()
        except Exception:
            return ""
    async def resolve_org_and_title(self, page_text: str, url: str, default_org: str | None, default_title: str | None) -> dict:
        prompt = f"""
You are extracting names for a directory of Christian ministries.
- Decide the umbrella organization/church name.
- Decide the specific ministry title on THIS page (not an external directory listing).
- If page is an aggregator (MapQuest, Yelp, Facebook, Linktree, etc.), infer the real org from context; do NOT use the aggregator as the org.

INPUT
URL: {url}
Default org: {default_org or ""}
Default title: {default_title or ""}
PAGE TEXT (truncated):
{(page_text or "")[:2500]}

OUTPUT JSON ONLY:
{{"organization_name":"...","title":"...","confidence":0..1,"reason":"..."}}
"""
        try:
            resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
            data = json.loads((resp.content or "").strip())
            if not isinstance(data, dict): raise ValueError("bad json")
            data.setdefault("organization_name", default_org or default_title or "")
            data.setdefault("title", default_title or default_org or "")
            if not isinstance(data.get("confidence"), (int, float)): data["confidence"] = 0.0
            return data
        except Exception:
            return {"organization_name": default_org or default_title or "", "title": default_title or default_org or "", "confidence": 0.0, "reason": "fallback"}

# ---------- Browser (Playwright MCP) ---------------------------------------------------------
class Browser:
    def __init__(self):
        self.ws_url = os.getenv("PLAYWRIGHT_MCP_URL")
        self._task: asyncio.Task | None = asyncio.create_task(self._open())
        self._toolkit: PlayWrightBrowserToolkit | None = None
    async def _open(self):
        pw = await async_playwright().start()
        if self.ws_url: return await pw.chromium.connect(self.ws_url)
        return await pw.chromium.launch(headless=True)
    async def toolkit(self) -> PlayWrightBrowserToolkit:
        if self._toolkit is None:
            browser = await self._task
            self._toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        return self._toolkit

# ---------- Helpers -------------------------------------------------------------------------
def _is_directory_domain(host: str) -> bool:
    host = (host or "").lower()
    return any(host == d or host.endswith("."+d) for d in DIRECTORY_DOMAINS)

def safe_urljoin(base: str, href: str | None) -> str | None:
    if not href: return None
    href = href.strip()
    bad_prefixes = ("#","javascript:","mailto:","tel:","sms:","fax:","data:","blob:","about:","chrome:")
    if not href or href.startswith(bad_prefixes): return None
    try:
        joined = urljoin(base, href)
        p = urlparse(joined)
        if p.scheme not in ("http","https") or not p.netloc: return None
        _ = p.netloc
        return joined
    except Exception:
        return None

# ---------- Ministry link extraction ---------------------------------------------------------
MINISTRY_WORDS = ("ministry","ministries","program","group","recovery","care","support","outreach")

def extract_ministry_links(soup: BeautifulSoup, base: str) -> list[tuple[str,str]]:
    links: list[tuple[str,str]] = []
    base_host = urlparse(base).netloc
    for a in soup.find_all("a", href=True):
        text = (a.get_text(" ", strip=True) or "").lower()
        href = safe_urljoin(base, a.get("href"))
        if not href: continue
        low_href = href.lower()
        if any(k in low_href for k in EXCLUDED_URL_KEYWORDS): continue
        same_site = urlparse(href).netloc == base_host
        if same_site and any(w in text for w in MINISTRY_WORDS):
            links.append((text.title()[:120], href))
    seen=set(); out=[]
    for t,u in links:
        if u not in seen: out.append((t,u)); seen.add(u)
    return out

# ---------- Single-link "Ministries" hub detection ------------------------------------------
HUB_LABEL_HINTS = ("ministries","ministry","get involved","serve","groups","next steps")

def find_ministry_hub_link(soup: BeautifulSoup, base_url: str) -> str | None:
    base_host = urlparse(base_url).netloc
    best: tuple[float,str] | None = None
    for a in soup.find_all("a", href=True):
        label = (a.get_text(" ", strip=True) or "").lower()
        href = safe_urljoin(base_url, a.get("href"))
        if not href: continue
        if urlparse(href).netloc != base_host: continue
        path = urlparse(href).path.lower()
        score = 0.0
        if any(h in label for h in HUB_LABEL_HINTS): score += 2.0
        if "/ministr" in path: score += 2.0
        if "/serve" in path or "/groups" in path or "/next-steps" in path: score += 1.0
        if score > 0 and (best is None or score > best[0]):
            best = (score, href)
    return best[1] if best else None

# ---------- URL date-pattern detection -------------------------------------------------------
MONTH_TOKENS = ("jan","january","feb","february","mar","march","apr","april","may","jun","june",
                "jul","july","aug","august","sep","sept","september","oct","october","nov","november","dec","december")
DATE_PATH_PATTERNS = [
    re.compile(r"/(?:19|20)\d{2}/(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12][0-9]|3[01])(?:/|$)"),
    re.compile(r"/(?:19|20)\d{2}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12][0-9]|3[01])(?:/|$)"),
    re.compile(r"/(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12][0-9]|3[01])/(?:19|20)\d{2}(?:/|$)"),
    re.compile(r"/(?:19|20)\d{2}_(?:0?[1-9]|1[0-2])_(?:0?[1-9]|[12][0-9]|3[01])(?:/|$)"),
    re.compile(r"/(?:19|20)\d{2}/(?:" + "|".join(MONTH_TOKENS) + r")/(?:0?[1-9]|[12][0-9]|3[01])(?:/|$)", re.I),
    re.compile(r"/(?:" + "|".join(MONTH_TOKENS) + r")/(?:0?[1-9]|[12][0-9]|3[01])/(?:19|20)\d{2}(?:/|$)", re.I),
]
def url_has_dated_path(u: str) -> bool:
    try: path = urlparse(u).path.lower()
    except Exception: return False
    if not path: return False
    return any(rx.search(path) for rx in DATE_PATH_PATTERNS)

# ---------- LLM gate (relaxed banned-cue logic) ---------------------------------------------
async def is_ministry_page(url: str, page_text: str) -> tuple[bool, str]:
    low = (page_text or "").lower()
    banned_cues = ("donation","tithe","careers","jobs","employment","wedding",
                   "directory","list of","top ","best ","retreat centers near",
                   "advertise","sponsored","submit listing","add listing","find a grave",
                   "news","blog","press release","upcoming events","upcoming sermons","cause iq",
                   "upcoming conferences","calendar","upcoming retreats","event calendar")
    positive = ("ministry","ministries","serve","outreach","group","church","care","recovery","youth",
                "christian","gospel","fellowship","missions","christ","faith","bible","discipleship","worship","prayer","counseling")
    if any(c in low for c in banned_cues):
        has_positive = any(w in low for w in positive)
        looks_like_hub = ("ministr" in low and "our " in low) or ("ministries" in low)
        if not (has_positive or looks_like_hub):
            return (False, "banned cue without ministry signals")
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"), callbacks=[CALLBACK])
        prompt = (
            "Decide if this page describes a real, ongoing Christian ministry/program provided by a specific organization "
            "(YES), or if it's a directory/listing, news article, donation page, job board, or general promo (NO). "
            "Answer with JSON only: {\"ok\": true/false, \"reason\": \"...\"}. TEXT:\n" + (page_text[:3000] or "")
        )
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        data = json.loads((resp.content or "").strip())
        if isinstance(data, dict) and "ok" in data:
            return (bool(data.get("ok")), str(data.get("reason","")))
    except Exception as e:
        return (False, f"llm_error:{type(e).__name__}")
    return (False, "llm_no_decision")

# ---------- Scraper -------------------------------------------------------------------------
class MinistryScraper:
    def __init__(self, classifier: CategoryClassifier, summariser: Summariser,
                 batch: int, concurr: int, max_pages: int, csv_path: Path,
                 negative_patterns: Optional[list[re.Pattern]] = None,
                 allow_dated_urls: bool = False):
        self.classifier, self.summariser = classifier, summariser
        self.batch_size, self.max_pages = batch, max_pages
        self.sema = asyncio.Semaphore(concurr)
        self.browser = Browser()
        self.csv_path = csv_path
        self._seen_links: set[str] = set()
        self.negative_patterns = negative_patterns or []
        self.allow_dated_urls = allow_dated_urls

        # Google API + rate/cache
        self.cfg = load_settings()
        self.rate = RateManager()
        try: self.rate.set_rate("google_search", float(self.cfg.default_google_search_qps))
        except Exception: pass
        try: self.rate.set_rate("google_places", float(self.cfg.default_google_places_qps))
        except Exception: pass
        self.store = Store("weekday_store.sqlite3")

        # Ensure header
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=CSV_HEADERS, quoting=csv.QUOTE_ALL).writeheader()

    async def _resolve_address_google(self, *, org_name: str, homepage: str, page_html: str | None, county: str, state_abbrev: str) -> dict:
        api_key = self.cfg.google_maps_api_key
        if not api_key: return {}
        try:
            if page_html:
                a1 = await g_parse_ministry_page_address(page_html, api_key=api_key, rate=self.rate, target_county=county)
                if a1 and getattr(a1, "city", None) and getattr(a1, "state", None):
                    return {
                        "address": getattr(a1, "street", None) or (getattr(a1, "formatted", "") or "").split(",")[0].strip(),
                        "city": getattr(a1, "city", None),
                        "state": getattr(a1, "state", None),
                        "zip": getattr(a1, "postal_code", None),
                    }
        except Exception:
            pass
        try:
            org = OrgResolution(org_name=org_name, homepage=homepage, domain=(urlparse(homepage).hostname or ""))
            a2 = await g_resolve_org_address(org, county, state_abbrev, api_key=api_key, store=self.store, rate=self.rate)
            if a2:
                return {
                    "address": getattr(a2, "street", None) or (getattr(a2, "formatted", "") or "").split(",")[0].strip(),
                    "city": getattr(a2, "city", None),
                    "state": getattr(a2, "state", None),
                    "zip": getattr(a2, "postal_code", None),
                }
        except Exception:
            pass
        return {}

    async def scrape_ministry(self, meta: dict[str,str], state_abbrev: str, county: str = "") -> list[MinistryRecord]:
        async with self.sema:
            url, result_title = meta["url"], meta.get("title","")

            host0 = (urlparse(url).hostname or "").lower()
            if host0 and (_is_directory_domain(host0) or any(host0 == d or host0.endswith("."+d) for d in AGGREGATOR_DOMAINS)):
                print(f"            ⨯ skipped (hard-domain): {url}")
                return []

            # Early dated-URL guard at page boundary
            if not self.allow_dated_urls and url_has_dated_path(url):
                print(f"            ⨯ skipped (dated-url): {url}")
                return []

            # Early negative keyword guard on the SERP item
            if candidate_has_negative(meta, self.negative_patterns):
                print(f"            ⨯ skipped (serp-neg): {url}")
                return []

            # Fetch via MCP, fall back to HTTP
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
                if not page_html: return []
                page_text = visible_text(page_html)

            if not page_text.strip(): return []

            # Page-level negatives
            if contains_negative(" ".join([result_title, url, page_text]), self.negative_patterns):
                print(f"            ⨯ skipped (page-neg): {url}")
                return []

            soup = BeautifulSoup(page_html or "", "html.parser")
            home_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            host = (urlparse(url).hostname or "").lower()
            is_agg = any(host.endswith(d) for d in AGGREGATOR_DOMAINS)
            page_title = (soup.title.string or "").strip() if soup.title else ""

            org_name_guess = extract_org_name(soup, url, result_title)
            title_guess = extract_ministry_title(soup, url, org_name_guess, result_title)

            page_text_for_ai = (
                f"SEARCH_RESULT_TITLE: {result_title or ''}\n"
                f"PAGE_TITLE: {page_title}\n"
                f"HOST: {host}\n"
                f"IS_AGGREGATOR: {is_agg}\n" + page_text
            )
            resolved = await self.summariser.resolve_org_and_title(page_text_for_ai, url, org_name_guess, title_guess)
            org_name = _clean_org_text((resolved.get("organization_name") or "").strip()) or _clean_org_text(org_name_guess) or _clean_org_text(title_guess) or extract_org_name(soup, url, result_title)
            page_level_title = (resolved.get("title") or title_guess).strip()

            # Fan-out: detect ministries index or single hub
            sublinks = extract_ministry_links(soup, url)
            is_index_like = ("/ministr" in (urlparse(url).path.lower())) or (len(sublinks) >= 3)
            if not is_index_like:
                hub = find_ministry_hub_link(soup, url)
                if hub:
                    print(f"            ↪ following ministries hub: {hub}")
                    html_hub = await fetch_html(hub)
                    if html_hub:
                        soup_hub = BeautifulSoup(html_hub, "html.parser")
                        sublinks = extract_ministry_links(soup_hub, hub)
                        if len(sublinks) >= 1:
                            is_index_like = True

            records: list[MinistryRecord] = []
            if is_index_like and sublinks:
                for link_title, link_url in sublinks[:30]:
                    if not self.allow_dated_urls and url_has_dated_path(link_url):
                        continue
                    if contains_negative(" ".join([link_title, link_url]), self.negative_patterns):
                        continue
                    html2 = await fetch_html(link_url)
                    if not html2:
                        continue
                    text2 = visible_text(html2)
                    if contains_negative(text2, self.negative_patterns):
                        continue
                    soup2 = BeautifulSoup(html2, "html.parser")
                    org2_guess = org_name
                    title2_guess = extract_ministry_title(soup2, link_url, org2_guess, link_title) or link_title
                    page_title2 = (soup2.title.string or "").strip() if soup2.title else ""
                    host2 = (urlparse(link_url).hostname or "").lower()
                    is_agg2 = any(host2.endswith(d) for d in AGGREGATOR_DOMAINS)
                    page_text2_for_ai = (
                        f"SEARCH_RESULT_TITLE: {link_title or ''}\n"
                        f"PAGE_TITLE: {page_title2}\n"
                        f"HOST: {host2}\n"
                        f"IS_AGGREGATOR: {is_agg2}\n" + text2
                    )
                    resolved2 = await self.summariser.resolve_org_and_title(page_text2_for_ai, link_url, org2_guess, title2_guess)
                    org2 = (resolved2.get("organization_name") or org2_guess).strip()
                    title2 = (resolved2.get("title") or title2_guess).strip()

                    summary2 = await self.summariser.summary(text2)
                    cats2 = await self.classifier.classify(summary2 or text2[:600])
                    addr2 = await self._resolve_address_google(org_name=org2, homepage=home_url, page_html=html2, county=county, state_abbrev=state_abbrev)
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
                    if record_has_negative(rec.model_dump(by_alias=True), self.negative_patterns):
                        continue
                    records.append(rec)
                return records

            summary = await self.summariser.summary(page_text)
            ok, reason = await is_ministry_page(url, page_text)
            if not ok:
                print(f"            ⨯ skipped (gate): {url}  — {reason}")
                return []
            cats = await self.classifier.classify(summary or page_text[:600])
            addr = await self._resolve_address_google(org_name=org_name, homepage=home_url, page_html=page_html, county=county, state_abbrev=state_abbrev)
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
            if record_has_negative(rec.model_dump(by_alias=True), self.negative_patterns):
                print(f"            ⨯ skipped (record-neg): {url}")
                return []
            return [rec]

    async def scrape_county(self, county: str, state_abbrev: str) -> AsyncIterator[MinistryRecord]:
        seen: set[str] = set()
        SERP_NEGATIVE_DOMAINS = {
            "zillow.com","realtor.com","redfin.com", "causeiq.com","charitynavigator.org","greatnonprofits.org", 
            "whatsupnewp.com","discovernewport.org","loc.gov", "findagrave.com","legacy.com", "archive.org",
            "mapquest.com","yelp.com","yellowpages.com","superpages.com","reddit.com","tripadvisor.com",
            "trustpilot.com","bbb.org","linktr.ee","tiktok.com","pinterest.com","tumblr.com",
            "facebook.com","m.facebook.com","instagram.com","twitter.com","x.com","linkedin.com","youtube.com",
        }
        SERP_NEGATIVE_KEYWORDS = ("zillow","calendar","press release","news","obituary","archive")

        def _serp_negative(b: dict) -> bool:
            host = (urlparse(b["url"]).hostname or "").lower()
            title = (b.get("title") or "").lower()
            snippet = (b.get("snippet") or "").lower()
            url_low = (b.get("url") or "").lower()
            if any(host == d or host.endswith("."+d) for d in SERP_NEGATIVE_DOMAINS):
                return True
            hay = " ".join([title, snippet, url_low])
            if any(k in hay for k in SERP_NEGATIVE_KEYWORDS):
                return True
            path = (urlparse(b["url"]).path or "").lower()
            if any(seg in path for seg in ("/news/","/blog/","/press/","/article","/stories/","/story/","/obituaries/","/obituary/","/press-release")):
                return True
            if not self.allow_dated_urls and url_has_dated_path(b["url"]):
                return True
            return False

        for page in range(self.max_pages):
            batch_sr = await google_search_ministries(
                county, state_abbrev, page,
                batch_size=self.batch_size,
                cse_id=self.cfg.google_cse_id,
                api_key=self.cfg.google_search_api_key,
                rate=self.rate,
                store=self.store,
            )
            batch = [{
                "title": getattr(s, "title", ""),
                "url": getattr(s, "url", ""),
                "snippet": getattr(s, "snippet", ""),
                "source": getattr(s, "source", "cse"),
                "site": urlparse(getattr(s, "url", "") or "").hostname or "",
            } for s in batch_sr]

            batch = [b for b in batch if not candidate_has_negative(b, self.negative_patterns) and not _serp_negative(b)]

            new = [
                b for b in batch
                if b["url"] not in seen
                and (MINISTRY_REGEX.search(b["title"]) or MINISTRY_REGEX.search(b["url"]))
                and not b["url"].lower().endswith((".pdf",".doc",".docx",".ppt",".pptx"))  # .html/.htm allowed
                and not await is_download(b["url"])
            ]
            if not new:
                break
            seen.update(b["url"] for b in new)
            for m in new:
                print(f"        ↳ queueing {m['url']}")
            tasks = [self.scrape_ministry(meta, state_abbrev, county) for meta in new]
            for coro in asyncio.as_completed(tasks):
                recs = await coro
                for rec in recs:
                    yield rec

    def _write_record(self, record: MinistryRecord):
        try:
            if record_has_negative(record.model_dump(by_alias=True), self.negative_patterns):
                return
            _ltm_host = (urlparse(record.link_to_ministry).hostname or "").lower()
            _src_host = (urlparse(record.source_site).hostname or "").lower() if record.source_site else ""
            if _ltm_host and (_is_directory_domain(_ltm_host) or any(_ltm_host.endswith(d) for d in AGGREGATOR_DOMAINS)):
                return
            if _src_host and (_is_directory_domain(_src_host) or any(_src_host.endswith(d) for d in AGGREGATOR_DOMAINS)):
                return
        except Exception:
            pass

        if record.link_to_ministry in self._seen_links:
            return
        self._seen_links.add(record.link_to_ministry)

        row = {
            "organization_name": record.org_name or "",
            "Ministry Categories": "|".join(record.ministry_category) if isinstance(record.ministry_category, (list, tuple, set)) else (record.ministry_category or ""),
            "Title": record.title or "",
            "link_to_ministry": record.link_to_ministry or "",
            "phone": _normalize_phone(record.phone or ""),
            "Content": _norm(record.content_summary or ""),
            "email": (record.email or "").strip().lower(),
            "address": _norm(record.address or ""),
            "city": _norm(record.city or ""),
            "state": (record.state or "").upper(),
            "zip": _normalize_zip(record.zip or ""),
            "country": record.country or "USA",
            "source_site": record.source_site or "",
        }

        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=CSV_HEADERS, quoting=csv.QUOTE_ALL).writeheader()

        with self.csv_path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, quoting=csv.QUOTE_ALL, extrasaction="ignore", restval="")
            writer.writerow(row)
        print(f"            ✔ wrote {row['organization_name'][:60]}…")

    async def scrape_state(self, state_name: str, start_county: str | None = None):
        state_abbrev = get_state_abbrev(state_name)
        counties = COUNTY_MAP.get(state_abbrev, [])
        if start_county:
            sc = start_county.strip().lower()
            try:
                start_idx = next(i for i, c in enumerate(counties) if c.strip().lower() == sc)
                counties = counties[start_idx:]
                print(f"Resuming at county: {counties[0]} (index {start_idx})")
            except StopIteration:
                print(f"[warn] start_county '{start_county}' not found — starting from the first county.")
        for county in counties:
            print(f"    … {county}")
            async for rec in self.scrape_county(county, state_abbrev):
                self._write_record(rec)

    async def close(self):
        if _async_session and not _async_session.closed:
            await _async_session.close()

# ---------- Cleaning -------------------------------------------------------------------------
NON_MINISTRY_KEYWORDS = [
    "article","zillow","ebay","reddit","tv","youtube","imdb","movie",
    "blog","obituary","story","amazon","tribune","university","forum",
    "news","opinion","case","review","wikipedia","facebook","twitter","instagram",
    "craigslist","yelp","glassdoor","indeed","linkedin","pinterest","tiktok",
    "donate","donation","tithe","careers","jobs","employment","find a grave","findagrave","causeiq",
    "insurance","mortgage","realty","real estate","apartments","housing","rentals","rental",
    "records","genealogy","genealogical","archives",
    "video","videos","podcast","podcasts","press release","listing","listings","directory","directories",
    "christian school","christian academy","photos","photo","gallery","galleries","image","images","mapquest","maps","map",
]

def clean_state_output(csv_path: Path) -> None:
    try:
        if not csv_path.exists(): return
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        if df.empty: return
        df = _ensure_cols(df)
        low = df[["Title","organization_name","Content","link_to_ministry","source_site"]].apply(lambda s: s.astype(str).str.lower())
        joined = low.agg(" ".join, axis=1)
        mask_bad = joined.apply(lambda t: any(k in t for k in NON_MINISTRY_KEYWORDS) or contains_negative(t, NEG_PATTERNS))
        # Add dated URL path filter
        date_mask = df['link_to_ministry'].astype(str).apply(url_has_dated_path)
        mask_bad = mask_bad | date_mask

        removed = df.loc[mask_bad].copy()
        kept = df.loc[~mask_bad].copy()

        removed_path = csv_path.with_suffix(".removed.csv")
        if not removed.empty:
            _ensure_cols(removed).to_csv(removed_path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig", lineterminator="\n")
        _ensure_cols(kept).to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig", lineterminator="\n")
        print(f"Cleaned file: kept {len(kept)} rows, removed {len(removed)} rows by keyword/date filters")
    except Exception as e:
        print(f"[clean] error during cleanup: {e}")

async def ai_validate_output(csv_path: Path, remove_if_confident: float = 0.90) -> None:
    try:
        if not csv_path.exists(): return
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        if df.empty: return
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_TRIAGE","gpt-4o-mini"), temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"), callbacks=[CALLBACK])
        def mk_prompt(row: pd.Series) -> str:
            url = str(row.get('link_to_ministry','') or '')
            src = str(row.get('source_site','') or '')
            try: url_domain = (urlparse(url).hostname or '').lower()
            except Exception: url_domain = ''
            try: org_domain = (urlparse(src).hostname or '').lower()
            except Exception: org_domain = ''
            same_domain = bool(url_domain and org_domain and (url_domain == org_domain or url_domain.endswith('.'+org_domain) or org_domain.endswith('.'+url_domain)))
            return (
                "Validate a CHRISTIAN ministry record. Keep only real, ongoing ministries run by a specific organization. "
                "NO for 3rd‑party directories/aggregators, social/linktree hubs, news/blog/articles, obituaries, forums, marketplaces, university/academic pages, or generic promo.\n"
                "Exception: 'Ministries' hub on the official org domain is YES.\n"
                "If uncertain, prefer YES with low confidence; NO only when clear.\n\n"
                f"Title: {row.get('Title','')[:200]}\n"
                f"Organization: {row.get('organization_name','')[:200]}\n"
                f"Categories: {row.get('Ministry Categories','')[:200]}\n"
                f"URL: {url[:400]}\n"
                f"Source Site: {src[:200]}\n"
                f"URL Domain: {url_domain}\n"
                f"Org Domain: {org_domain}\n"
                f"Same Domain: {same_domain}\n"
                f"Summary: {(row.get('Content','') or '')[:600]}\n\n"
                "Respond JSON ONLY as {\"ok\": true|false, \"confidence\": 0..1, \"reason\": \"...\"}."
            )
        oks, confs, reasons = [], [], []
        for _, row in df.iterrows():
            try:
                resp = await llm.ainvoke([HumanMessage(content=mk_prompt(row))])
                data = json.loads((resp.content or "").strip())
                ok = bool(data.get("ok", True))
                conf = float(data.get("confidence", 0.0) or 0.0)
                reason = str(data.get("reason", "")).strip()
            except Exception:
                ok, conf, reason = True, 0.0, "llm_error"
            oks.append(ok); confs.append(conf); reasons.append(reason)
        df["ai_ok"] = oks; df["ai_confidence"] = confs; df["ai_reason"] = reasons
        mask_ai_remove = (~df["ai_ok"]) & (df["ai_confidence"] >= remove_if_confident)
        removed_ai = df.loc[mask_ai_remove].copy()
        kept_ai = df.loc[~mask_ai_remove].copy()
        ai_removed_path = csv_path.with_suffix(".ai_removed.csv")
        if not removed_ai.empty:
            _ensure_cols(removed_ai).assign(
                ai_ok=removed_ai["ai_ok"],
                ai_confidence=removed_ai["ai_confidence"],
                ai_reason=removed_ai["ai_reason"]
            ).to_csv(ai_removed_path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig", lineterminator="\n")
        _ensure_cols(kept_ai).to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig", lineterminator="\n")
        print(f"AI validation: kept {len(kept_ai)} rows, removed {len(removed_ai)} rows (confidence >= {remove_if_confident:.2f})")
    except Exception as e:
        print(f"[ai-clean] error during AI validation: {e}")

# ---------- CLI / main -----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Christian ministry directory scraper")
    p.add_argument("--state", help="State to scrape (name or abbreviation). If --continuous is set and --state is omitted, starts at Alabama.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--concurrency", type=int, default=CONCURRENCY_DEFAULT)
    p.add_argument("--max-pages", type=int, default=MAX_PAGES_DEFAULT, help="Base per-county page cap (used directly unless --scale-pages-by-pop is on)")
    p.add_argument("--max-pages-cap", type=int, default=30, help="Upper cap per county when scaling by population")
    p.add_argument("--min-pages-per-county", type=int, default=3, help="Lower bound per county when scaling by population")
    p.add_argument("--ref-counties", type=int, default=50, help="Reference county count in the scaling formula")
    p.add_argument("--pop-baseline", type=int, default=0, help="Population baseline used in scaling; 0 = use median of STATE_POP")
    p.add_argument("--start-county", type=str, default=None, help="Start scraping at this county (case-insensitive) within the FIRST state processed")
    p.add_argument("--test-url", help="Debug a single URL and print resolved names")
    p.add_argument("--neg", nargs="*", default=None, help="Paths to negative keyword files to merge (csv/json/txt)")
    # Runtime controls
    p.add_argument("--allow-dated-urls", action="store_true", help="Do not exclude URLs that contain a date-like path (e.g., /YYYY/MM/DD/)")
    p.add_argument("--no-clean", action="store_true", help="Skip post-scrape cleaning and AI validation")
    p.add_argument("--continuous", action="store_true", help="Run continuously across states alphabetically (starting at --state if provided)")
    p.add_argument("--scale-pages-by-pop", action="store_true", help="Scale per-county max pages by state population and county count")
    return p.parse_args()

def compute_scaled_pages(state_abbrev: str, base_max_pages: int, num_counties: int, args) -> int:
    pop = STATE_POP.get(state_abbrev)
    if not pop or not args.scale_pages_by_pop:
        return base_max_pages
    pop_baseline = args.pop_baseline if args.pop_baseline > 0 else int(median(STATE_POP.values()))
    ref_cty = max(1, args.ref_counties)
    scaled = base_max_pages * (pop / pop_baseline) * (ref_cty / max(1, num_counties))
    val = int(round(scaled))
    val = max(args.min_pages_per_county, min(args.max_pages_cap, val))
    print(f"[scale] {state_abbrev}: base={base_max_pages}, pop={pop}, baseline={pop_baseline}, counties={num_counties}, ref={ref_cty} → max_pages={val}")
    return val

async def run_state(state_abbrev: str, args, classifier: CategoryClassifier, summariser: Summariser):
    csv_path = OUTPUT_DIR / f"ministry_directory_{state_abbrev}.csv"
    # Determine per-state max_pages (scaled if requested)
    num_counties = len(COUNTY_MAP.get(state_abbrev, []))
    effective_max_pages = compute_scaled_pages(state_abbrev, args.max_pages, num_counties, args)
    scraper = MinistryScraper(classifier, summariser, args.batch_size, args.concurrency, effective_max_pages, csv_path,
                              negative_patterns=NEG_PATTERNS, allow_dated_urls=args.allow_dated_urls)
    try:
        print(f"\n=== Processing {state_abbrev} (max-pages per county = {effective_max_pages}) ===")
        await scraper.scrape_state(state_abbrev, start_county=args.start_county)
        if not args.no_clean:
            clean_state_output(csv_path)
            await ai_validate_output(csv_path)
    finally:
        await scraper.close()

async def main():
    args = parse_args()

    # Merge extra negatives if provided
    global NEGATIVE_KEYWORDS, NEG_PATTERNS
    if args.neg:
        extra = load_negative_keywords(sources=args.neg)
        merged = list(dict.fromkeys(NEGATIVE_KEYWORDS + extra))
        NEGATIVE_KEYWORDS = merged
        NEG_PATTERNS = build_negative_patterns(NEGATIVE_KEYWORDS)

    # Single-URL debug
    if args.test_url:
        st = args.state or "AL"
        state_abbrev = get_state_abbrev(st)
        csv_path = OUTPUT_DIR / f"ministry_directory_{state_abbrev}.csv"
        classifier = CategoryClassifier(CATEGORIES_PATH)
        summariser = Summariser()
        # Use scaled pages even in test to show the computed value in logs
        num_counties = len(COUNTY_MAP.get(state_abbrev, []))
        effective_max_pages = compute_scaled_pages(state_abbrev, args.max_pages, num_counties, args)
        scraper = MinistryScraper(classifier, summariser, 1, 1, effective_max_pages, csv_path, negative_patterns=NEG_PATTERNS, allow_dated_urls=args.allow_dated_urls)
        meta = {"url": args.test_url, "title": "(manual)"}
        recs = await scraper.scrape_ministry(meta, state_abbrev=state_abbrev)
        for r in recs:
            print("DEBUG →", r.model_dump(by_alias=True))
        await scraper.close()
        return

    classifier = CategoryClassifier(CATEGORIES_PATH)
    summariser = Summariser()

    if args.continuous:
        # Walk states alphabetically by name
        states = sorted([(s.name, s.abbr) for s in us.states.STATES if s.abbr], key=lambda x: x[0].lower())
        start_abbr = get_state_abbrev(args.state) if args.state else "AL"
        # Find start index
        try:
            start_idx = next(i for i, (_, ab) in enumerate(states) if ab == start_abbr)
        except StopIteration:
            start_idx = 0
        for i, (nm, ab) in enumerate(states[start_idx:], start=start_idx):
            # For first state, use user-provided start county; thereafter none
            start_cty = args.start_county if i == start_idx else None
            # Build a shallow args clone with state+county for this round
            class _Obj: pass
            per_state_args = _Obj(); per_state_args.__dict__ = dict(args.__dict__)
            per_state_args.start_county = start_cty
            await run_state(ab, per_state_args, classifier, summariser)
        return

    # Non-continuous: require a state
    if not args.state:
        raise SystemExit("Error: --state is required unless --continuous is set")
    state_abbrev = get_state_abbrev(args.state)
    await run_state(state_abbrev, args, classifier, summariser)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user – exiting gracefully.")
