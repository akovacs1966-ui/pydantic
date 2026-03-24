from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from typing import Any
from urllib.parse import urlparse

import httpx

from schemas import Source, SourceType


PRIMARY_DOMAINS = {
    "sec.gov",
    "www.sec.gov",
}

PRIMARY_DOMAIN_SUFFIXES = (
    ".sec.gov",
)

IR_HINTS = (
    "investor",
    "investors",
    "ir.",
    "/investor",
    "/investors",
    "investor-relations",
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def stable_source_id(url: str) -> str:
    h = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:10]
    return f"src_{h}"


def classify_source_type(url: str) -> SourceType:
    host = (urlparse(url).hostname or "").lower()
    if host in PRIMARY_DOMAINS or host.endswith(PRIMARY_DOMAIN_SUFFIXES):
        return SourceType.primary
    if any(h in url.lower() for h in IR_HINTS):
        return SourceType.primary
    return SourceType.unknown


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript)\b.*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</p\s*>", "\n", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = unescape(html)
    lines = [normalize_whitespace(l) for l in html.splitlines()]
    lines = [l for l in lines if l]
    text = "\n".join(lines).strip()
    text = text.replace("\x00", "")
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def pick_excerpt(text: str, *, max_chars: int = 4000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


@dataclass(frozen=True)
class SearchResult:
    title: str
    href: str
    body: str | None = None
    raw: dict[str, Any] | None = None


async def ddg_search(query: str, *, max_results: int = 8) -> list[Source]:
    """
    DuckDuckGo search via the `ddgs` library (DDGS).

    Notes:
    - This function is async but uses a thread offload internally because DDGS is sync.
    - Returned Source objects include title/url/snippet and primary-source classification.
    """
    try:
        from ddgs import DDGS  # type: ignore
    except Exception as e:  # pragma: no cover
        try:
            # Backwards-compat: older package name.
            from duckduckgo_search import DDGS  # type: ignore
        except Exception as e2:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: ddgs (or duckduckgo-search). Install it via requirements.txt."
            ) from e2

    def _run() -> list[Source]:
        out: list[Source] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = str(r.get("title") or "").strip()
                href = str(r.get("href") or "").strip()
                body = (r.get("body") or None) if isinstance(r.get("body"), str) else None
                if not title or not href:
                    continue
                sid = stable_source_id(href)
                out.append(
                    Source(
                        id=sid,
                        title=title,
                        url=href,
                        snippet=body,
                        retrieved_at=utcnow(),
                        source_type=classify_source_type(href),
                        metadata={"engine": "duckduckgo", "query": query},
                    )
                )
        return out

    import asyncio

    return await asyncio.to_thread(_run)


async def fetch_source_excerpt(
    url: str,
    *,
    timeout_s: float = 20.0,
    max_bytes: int = 1_500_000,
    max_chars: int = 4000,
) -> str | None:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; DeepResearchAgent/1.0; +https://example.invalid)",
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_s) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        content_type = (r.headers.get("content-type") or "").lower()
        content = r.content[:max_bytes]

        if "text/html" in content_type or content_type.startswith("text/"):
            text = strip_html_to_text(content.decode(r.encoding or "utf-8", errors="ignore"))
            return pick_excerpt(text, max_chars=max_chars) if text else None

        if "application/pdf" in content_type:
            return None

        return None


async def enrich_sources_with_excerpts(
    sources: list[Source],
    *,
    max_to_fetch: int = 4,
    max_concurrency: int = 4,
) -> list[Source]:
    """
    Fetch content excerpts for a subset of sources.
    Prioritizes primary sources when possible.
    """
    if not sources:
        return []

    ranked = sorted(
        sources,
        key=lambda s: (0 if s.source_type == SourceType.primary else 1),
    )
    to_fetch = ranked[:max_to_fetch]

    import asyncio

    sem = asyncio.Semaphore(max_concurrency)

    async def _one(s: Source) -> Source:
        if s.content_excerpt:
            return s
        async with sem:
            try:
                excerpt = await fetch_source_excerpt(s.url)
            except Exception:
                excerpt = None
        return s.model_copy(update={"content_excerpt": excerpt})

    enriched = await asyncio.gather(*(_one(s) for s in sources))
    return list(enriched)

