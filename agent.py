from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from research_tools import ddg_search, enrich_sources_with_excerpts
from schemas import AngleSection, ResearchReport, ResolvedEntity, Source, SourceType

MODEL_NAME = "openai:gpt-5-mini"

TICKER_RE = re.compile(r"^[A-Za-z]{1,5}([.-][A-Za-z]{1,3})?$")

SYSTEM_INSTRUCTIONS = """
You are a deep research agent.

You will be given:
- A user input (either a free-text query or a stock ticker)
- DuckDuckGo search results (titles, URLs, snippets) and sometimes short content excerpts

Rules:
- Be evidence-based: every key claim should cite one or more provided Source IDs.
- Prefer primary sources for financials and guidance (SEC filings, earnings releases, investor relations).
- If evidence is missing, say so explicitly and record it under risks/uncertainties.
- Extract numbers and specific phrasing when available.
""".strip()


class AnglePlan(BaseModel):
    angle_title: str = Field(..., description="Non-overlapping research angle title.")
    angle_query: str = Field(..., description="DuckDuckGo query for this angle.")
    rationale: str = Field(..., description="Why this angle matters.")


class AnglePlans(BaseModel):
    angles: list[AnglePlan] = Field(..., min_length=3, max_length=4)


class TickerResolution(BaseModel):
    company_name: str = Field(..., description="Resolved company name for the ticker.")
    context: str = Field(
        ...,
        description="1-2 sentence context: what the company does / why the ticker is relevant.",
    )
    confidence: str = Field(..., description="low/medium/high")

agent = Agent(
    MODEL_NAME,
    instructions=SYSTEM_INSTRUCTIONS,
    output_type=ResearchReport,
)


def _safe_text(s: str) -> str:
    # Avoid lone surrogates / invalid JSON payloads.
    s = s.replace("\x00", "")
    return s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


async def _resolve_ticker(ticker: str) -> ResolvedEntity:
    ticker_norm = ticker.upper()
    results = await ddg_search(f"{ticker_norm} stock company", max_results=6)
    payload = _format_sources_for_prompt(results[:5])

    resolver = Agent(
        MODEL_NAME,
        instructions=SYSTEM_INSTRUCTIONS
        + "\n\nResolve the ticker to the company name using only the provided sources.\n"
        + "If ambiguous, pick the best-supported answer and explain uncertainty in context.",
        output_type=TickerResolution,
    )
    prompt = (
        f"Ticker: {ticker_norm}\n\n"
        f"Sources:\n{payload}\n\n"
        "Return the best company name and short context."
    )
    r = await resolver.run(prompt)
    tr = r.output
    return ResolvedEntity(
        kind="ticker",
        input=ticker,
        ticker=ticker_norm,
        name=tr.company_name.strip(),
        context=tr.context.strip(),
    )


def _format_sources_for_prompt(sources: list[Source]) -> str:
    lines: list[str] = []
    for s in sources:
        st = s.source_type.value
        snip = _safe_text((s.snippet or "").strip())
        excerpt = _safe_text((s.content_excerpt or "").strip())
        lines.append(
            "\n".join(
                [
                    f"- id: {s.id}",
                    f"  title: {_safe_text(s.title)}",
                    f"  url: {_safe_text(s.url)}",
                    f"  source_type: {st}",
                    f"  snippet: {snip}" if snip else "  snippet: (none)",
                    f"  excerpt: {excerpt}" if excerpt else "  excerpt: (not fetched)",
                ]
            )
        )
    return "\n".join(lines)


async def _generate_angles(resolved: ResolvedEntity) -> list[AnglePlan]:
    if resolved.kind == "ticker":
        topic = f"{resolved.name} ({resolved.ticker})"
        discover_query = f"{resolved.name} {resolved.ticker} latest earnings guidance competition SWOT 12 month stock performance"
    else:
        topic = resolved.input
        discover_query = resolved.input

    discover = await ddg_search(discover_query, max_results=6)
    payload = _format_sources_for_prompt(discover[:5])

    planner = Agent(
        MODEL_NAME,
        instructions=SYSTEM_INSTRUCTIONS
        + "\n\nGenerate 3-4 non-overlapping research angles based on the discover sources.\n"
        + "Angles must have distinct queries and avoid redundancy.",
        output_type=AnglePlans,
    )
    prompt = (
        f"Topic: {topic}\n\n"
        f"Discover query: {discover_query}\n\n"
        f"Top discover sources:\n{payload}\n\n"
        "Return 3-4 angles with angle_title, angle_query, and rationale."
    )
    r = await planner.run(prompt)
    return r.output.angles


def _dedupe_sources(sources: list[Source]) -> list[Source]:
    seen: set[str] = set()
    out: list[Source] = []
    for s in sources:
        if s.id in seen:
            continue
        seen.add(s.id)
        out.append(s)
    return out


def _choose_key_sources(all_sources: list[Source], *, max_sources: int = 8) -> list[Source]:
    primary = [s for s in all_sources if s.source_type == SourceType.primary]
    secondary = [s for s in all_sources if s.source_type != SourceType.primary]
    chosen = (primary + secondary)[:max_sources]
    return chosen


async def _build_angle_section(
    resolved: ResolvedEntity,
    angle: AnglePlan,
) -> AngleSection:
    q = angle.angle_query
    sources = await ddg_search(q, max_results=8)
    sources = _dedupe_sources(sources)
    sources = await enrich_sources_with_excerpts(sources, max_to_fetch=4, max_concurrency=4)

    topic = (
        f"{resolved.name} ({resolved.ticker})" if resolved.kind == "ticker" else resolved.input
    )
    payload = _format_sources_for_prompt(sources[:8])

    extractor = Agent(
        MODEL_NAME,
        instructions=SYSTEM_INSTRUCTIONS
        + "\n\nYou are writing one report section for a single angle.\n"
        + "Output must be strictly grounded in the provided sources.\n"
        + "Every claim must have supported_by listing Source IDs.\n"
        + "If sources conflict, include that in notes and reduce confidence.",
        output_type=AngleSection,
    )
    prompt = (
        f"Topic: {topic}\n"
        f"Angle title: {angle.angle_title}\n"
        f"Angle rationale: {angle.rationale}\n"
        f"Angle query: {q}\n\n"
        f"Sources:\n{payload}\n\n"
        "Create an AngleSection with executive_takeaway and 5-10 key findings."
    )
    r = await extractor.run(prompt)
    sec = r.output
    sec.sources = sources  # keep full list we gathered (including excerpts)
    return sec


async def deep_research(user_input: str) -> ResearchReport:
    text = _safe_text(user_input.strip())
    is_ticker = bool(TICKER_RE.match(text))

    if is_ticker:
        resolved = await _resolve_ticker(text)
    else:
        resolved = ResolvedEntity(kind="query", input=text, context=None)

    angles = await _generate_angles(resolved)

    sections = await asyncio.gather(*(_build_angle_section(resolved, a) for a in angles))

    all_sources: list[Source] = _dedupe_sources([s for sec in sections for s in sec.sources])
    key_sources = _choose_key_sources(all_sources)

    synthesizer = Agent(
        MODEL_NAME,
        instructions=SYSTEM_INSTRUCTIONS
        + "\n\nSynthesize a full report across multiple sections.\n"
        + "Be explicit about uncertainties and what evidence is missing.\n",
        output_type=ResearchReport,
    )

    scope_line = (
        f"{resolved.name} ({resolved.ticker})" if resolved.kind == "ticker" else resolved.input
    )
    sections_payload = "\n\n".join(
        [
            "\n".join(
                [
                    f"SECTION: {sec.angle_title}",
                    f"QUERY: {sec.angle_query}",
                    f"TAKEAWAY: {sec.executive_takeaway}",
                    "KEY_FINDINGS:",
                    "\n".join(
                        f"- {c.claim_text} (evidence: {', '.join(c.supported_by) or 'none'})"
                        for c in sec.key_findings
                    )
                    or "- (none)",
                ]
            )
            for sec in sections
        ]
    )
    key_sources_payload = _format_sources_for_prompt(key_sources)

    prompt = (
        f"User input: {text}\n"
        f"Resolved scope: {scope_line}\n\n"
        f"Resolved entity JSON:\n{resolved.model_dump_json(indent=2)}\n\n"
        f"Section summaries:\n{sections_payload}\n\n"
        f"Key sources:\n{key_sources_payload}\n\n"
        "Return a ResearchReport with:\n"
        "- executive_summary (detailed)\n"
        "- risks_uncertainties (bullets)\n"
        "- watch_next (bullets)\n"
        "- sections preserved (do not invent new section titles)\n"
        "- key_sources set to the provided key sources (by id/url/title)\n"
    )

    r = await synthesizer.run(prompt)
    out = r.output
    out.input = text
    out.resolved_entity = resolved
    out.generated_at = datetime.now(timezone.utc)
    out.sections = sections
    out.key_sources = key_sources
    return out


def deep_research_sync(user_input: str) -> ResearchReport:
    return asyncio.run(deep_research(user_input))

