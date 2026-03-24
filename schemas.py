from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ShortAnswer(BaseModel):
    """Structured output for the agent."""

    answer: str = Field(..., description="A short answer to the user's question.")


class SourceType(str, Enum):
    primary = "primary"
    secondary = "secondary"
    unknown = "unknown"


class Source(BaseModel):
    id: str = Field(..., description="Stable ID unique within the report.")
    title: str = Field(..., description="Page or document title.")
    url: str = Field(..., description="Canonical URL when possible.")
    snippet: str | None = Field(None, description="Search snippet / summary from the engine.")
    retrieved_at: datetime = Field(..., description="When this source was retrieved.")
    source_type: SourceType = Field(SourceType.unknown, description="Primary/secondary/unknown.")
    content_excerpt: str | None = Field(
        None,
        description="Short extracted excerpt (truncated) used to ground claims.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional extra fields.")


Confidence = Literal["low", "medium", "high"]


class Claim(BaseModel):
    claim_text: str = Field(..., description="A specific, checkable statement.")
    numbers: list[str] = Field(
        default_factory=list,
        description="Important figures as strings (e.g. '$26.0B', 'Q4 FY2025').",
    )
    confidence: Confidence = Field(
        "medium",
        description="How strongly the evidence supports the claim.",
    )
    supported_by: list[str] = Field(
        default_factory=list,
        description="List of Source IDs that support this claim.",
    )
    notes: str | None = Field(None, description="Clarifications, caveats, or context.")


class AngleSection(BaseModel):
    angle_title: str = Field(..., description="Human-friendly angle title.")
    angle_query: str = Field(..., description="Search query used for the angle deep dive.")
    executive_takeaway: str = Field(
        ...,
        description="1–3 sentence takeaway for this angle.",
    )
    key_findings: list[Claim] = Field(default_factory=list, description="Evidence-backed claims.")
    sources: list[Source] = Field(default_factory=list, description="Sources used in this angle.")


class ResolvedEntity(BaseModel):
    kind: Literal["ticker", "query"] = Field(..., description="Whether input was treated as ticker.")
    input: str = Field(..., description="Original user input.")
    ticker: str | None = Field(None, description="Normalized ticker (if kind=ticker).")
    name: str | None = Field(None, description="Resolved company or entity name.")
    context: str | None = Field(None, description="Short context about the entity/query.")


class ResearchReport(BaseModel):
    input: str = Field(..., description="Original user input.")
    resolved_entity: ResolvedEntity = Field(..., description="Resolved entity or query context.")
    generated_at: datetime = Field(..., description="Report generation timestamp.")
    executive_summary: str = Field(..., description="High-level summary across all angles.")
    sections: list[AngleSection] = Field(default_factory=list, description="Angle-based sections.")
    key_sources: list[Source] = Field(
        default_factory=list,
        description="Globally important sources (subset of all_sources).",
    )
    risks_uncertainties: list[str] = Field(
        default_factory=list,
        description="Material risks, unknowns, and conflicting evidence.",
    )
    watch_next: list[str] = Field(
        default_factory=list,
        description="Concrete things to monitor next (events, filings, metrics).",
    )


def render_report_md(report: ResearchReport) -> str:
    resolved = report.resolved_entity
    header_bits: list[str] = []
    if resolved.kind == "ticker":
        header_bits.append(f"**Ticker**: `{resolved.ticker or resolved.input}`")
        if resolved.name:
            header_bits.append(f"**Company**: {resolved.name}")
    else:
        header_bits.append(f"**Query**: {resolved.input}")

    if resolved.context:
        header_bits.append(f"**Context**: {resolved.context}")

    md: list[str] = []
    md.append("## Executive summary")
    md.append(report.executive_summary.strip())
    md.append("")
    md.append("## Scope")
    md.append("\n".join(f"- {b}" for b in header_bits))
    md.append("")

    md.append("## Sections")
    for idx, sec in enumerate(report.sections, start=1):
        md.append(f"### {idx}. {sec.angle_title}")
        md.append(f"- **Angle query**: `{sec.angle_query}`")
        md.append(f"- **Takeaway**: {sec.executive_takeaway.strip()}")
        md.append("")
        if sec.key_findings:
            md.append("#### Key findings (evidence-based)")
            for claim in sec.key_findings:
                evidence = ", ".join(f"`{sid}`" for sid in claim.supported_by) or "`(none)`"
                numbers = ", ".join(claim.numbers) if claim.numbers else ""
                numbers_part = f" ({numbers})" if numbers else ""
                notes_part = f" — {claim.notes.strip()}" if claim.notes else ""
                md.append(
                    f"- **{claim.claim_text.strip()}**{numbers_part}{notes_part}\n"
                    f"  - Evidence: {evidence} (confidence: {claim.confidence})"
                )
            md.append("")
        if sec.sources:
            md.append("#### Sources")
            for s in sec.sources:
                snip = f" — {s.snippet.strip()}" if s.snippet else ""
                md.append(f"- `{s.id}`: [{s.title}]({s.url}){snip}")
            md.append("")

    md.append("## Risks and uncertainties")
    if report.risks_uncertainties:
        md.extend(f"- {r}" for r in report.risks_uncertainties)
    else:
        md.append("- (None explicitly identified.)")
    md.append("")

    md.append("## What to watch next")
    if report.watch_next:
        md.extend(f"- {w}" for w in report.watch_next)
    else:
        md.append("- (No watchlist items generated.)")
    md.append("")

    if report.key_sources:
        md.append("## Key sources (global)")
        for s in report.key_sources:
            md.append(f"- `{s.id}`: [{s.title}]({s.url})")
        md.append("")

    return "\n".join(md).strip() + "\n"

