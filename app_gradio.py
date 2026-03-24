from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import gradio as gr

from agent import deep_research_sync
from schemas import render_report_md


def chat(message: str, history: list) -> str:
    # `history` is unused; each run performs its own multi-step research.
    report = deep_research_sync(message)
    return render_report_md(report)


def main() -> None:
    demo = gr.ChatInterface(
        fn=chat,
        title="Deep Research Agent (DuckDuckGo)",
        description="Enter a free-text query or a stock ticker (e.g. NVDA). The agent runs multi-step web research and returns an evidence-based report.",
    )
    demo.launch()


if __name__ == "__main__":
    main()

