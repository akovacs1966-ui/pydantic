from __future__ import annotations

import json
import sys

from dotenv import load_dotenv

load_dotenv()

from agent import deep_research_sync  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for q in ["NVDA", "How is generative AI impacting semiconductor demand?"]:
        print("=" * 80)
        print("INPUT:", q)
        report = deep_research_sync(q)
        print(report.executive_summary[:800].strip() + "\n")
        print("Sections:", [s.angle_title for s in report.sections])
        print("Key sources:", [(s.id, s.url) for s in report.key_sources[:3]])
        print("\nJSON (truncated):")
        data = report.model_dump()
        print(json.dumps(data, indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()

