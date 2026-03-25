# Deep Research Agent with PydanticAI + Gradio

A sophisticated **multi-turn research agent** built with **PydanticAI** and **Gradio** that conducts evidence-based research on any topic or stock ticker, generating structured reports with citations and risk assessments.

## Overview

This agent performs **deep, multi-angle research** by:

1. **Resolving Queries**: Detects stock tickers (e.g., `NVDA`) and resolves them to companies
2. **Generating Research Angles**: Creates 3-4 complementary research perspectives based on initial discovery
3. **Gathering Evidence**: Searches DuckDuckGo for each angle and fetches full-text excerpts
4. **Extracting Claims**: Uses AI to distill evidence into structured, cited findings
5. **Synthesizing Reports**: Combines all angles into a comprehensive report with:
   - Executive summary
   - Key findings with confidence levels
   - Source citations (prioritizing primary sources like SEC filings)
   - Risk and uncertainties
   - Recommendations for follow-up research

## Key Features

- **Evidence-Based**: Every claim includes citations to source IDs and confidence levels
- **Structured Output**: Leverages Pydantic for type-safe, validated data
- **Multi-Agent Pipeline**: Separate specialized agents for ticker resolution, angle generation, claim extraction, and synthesis
- **Intelligent Source Classification**: Identifies primary sources (SEC filings, IR) vs. secondary
- **Async Concurrency**: Parallel source fetching and concurrent angle processing
- **Gradio UI**: Interactive web interface for easy querying
- **CLI Testing**: Included `smoke_test.py` for batch research
- Saját github

## Prerequisites

- Python **3.10+**
- OpenAI API key (for `gpt-5-mini` model)

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your OpenAI API key:
   ```bash
   # Create .env file in project root
   OPENAI_API_KEY=sk-...your-key-here...
   ```

## Running

### Web UI (Gradio)
```bash
python app_gradio.py
```
Opens an interactive chat interface at `http://localhost:7860`

### CLI Test
```bash
python smoke_test.py
```
Runs sample research queries and outputs results to console (NVDA stock ticker and a free-text query example).

## Project Structure

- **`agent.py`**: Core research pipeline
  - `deep_research(user_input)`: Async entry point for research
  - `_resolve_ticker(ticker)`: Resolves stock tickers to company names
  - `_generate_angles(resolved)`: Creates 3-4 research angles
  - `_build_angle_section(resolved, angle)`: Deep research on a single angle
  - Multiple `Agent` instances for different roles (resolver, planner, extractor, synthesizer)

- **`schemas.py`**: Pydantic models for structured data
  - `ResearchReport`: Complete report structure
  - `AngleSection`: Single research angle with findings
  - `Claim`: Evidence-backed statement with confidence and citations
  - `Source`: Source metadata with excerpt
  - `ResolvedEntity`: Resolved input (ticker or query)

- **`research_tools.py`**: Research utilities
  - `ddg_search()`: DuckDuckGo search integration
  - `fetch_source_excerpt()`: Fetches and extracts text from URLs
  - `enrich_sources_with_excerpts()`: Parallel excerpt fetching with rate limiting

- **`app_gradio.py`**: UI layer
  - Gradio `ChatInterface` wrapper around the agent

- **`smoke_test.py`**: Test script for batch research

## Example Usage

### Interactive (Web UI)
```
User: NVDA
→ [Resolves to NVIDIA Corporation]
→ [Generates angles: Latest Earnings, Competitive Position, AI Demand, Stock Performance]
→ [Returns report with 50+ sourced claims across angles]
```

### Programmatic
```python
from agent import deep_research_sync
from schemas import render_report_md

report = deep_research_sync("Is Tesla a good investment?")
print(render_report_md(report))
```

## Model Configuration

The agent uses **OpenAI `gpt-5-mini`** for speed and cost efficiency. You can modify the model in `agent.py`:
```python
MODEL_NAME = "openai:gpt-5-mini"
```

See [PydanticAI model docs](https://ai.pydantic.dev/models/) for switching to other providers (Anthropic, Gemini, etc.).

## Architecture Highlights

1. **Type-Safe Agents**: Each agent specifies its `output_type` (Pydantic model), ensuring validated outputs
2. **Async Pipeline**: Concurrent angle research and source fetching for performance
3. **Source Prioritization**: Distinguishes primary sources (SEC, IR) for higher credibility
4. **Deduplication**: Removes duplicate sources across angles
5. **Safe Text Handling**: Prevents encoding/JSON serialization errors

## Limitations & Future Work

- Sources limited to DuckDuckGo results (no real-time data feeds or databases)
- Content excerpt size capped to 4000 chars per source
- Max 3-4 research angles per query (configurable)
- No multi-turn conversation memory (each query is independent)

## References

- [PydanticAI Documentation](https://ai.pydantic.dev/)
- [Gradio Documentation](https://www.gradio.app/)
- [OpenAI API](https://platform.openai.com/docs/)
