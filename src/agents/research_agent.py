"""ResearchAgent — the judgement code cannot do, grounded in retrieved text.

Replaces `TropeSleuth._extract_claims_from_snippet`, which keyword-matched a
snippet and then returned a hardcoded claim sentence. Here a Gemini agent
formulates the queries, reads what came back, and writes the claims; code then
throws away any claim it cannot trace to text that was actually retrieved.

The agent holds one tool, `parallel_search(query)`, and researches three modes:
craft precedent, trope fatigue, comparable reception. When no genre is supplied
it names one from the logline as its first structured output.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.agents.adk_support import (
    ToolEventLog,
    agent_model,
    create_adk_model,
    has_gemini_key,
    run_agent,
    stop_after_output,
)
from src.search.parallel_search_client import ParallelSearchClient

RETRIEVAL_MODES = ("craft_precedent", "trope_fatigue", "comparable_reception")

OUTPUT_KEY = "research"


class Claim(BaseModel):
    category: str = Field(description="One of: " + ", ".join(RETRIEVAL_MODES))
    valence: str = Field(description="positive, negative or neutral")
    claim: str = Field(description="One sentence, in your own words.")
    evidence: str = Field(
        description="A verbatim span copied from the retrieved text, not a paraphrase."
    )
    source_url: str = Field(description="The URL the span came from, exactly as retrieved.")


class ResearchOutput(BaseModel):
    inferred_genre: str = Field(description="Genre named from the logline.")
    queries_executed: List[str] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)


INSTRUCTION = """You are a film research analyst preparing a pre-mortem for a producer.

You have one tool: parallel_search(query). Use it. You may call it several times.

Research these three modes, in this order:
1. craft_precedent — released films that shipped with the flagged craft traits below,
   and what happened to their reception.
2. trope_fatigue — whether the premise's central conceit is currently saturated.
3. comparable_reception — comparable films you name from the logline, and how they
   were received.

Then answer with structured output.

Hard rules, because a producer will make a real decision on this:
- Every claim MUST carry `evidence`: a span copied VERBATIM, character for character,
  from the text a search returned. Never paraphrase into the evidence field. Never
  write evidence for a page you did not retrieve.
- `source_url` MUST be one of the URLs a search returned, copied exactly.
- If the searches returned nothing usable, return an empty claims list. An empty list
  is a correct answer. An invented claim is not.
- Set `category` to one of: craft_precedent, trope_fatigue, comparable_reception.
- `queries_executed` lists the queries you actually ran.
- Name `inferred_genre` from the logline even when a genre was given, in which case
  repeat the given genre.
"""


def _normalise(text: str) -> str:
    """Whitespace-collapsed, casefolded, for substring comparison."""
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def filter_claims(
    claims: List[Claim], retrieved: Dict[str, str]
) -> Tuple[List[Claim], List[str]]:
    """Drop every claim that does not trace to retrieved text.

    A claim survives only when its `source_url` is among the URLs a search
    returned AND its `evidence` is a span of the text retrieved for that URL.
    """
    haystacks = {url: _normalise(text) for url, text in retrieved.items()}
    kept: List[Claim] = []
    dropped: List[str] = []
    for c in claims:
        if c.source_url not in haystacks:
            dropped.append(f"{c.source_url!r} is not among retrieved URLs")
            continue
        if not c.evidence or _normalise(c.evidence) not in haystacks[c.source_url]:
            dropped.append(f"evidence for {c.source_url!r} is not a span of the retrieved text")
            continue
        kept.append(c)
    return kept, dropped


class ResearchAgent:
    """Gemini via ADK, one search tool, output filtered against what was retrieved."""

    def __init__(
        self,
        search_client: Optional[ParallelSearchClient] = None,
        model: Optional[str] = None,
        max_iterations: int = 3,
        num_results: int = 4,
    ):
        # ponytail: ParallelSearchClient already caches on a hash of the query
        # string under data/cache/search/, which is exactly the required cache.
        # A second cache layer here would be two sources of truth.
        self.client = search_client or ParallelSearchClient()
        self.model = model or agent_model()
        self.max_iterations = max_iterations
        self.num_results = num_results

    # ------------------------------------------------------------------ tool

    def _search_tool(self, retrieved: Dict[str, str], queries: List[str], mocked: List[bool]):
        client = self.client
        num_results = self.num_results

        def parallel_search(query: str) -> dict:
            """Search the live web for film reception, craft precedent and trope discussion.

            Args:
                query: A web search query.

            Returns:
                The retrieved pages, each with a url, title and text.
            """
            queries.append(query)
            raw = client.search(query, num_results=num_results)
            # a cached mock is still a mock: check the payload flag, not just _source
            mocked.append(bool(raw.get("_mock")) or raw.get("_source") == "smart_mock")
            results = []
            for item in raw.get("results", []):
                url = item.get("url")
                text = item.get("text") or item.get("snippet") or ""
                if not url:
                    continue
                retrieved[url] = (retrieved.get(url, "") + " " + text).strip()
                results.append({"url": url, "title": item.get("title", ""), "text": text})
            return {"results": results}

        return parallel_search

    # ------------------------------------------------------------------ agent

    def build_loop(self, search_tool: Callable, on_event: Optional[Callable] = None):
        """The bounded research loop: one LlmAgent, at most `max_iterations` passes."""
        from google.adk.agents import LlmAgent, LoopAgent

        log = ToolEventLog(on_event=on_event)
        researcher = LlmAgent(
            name="research_agent",
            model=create_adk_model(self.model),
            description="Researches craft precedent, trope fatigue and comparable reception.",
            instruction=INSTRUCTION,
            tools=[search_tool],
            output_schema=ResearchOutput,
            output_key=OUTPUT_KEY,
            before_tool_callback=log.before,
            after_tool_callback=log.after,
            after_agent_callback=stop_after_output("research_complete"),
        )
        loop = LoopAgent(
            name="research_loop", sub_agents=[researcher], max_iterations=self.max_iterations
        )
        loop._lumen_tool_log = log  # tests and the orchestrator read the events off this
        return loop

    def _invoke(
        self, prompt: str, search_tool: Callable, on_event: Optional[Callable] = None
    ) -> List[ResearchOutput]:
        """One ADK run; returns the structured output of every loop iteration."""
        loop = self.build_loop(search_tool, on_event=on_event)
        raw = run_agent(loop, prompt, OUTPUT_KEY)
        self._tool_events = list(loop._lumen_tool_log.events)
        return [ResearchOutput(**r) if isinstance(r, dict) else r for r in raw]

    # ------------------------------------------------------------------ run

    def run(
        self,
        premise: str,
        risk_flags: Optional[List[Dict[str, Any]]] = None,
        genre: Optional[str] = None,
        title: str = "Untitled Submission",
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        retrieved: Dict[str, str] = {}
        queries: List[str] = []
        mocked: List[bool] = []
        self._tool_events: List[Dict[str, Any]] = []
        reasons: List[str] = []

        if not has_gemini_key():
            reasons.append("no GEMINI_API_KEY: the research agent did not run")
            outputs: List[ResearchOutput] = []
        else:
            tool = self._search_tool(retrieved, queries, mocked)
            try:
                outputs = self._invoke(
                    self._prompt(premise, risk_flags or [], genre, title),
                    tool,
                    on_event=on_event,
                )
            except Exception as exc:  # a failed Gemini call degrades the run
                reasons.append(f"research agent call failed: {type(exc).__name__}: {exc}".strip(": "))
                outputs = []

        claims: List[Claim] = []
        dropped: List[str] = []
        for out in outputs:
            kept, drops = filter_claims(out.claims, retrieved)
            claims.extend(kept)
            dropped.extend(drops)

        if getattr(self.client, "force_mock", False) or (mocked and all(mocked)):
            reasons.append("search results are mock output, not live retrieval")
        proposed = sum(len(o.claims) for o in outputs)
        if dropped:
            reasons.append(
                f"{len(dropped)} of {proposed} claim(s) dropped by the grounding filter"
                + ("; none survived" if not claims else "")
            )
        # An agent that searched and then answered with an empty claims list is
        # obeying its instruction, but the producer still gets nothing sourced.
        # Silence about that is what made a zero-claim run look like a clean one.
        if not outputs and not reasons:
            reasons.append("research agent produced no structured output")
        elif outputs and proposed == 0:
            reasons.append("the research agent returned no claims at all")

        inferred = genre or next((o.inferred_genre for o in outputs if o.inferred_genre), None)
        return {
            "claims": [c.model_dump() for c in claims],
            "dropped": dropped,
            "queries_executed": queries or [q for o in outputs for q in o.queries_executed],
            "retrieved_urls": list(retrieved),
            "inferred_genre": inferred,
            "tool_events": self._tool_events,
            # a partial drop is the filter working; losing every claim is not
            "degraded": bool(reasons) and not (len(reasons) == 1 and dropped and claims),
            "degradation_reasons": reasons,
        }

    def _prompt(
        self, premise: str, risk_flags: List[Dict[str, Any]], genre: Optional[str], title: str
    ) -> str:
        flags = "\n".join(
            f"- [{f.get('severity', 'FLAG')}] {f.get('category')}: {f.get('issue')} — {f.get('detail')}"
            for f in risk_flags
        ) or "- none flagged by the craft model"
        return (
            f"Title: {title}\n"
            f"Genre: {genre or 'NOT SUPPLIED — name it from the logline'}\n"
            f"Logline / premise:\n{premise}\n\n"
            f"Craft traits the deterministic model flagged:\n{flags}\n\n"
            "Research the three modes and answer with structured output."
        )
