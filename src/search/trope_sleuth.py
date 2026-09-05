"""
🎬 TROPE SLEUTH & AUDIENCE CLAIM EXTRACTOR 🎬

Formulates hypothesis-driven Parallel Search queries for:
- Script & Pacing analysis (Mode 1): Trope risks, 3rd act pacing traps, visual darkness precedents
- Premise & Pitch analysis (Mode 2): Genre fatigue, clichés, comparable show audience consensus
Normalizes findings into the standardized Claim Taxonomy.
"""

from typing import Dict, Any, List, Optional
from src.search.parallel_search_client import ParallelSearchClient


class TropeSleuth:
    """Investigates audience reception, tropes, and craft precedents via Parallel Search."""

    def __init__(self, search_client: Optional[ParallelSearchClient] = None):
        self.client = search_client or ParallelSearchClient()

    def investigate_script_craft(
        self,
        script_metrics: Dict[str, Any],
        vision_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Formulate targeted Parallel Search queries based on detected script & visual craft traits.
        """
        queries = []
        climax_acc = script_metrics.get("climax_acceleration", 1.0)
        wpm = script_metrics.get("words_per_minute", 110.0)
        dark_ratio = vision_metrics.get("dark_frame_ratio", 0.1)

        # 1. Darkness / Cinematography query
        if dark_ratio >= 0.35:
            queries.append(
                f"site:reddit.com/r/television \"too dark\" \"cinematography\" \"can't see\" battle episode discussion"
            )
        else:
            queries.append(
                f"site:reddit.com/r/television \"cinematography\" \"lighting\" visual masterpiece episode"
            )

        # 2. Pacing & Climax query
        if climax_acc >= 1.7:
            queries.append(
                f"site:reddit.com/r/television \"rushed\" \"climax\" \"pacing\" season finale complaints"
            )
        elif climax_acc <= 0.8:
            queries.append(
                f"site:reddit.com/r/television \"dragging\" \"slow burn\" \"anti-climax\" episode reaction"
            )
        else:
            queries.append(
                f"site:reddit.com/r/television \"pacing\" \"climax acceleration\" episode discussion"
            )

        # 3. Dialogue velocity / Bottle episode query
        if wpm >= 140:
            queries.append(
                f"site:reddit.com/r/television \"rapid fire dialogue\" prestige drama episode reaction"
            )
        elif wpm <= 70:
            queries.append(
                f"site:reddit.com/r/television \"lack of dialogue\" \"silent episode\" audience reception"
            )

        search_dossier = []
        structured_claims = []

        for q in queries:
            search_res = self.client.search(q, num_results=3)
            search_dossier.append(search_res)
            for item in search_res.get("results", []):
                claims = self._extract_claims_from_snippet(item, "script_mode")
                structured_claims.extend(claims)

        return {
            "mode": "script_investigation",
            "queries_executed": queries,
            "claims_count": len(structured_claims),
            "claims": structured_claims,
            "raw_search_dossier": search_dossier
        }

    def investigate_premise_market(
        self,
        premise_info: Dict[str, Any],
        vision_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Formulate targeted Parallel Search queries for a pitch/logline.
        Hunts for audience fatigue, comparable successes/failures, and cliché traps.
        """
        genre = premise_info.get("genre", "Drama")
        tone = premise_info.get("tone", "Dramatic")
        logline = premise_info.get("logline", "")

        queries = [
            f"site:reddit.com/r/television \"{genre}\" \"cliché\" OR \"overdone\" tropes to avoid",
            f"site:reddit.com/r/television \"{genre}\" \"what makes a great\" pitch OR pilot",
            f"site:reddit.com/r/television \"{tone}\" \"audience fatigue\" review discussion"
        ]

        search_dossier = []
        structured_claims = []

        for q in queries:
            search_res = self.client.search(q, num_results=3)
            search_dossier.append(search_res)
            for item in search_res.get("results", []):
                claims = self._extract_claims_from_snippet(item, "premise_mode")
                structured_claims.extend(claims)

        return {
            "mode": "premise_market_sleuth",
            "queries_executed": queries,
            "claims_count": len(structured_claims),
            "claims": structured_claims,
            "raw_search_dossier": search_dossier
        }

    def _extract_claims_from_snippet(self, search_item: Dict[str, Any], context_mode: str) -> List[Dict[str, Any]]:
        """Normalize raw web snippet into standardized Claim Taxonomy."""
        snippet = search_item.get("snippet", "")
        title = search_item.get("title", "")
        combined = f"{title} {snippet}".lower()
        claims = []

        if "dark" in combined or "lighting" in combined or "compression" in combined:
            claims.append({
                "category": "Cinematography",
                "valence": "negative" if ("too dark" in combined or "unwatchable" in combined or "grey blocks" in combined or "underexposed" in combined) else "positive",
                "claim": "Extreme darkness and low contrast trigger severe audience backlash on consumer screens and streaming compression.",
                "evidence": snippet[:180],
                "source": search_item.get("url")
            })

        if "pacing" in combined or "rushed" in combined or "climax" in combined:
            claims.append({
                "category": "Pacing",
                "valence": "negative" if ("rushed" in combined or "compression" in combined or "cheated" in combined or "dragging" in combined) else "positive",
                "claim": "Sudden pacing spikes in the third act without adequate narrative setup create severe emotional disconnect.",
                "evidence": snippet[:180],
                "source": search_item.get("url")
            })

        if "dialogue" in combined or "chamber" in combined or "courtroom" in combined:
            claims.append({
                "category": "Plot Logic",
                "valence": "positive" if ("sharp" in combined or "witty" in combined or "measure of a man" in combined) else "negative",
                "claim": "Single-location or chamber episodes require high dialogue density and razor-sharp ideological conflict to hold viewer attention.",
                "evidence": snippet[:180],
                "source": search_item.get("url")
            })

        if not claims:
            claims.append({
                "category": "Payoff",
                "valence": "neutral",
                "claim": "High-prestige execution requires close harmony between visual language and narrative stakes.",
                "evidence": snippet[:180],
                "source": search_item.get("url")
            })

        return claims
