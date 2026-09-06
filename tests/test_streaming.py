"""Unit tests for Event Bus and Real-time SSE Streaming."""
import asyncio
import unittest
from unittest.mock import MagicMock

from src.utils.event_bus import PredictionEventBus, get_event_bus
from src.agents.orchestrator import Submission
from tests.stub_agents import StubResearchAgent, StubSynthesisAgent, stub_orchestrator


class TestEventBus(unittest.TestCase):

    def setUp(self):
        self.bus = PredictionEventBus(buffer_ttl_sec=60.0)

    def test_publish_and_subscribe_history(self):
        pid = "test-pred-001"
        self.bus.publish(pid, "stage", {"stage": "ingestion", "progress": 0.2})
        self.bus.publish(pid, "stage", {"stage": "research", "progress": 0.5})

        history = self.bus.get_history(pid)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["event"], "stage")
        self.assertEqual(history[0]["data"]["stage"], "ingestion")
        self.assertEqual(history[1]["data"]["stage"], "research")

    def test_sse_generator_formatting(self):
        async def _run():
            bus = PredictionEventBus(buffer_ttl_sec=60.0)
            pid = "test-pred-002"

            async def _producer():
                await asyncio.sleep(0.01)
                bus.publish(pid, "stage", {"stage": "ingestion", "progress": 0.2})
                await asyncio.sleep(0.01)
                bus.publish(pid, "complete", {"outcome": "hit", "score": 85})

            asyncio.create_task(_producer())
            chunks = []
            async for chunk in bus.sse_generator(pid, timeout_sec=2.0):
                chunks.append(chunk)

            return "".join(chunks)

        output = asyncio.run(_run())
        self.assertIn("event: stage\ndata: {\"stage\": \"ingestion\", \"progress\": 0.2}\n\n", output)
        self.assertIn("event: complete\ndata: {\"outcome\": \"hit\", \"score\": 85}\n\n", output)


class TestOrchestratorStreaming(unittest.TestCase):

    def test_orchestrator_emits_stages(self):
        research = StubResearchAgent()
        synthesis = StubSynthesisAgent()
        orchestrator = stub_orchestrator(research, synthesis)

        emitted = []

        def _on_event(event_type: str, data: dict):
            emitted.append((event_type, data))

        sub = Submission(
            story="An insomniac detective investigates a mysterious underground society.",
            title="Dark City",
            genre="Mystery",
        )

        report = orchestrator.run(sub, on_event=_on_event)
        self.assertIsNotNone(report)

        # Check emitted stages
        stage_names = [data["stage"] for event_type, data in emitted if event_type == "stage"]
        self.assertIn("ingestion", stage_names)
        self.assertIn("research", stage_names)
        self.assertIn("quant_oracle", stage_names)
        self.assertIn("synthesis", stage_names)
        self.assertIn("finalizing", stage_names)

        # Ingestion before research before quant_oracle before synthesis
        self.assertLess(stage_names.index("ingestion"), stage_names.index("research"))
        self.assertLess(stage_names.index("research"), stage_names.index("quant_oracle"))
        self.assertLess(stage_names.index("quant_oracle"), stage_names.index("synthesis"))


class TestStreamingAPI(unittest.TestCase):

    def setUp(self):
        from fastapi.testclient import TestClient
        import api
        self.api = api
        self.research = StubResearchAgent()
        self.synthesis = StubSynthesisAgent()
        self.api._agent.orchestrator = stub_orchestrator(self.research, self.synthesis)
        self.client = TestClient(api.app)

    def test_post_predictions_stream_returns_sse(self):
        payload = {
            "story": "An insomniac detective investigates a mysterious underground society.",
            "medium": "Feature film",
            "targetGeography": "Nordics",
            "classifications": {"genre": "Mystery", "title": "Dark City"},
            "materials": [],
        }
        res = self.client.post("/v1/predictions/stream", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/event-stream", res.headers.get("content-type", ""))
        body = res.text
        self.assertIn("event: stage", body)
        self.assertIn("event: tool_start", body)
        self.assertIn("event: tool_end", body)
        self.assertIn("event: complete", body)

    def test_non_streaming_endpoint_remains_unaffected(self):
        payload = {
            "story": "An insomniac detective investigates a mysterious underground society.",
            "medium": "Feature film",
            "targetGeography": "Nordics",
            "classifications": {"genre": "Mystery", "title": "Dark City"},
            "materials": [],
        }
        res = self.client.post("/v1/predictions", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn("application/json", res.headers.get("content-type", ""))
        data = res.json()
        self.assertIn("predictionId", data)
        self.assertIn("outcome", data)
        self.assertIn("score", data)
        self.assertIn("agents", data)
        self.assertEqual(len(data["agents"]), 4)

    def test_auth_enforcement_when_lumen_api_key_is_set(self):
        import os
        payload = {
            "story": "An insomniac detective investigates a mysterious underground society.",
            "medium": "Feature film",
            "targetGeography": "Nordics",
            "classifications": {"genre": "Mystery", "title": "Dark City"},
            "materials": [],
        }

        # Set API key in environment
        os.environ["LUMEN_API_KEY"] = "super-secret-lumen-key"
        try:
            # 1. Missing header -> 401
            res1 = self.client.post("/v1/predictions", json=payload)
            self.assertEqual(res1.status_code, 401)
            self.assertIn("Unauthorized", res1.json().get("detail", ""))

            # 2. Wrong header -> 401
            res2 = self.client.post("/v1/predictions", json=payload, headers={"X-Lumen-Key": "wrong-key"})
            self.assertEqual(res2.status_code, 401)

            # 3. Correct header -> 200
            res3 = self.client.post("/v1/predictions", json=payload, headers={"X-Lumen-Key": "super-secret-lumen-key"})
            self.assertEqual(res3.status_code, 200)
            self.assertIn("predictionId", res3.json())
        finally:
            del os.environ["LUMEN_API_KEY"]


if __name__ == "__main__":
    unittest.main()
