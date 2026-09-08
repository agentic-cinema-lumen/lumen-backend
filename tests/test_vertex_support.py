"""Unit tests for Vertex AI support, ADK model creation, and auth resolution."""

import os
import unittest
from unittest.mock import patch

from src.agents.adk_support import agent_model, create_adk_model, has_gemini_key, is_vertex_ai
from src.utils.llm_client import LLMClient


class TestVertexAISupport(unittest.TestCase):
    def test_default_agent_model_is_gemini_37_flash(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(agent_model(), "gemini-3.7-flash")

    def test_custom_agent_model_override(self):
        with patch.dict(os.environ, {"LUMEN_AGENT_MODEL": "gemini-3.7-flash"}):
            self.assertEqual(agent_model(), "gemini-3.7-flash")

    def test_is_vertex_ai_detection(self):
        with patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "true"}):
            self.assertTrue(is_vertex_ai())
        with patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "1"}):
            self.assertTrue(is_vertex_ai())
        with patch.dict(os.environ, {"GOOGLE_GENAI_USE_ENTERPRISE": "true"}):
            self.assertTrue(is_vertex_ai())
        with patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "false", "GOOGLE_GENAI_USE_ENTERPRISE": "false"}):
            self.assertFalse(is_vertex_ai())

    def test_has_gemini_key_vertex_mode(self):
        # In Vertex mode with project set, auth resolves True even without an API key
        with patch.dict(os.environ, {
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "GOOGLE_CLOUD_PROJECT": "my-cinema-project",
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
        }):
            self.assertTrue(has_gemini_key())

        # In Vertex mode without project or credentials, auth resolves False
        with patch.dict(os.environ, {
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "GOOGLE_CLOUD_PROJECT": "",
            "GOOGLE_APPLICATION_CREDENTIALS": "",
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
        }):
            self.assertFalse(has_gemini_key())

    def test_has_gemini_key_ai_studio_mode(self):
        with patch.dict(os.environ, {
            "GOOGLE_GENAI_USE_VERTEXAI": "false",
            "GEMINI_API_KEY": "AIzaSyTestKey",
        }):
            self.assertTrue(has_gemini_key())

    def test_create_adk_model_retry_options(self):
        model = create_adk_model("gemini-3.8-flash")
        self.assertEqual(model.model, "gemini-3.8-flash")
        self.assertIsNotNone(model.retry_options)
        self.assertEqual(model.retry_options.attempts, 3)

    def test_llm_client_vertex_ai_detection(self):
        with patch.dict(os.environ, {
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "GOOGLE_CLOUD_PROJECT": "my-cinema-project",
        }):
            client = LLMClient(force_mock=False)
            self.assertEqual(client.provider, "gemini")
            self.assertEqual(client.model, "gemini-3.7-flash")
            self.assertTrue(client.use_vertex)


if __name__ == "__main__":
    unittest.main()
