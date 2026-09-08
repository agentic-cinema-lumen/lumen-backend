#!/usr/bin/env python3
"""Diagnostic script to test Vertex AI connectivity with Gemini 3.8 Flash."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.env_helper import load_env_file

load_env_file()

def test_vertex():
    print("=" * 60)
    print("🔍 Testing Vertex AI & Gemini Connectivity")
    print("=" * 60)

    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1") or \
                 os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE", "").lower() in ("true", "1")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    model_name = os.environ.get("LUMEN_AGENT_MODEL", "gemini-3.8-flash")
    has_api_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    print(f"• Mode:               {'Vertex AI (GCP)' if use_vertex else 'Google AI Studio'}")
    print(f"• GCP Project:        {project or '(Not set)'}")
    print(f"• Location:           {location}")
    print(f"• Model:              {model_name}")
    print(f"• API Key present:    {has_api_key}")
    print("-" * 60)

    try:
        from google import genai
        from google.genai import types

        print("📡 Initializing google-genai Client...")
        client = genai.Client()
        print(f"✅ Client initialized successfully (backend: {'Vertex AI' if client.vertexai else 'AI Studio'}).")

        print(f"🤖 Sending test prompt to '{model_name}'...")
        response = client.models.generate_content(
            model=model_name,
            contents="Respond with 'Vertex AI is online and operational.' followed by a 1-sentence description of cinematic pacing.",
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=100,
            ),
        )

        print("\n🎉 Response received:")
        print(response.text)
        print("\n✅ Verification successful! No 429 rate limit detected.")
        return 0

    except Exception as e:
        print(f"\n❌ Error during model call: {type(e).__name__}: {e}")
        if "404" in str(e) or "NOT_FOUND" in str(e):
            print("\n💡 Model not found on Vertex AI. Testing available candidates...")
            candidates = [
                "gemini-3.7-flash",
                "gemini-3.6-flash",
                "gemini-3.5-flash",
                "gemini-3.0-flash",
                "gemini-2.5-flash",
                "gemini-2.0-flash",
            ]
            working = []
            for cand in candidates:
                try:
                    print(f"   Trying '{cand}'...", end="", flush=True)
                    res = client.models.generate_content(
                        model=cand,
                        contents="Say hello!",
                        config=types.GenerateContentConfig(max_output_tokens=50),
                    )
                    reply = res.text or (res.candidates[0].content.parts[0].text if res.candidates and res.candidates[0].content and res.candidates[0].content.parts else "")
                    print(f" ✅ Success! Response: {(reply or 'ok').strip()}")
                    working.append(cand)
                except Exception as cand_err:
                    print(f" ❌ {type(cand_err).__name__}: {cand_err}")

            if working:
                print(f"\n🎉 Working models found on Vertex AI: {', '.join(working)}")
                print(f"👉 Newest recommended: Set LUMEN_AGENT_MODEL={working[0]} in your .env file.")
                return 0

            print("\n🔍 Checking available models in this project/location via client.models.list()...")
            try:
                available = []
                for m in client.models.list():
                    name = getattr(m, 'name', str(m))
                    if 'gemini' in name.lower():
                        available.append(name)
                        if len(available) >= 10:
                            break
                if available:
                    print("Available Gemini models in Vertex AI:")
                    for a in available:
                        print(f"   • {a}")
                else:
                    print("No models returned by list().")
            except Exception as list_err:
                print(f"List models failed: {type(list_err).__name__}: {list_err}")
        elif "429" in str(e):
            print("⚠️ Still hitting 429: check if GOOGLE_GENAI_USE_VERTEXAI=true and GOOGLE_CLOUD_PROJECT are set.")
        elif "Could not resolve project" in str(e) or "credentials" in str(e).lower():
            print("💡 Tip: Run 'gcloud auth application-default login' to authenticate your local terminal.")
        return 1

if __name__ == "__main__":
    sys.exit(test_vertex())
