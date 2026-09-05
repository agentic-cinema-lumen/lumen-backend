# 🎬 Quant Residual ML Model & Agent Oracle

This package provides the pre-trained, production-ready **Quantitative Residual ML Model** and **QuantOracle** agent tool. It evaluates pre-production screenplay dynamics and visual keyframe metrics to predict expected IMDb ratings, craft residuals ($\text{Rating}_{\text{expected}} - \text{Baseline}_{\text{genre}}$), and diagnose craft vulnerabilities.

---

## ⚡ Quick Start for Agent Developers

```python
from src.quant.oracle import get_oracle

# Instant load (<1ms, no retraining, zero network calls)
oracle = get_oracle()

# 1. Direct Craft Prediction
result = oracle.predict_craft(
    title="Lunar Protocol",
    genre="Sci-Fi",
    dark_frame_ratio=0.50,         # 50% underexposed stills
    pacing_acceleration=1.85,      # 1.85x faster tempo in climax
    cuts_per_minute=16.0,          # cuts per minute
    words_per_minute=35.0          # dialogue density
)

print(f"Projected Rating: {result['expected_rating']} / 10.0")
print(f"Genre Baseline:   {result['genre_baseline_rating']} / 10.0")
print(f"Residual Delta:   {result['craft_residual_delta']:+0.2f}")
print(f"Verdict:          {result['craft_verdict']}")
```

---

## 🤖 Using as an LLM / Agent Tool

The oracle is designed to be passed directly to **Gemini**, **Claude**, **OpenAI**, or **MCP** agent loops.

### 1. Register Tool with LLM
```python
tool_spec = oracle.get_tool_spec()
# Pass tool_spec to your LLM client's tools=[...] array
```

### 2. Dispatch Tool Calls
```python
# In your agent's tool execution loop:
tool_call_result = oracle.execute_tool(tool_name, arguments)
```

The tool parameter schema:
| Argument | Type | Description | Benchmark Median |
| :--- | :--- | :--- | :--- |
| `genre` | `str` | Primary genre (e.g. "Drama", "Sci-Fi", "Action", "Horror") | `6.33` global mean |
| `cuts_per_minute` | `float` | Editing tempo (cuts/min) | `15.5` |
| `pacing_acceleration` | `float` | Third-act climax tempo vs setup acts | `1.05` (neutral) |
| `total_duration_min` | `float` | Runtime in minutes | `120.0` min |
| `words_per_minute` | `float` | Dialogue velocity | `55.0` WPM |
| `dark_frame_ratio` | `float` | Ratio of sub-40 luminance keyframes (0.0 to 1.0) | `0.30` |
| `mean_luminance` | `float` | Average keyframe brightness (0 to 255) | `65.0` |
| `show_historical_mean`| `float` | (Optional) Explicit baseline prior anchor | Auto-derived from genre |

> **Note on Missing Inputs**: Any missing feature is automatically and safely filled with benchmark cinema medians. You can pass as few or as many parameters as available!

---

## 📊 Output Schema Breakdown

Calling `oracle.predict_craft(...)` returns:

```json
{
  "title": "Lunar Protocol",
  "genre": "Sci-Fi",
  "expected_rating": 7.09,
  "genre_baseline_rating": 5.63,
  "craft_residual_delta": 1.46,
  "confidence_interval": {
    "lower": 6.65,
    "upper": 7.53,
    "margin_of_error": 0.441
  },
  "craft_verdict": "EXCEPTIONAL_CRAFT_LIFT",
  "verdict_summary": "Strong craft lift (+1.46 pts above 5.6 baseline). Screenplay pacing and visuals elevate execution.",
  "craft_vulnerabilities": [
    {
      "category": "Cinematography",
      "severity": "HIGH",
      "issue": "Extreme Darkness & Low Lighting Legibility",
      "detail": "50.0% of keyframes are sub-40 luminance.",
      "remedy": "Elevate shadow exposure and mid-tone contrast to prevent streaming compression crush on consumer displays."
    },
    {
      "category": "Pacing & Editing",
      "severity": "HIGH",
      "issue": "Severe Climax Tempo Spike (Rushed Payoff)",
      "detail": "Climax accelerates 1.85x faster than earlier acts, creating narrative disorientation.",
      "remedy": "Inject 2-3 transitional breathing beats in Act 3 to ground character payoff before climax."
    }
  ],
  "top_craft_attributions": [
    {
      "feature": "show_historical_mean",
      "input_value": 5.63,
      "point_impact": -0.595,
      "direction": "negative",
      "interpretation": "Historical genre baseline prior sets foundation with -0.59 star anchor."
    },
    {
      "feature": "total_duration_min",
      "input_value": 120.0,
      "point_impact": -0.065,
      "direction": "negative",
      "interpretation": "Runtime of 120 min contributes -0.07 stars to projected scope."
    }
  ]
}
```

---

## 📦 Ingesting Scripts, Keyframes, and Movie Packages

### Screenplay + Keyframes Directory
```python
result = oracle.predict_from_script_and_keyframes(
    script_path_or_text="data/movies/alien/script.txt",
    keyframes_path_or_dir="data/movies/alien/keyframes",
    genre="Horror, Sci-Fi",
    title="Alien"
)
```

### Full Movie Package
```python
result = oracle.predict_movie_package("data/movies/alien")
print("Projected Rating:", result["expected_rating"])
print("Actual Rating:", result["actual_imdb_rating"])
```

---

## 🛠️ Standalone CLI Commands

You can run the model directly from the terminal:

```bash
# Test with ad-hoc craft flags
python scripts/run_model.py --genre "Sci-Fi" --dark-ratio 0.50 --pacing-acc 1.85

# Test on a full movie package
python scripts/run_model.py --movie data/movies/apocalypse_now

# Print tool schema for LLM function calling
python scripts/run_model.py --tool-spec
```

---

## 📈 Model Architecture & Zero-Leakage Benchmark

- **Algorithm**: Ridge Regression with StandardScaler pipeline (L2 regularized, $\alpha=10.0$)
- **Training Set**: 100 genuine feature films from `data/movies/` (screenplays + 752 spaced widescreen Film-Grab stills)
- **Zero Target Leakage**: Excludes post-release signals (`log_votes`, broadcast timeslots); strictly restricted to pre-production observable screenplay and cinematography features.
- **5-Fold Cross-Validation Metrics**:
  - **MAE**: `0.441` IMDb points
  - **RMSE**: `0.604`
  - **Inference Latency**: `< 0.6` milliseconds
- **Model Artifact**: Persisted at `data/models/champion_model.joblib`
