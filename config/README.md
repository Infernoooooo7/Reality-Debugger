# Configuration

All tunable computer-vision, tracking, temporal, diagnostic and AI-usage
parameters live here, in one place, shared by the frontend (imported at build
time and validated with Zod) and the backend (loaded at start-up and
validated with Pydantic). Secrets and deployment settings (API keys, provider
choice, upload limits) stay in environment variables / `.env`.

Every parameter is an object:

```json
"scoreThreshold": {
  "value": 0.3,
  "unit": "probability",
  "purpose": "What the parameter controls.",
  "source": "empirical",
  "ref": "Where the value comes from."
}
```

`source` is one of:

| Source | Meaning |
| --- | --- |
| `model-doc` | taken from the model's documentation, metadata or official reference code |
| `paper` | taken from a published paper or its official implementation |
| `empirical` | measured in this project (see `docs/benchmarks/` and `docs/COMPUTER_VISION_RESEARCH.md`) |
| `default` | a library default kept on purpose |
| `heuristic` | a design choice with no published reference; documented reasoning, safe to tune |

| File | Used by | Contents |
| --- | --- | --- |
| `vision.json` | frontend | detector selection, frame rates, capture sizes, pixel-signal parameters |
| `detection.json` | frontend | score thresholds, NMS, deep-detector pre/post-processing, fusion |
| `tracking.json` | frontend | ByteTrack-style association thresholds, Kalman noise, track lifetimes |
| `temporal.json` | frontend + backend | movement/persistence/occlusion thresholds, scene change, observation cadence, finding lifecycle, video sampling |
| `diagnostics.json` | backend | local diagnostic rule thresholds, severities and scoring |
| `ai.json` | frontend + backend | when the optional AI reasoning layer is called, cooldowns, de-duplication, caching, image compression |
| `ontology_roots.json` | `tools/build_ontology.py` | WordNet roots that define semantic attributes |
| `ontology.generated.json`, `lexicon.generated.json` | frontend + backend | generated - do not edit; run `python tools/build_ontology.py` |
