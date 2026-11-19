# API keys

Keys are read from environment variables and are never stored in this repository.

| Provider prefix | Variable | Notes |
|---|---|---|
| `gemini:` | `GEMINI_API_KEY` | Direct Google Gemini API. **See the expiry note below.** |
| `openrouter:` | `OPENROUTER_API_KEY` | Any model through OpenRouter. |
| `anthropic:` | `ANTHROPIC_API_KEY` | |
| `openai:` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | Any OpenAI-compatible endpoint. |

## STOP using the Gemini key after 19 November 2026

The current `GEMINI_API_KEY` runs on credits that expire on **2026-11-19**.

- Through 2026-11-19: use the direct `gemini:` provider for Gemini roles so the credits are spent.
- From 2026-11-20: **do not use this key.** Switch Gemini roles in `configs/*.json` back to `openrouter:google/<model>`, or to a new key.

`bioloopbench/taskgen/llm.py` enforces this: the `gemini:` backend refuses to make calls after `GEMINI_KEY_STOP_AFTER`. If the key is replaced with one that has no expiry, update that constant (or set the `GEMINI_KEY_STOP_AFTER` environment variable to a new `YYYY-MM-DD` date) and this note.
