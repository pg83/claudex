# anthropic-to-openai-proxy

A single-file Python proxy that translates Anthropic Messages API requests into OpenAI Chat Completions format. Point Claude Code (or any Anthropic API client) at this proxy to use OpenAI-compatible models instead.

## Features

- Streaming and non-streaming responses
- Tool use (function calling)
- Thinking / extended reasoning
- Vision (image inputs)
- System prompts
- Configurable model mapping with short aliases (`opus`, `sonnet`, `haiku`)
- Debug logging of all 4 legs of each request
- Tested with OpenAI API, OpenRouter, and Yandex Eliza

## Quick start

```bash
pip install -r requirements.txt

# List available models at the upstream endpoint
python proxy.py models \
  --openai-base-url https://api.openai.com/v1 \
  --openai-api-key $OPENAI_API_KEY

# Start the proxy
python proxy.py serve \
  --port 8082 \
  --openai-api-key $OPENAI_API_KEY \
  --default-model gpt-5.4

# Use with Claude Code
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY=dummy claude
```

## Model mapping

By default all Claude model names map to `--default-model`. Override per-family with short aliases or per-model with full IDs:

```bash
python proxy.py serve \
  --default-model openai/gpt-5.4 \
  --model-map haiku=openai/gpt-5.4-nano sonnet=openai/gpt-5.4-mini opus=openai/gpt-5.4 \
  ...
```

Also accepts `MODEL_MAP` env var (comma-separated): `MODEL_MAP=haiku=openai/gpt-5.4-nano,opus=openai/gpt-5.4`.

## CLI reference

```
proxy.py serve   Start the proxy server
proxy.py models  List models available at the upstream endpoint
```

### `serve` options

| Flag | Env var | Default | Description |
|------|---------|---------|-------------|
| `--port` | `PROXY_PORT` | `8082` | Listen port |
| `--host` | `PROXY_HOST` | `127.0.0.1` | Listen address |
| `--openai-api-key` | `OPENAI_API_KEY` | — | **Required.** Upstream API key |
| `--openai-base-url` | `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Upstream base URL |
| `--default-model` | `DEFAULT_MODEL` | `gpt-5.4` | Fallback OpenAI model |
| `--model-map` | `MODEL_MAP` | — | Model mapping pairs (see above) |
| `--no-ssl-verify` | `NO_SSL_VERIFY` | `false` | Skip TLS verification |
| `--debug` | `DEBUG` | `false` | Log all requests/responses to stderr |

## Debug logging

Pass `--debug` to see all traffic on stderr:

```
══════ ANTHROPIC REQUEST ══════
POST /v1/messages | stream=true | model=claude-sonnet-4-6
{ ... }

══════ OPENAI REQUEST ══════
POST /chat/completions | model=openai/gpt-5.4
{ ... }

══════ OPENAI RESPONSE ══════
<< SSE: {"choices":[...]}

══════ ANTHROPIC RESPONSE ══════
>> SSE: event=content_block_delta {...}
```

Large fields (base64 images, long text) are automatically truncated.

## Requirements

- Python 3.9+
- fastapi, uvicorn, httpx (see `requirements.txt`)
