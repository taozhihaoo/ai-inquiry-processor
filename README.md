# AI Inquiry Processor

**A lightweight Python service that turns raw customer inquiries into structured triage data — summary, category, priority — via LLM structured outputs, with a CLI, a REST API, SQLite persistence, and idempotent duplicate handling.**

`Python 3.11+` · `OpenAI SDK (structured outputs)` · `FastAPI` · `SQLite` · `pytest (offline)` · `Docker`

> Independent portfolio project. The OpenAI integration is fully implemented and tested against a fake SDK; the default test suite runs completely offline. Demo mode requires no API credentials. Real OpenAI calls require you to provide your own key.

---

## Features

- **Structured outputs, enforced twice** — requests pin a strict `json_schema` response format; every response is independently re-validated (missing fields, wrong types, off-enum values become per-row errors, never corrupt data).
- **Provider abstraction** — a tiny `LLMClient` protocol with two implementations: `OpenAIClient` (official SDK) and `MockLLMClient` (deterministic, offline). Adding a vendor = one class.
- **Idempotent processing** — every inquiry gets a stable SHA-256 content identity. Resubmitting the same inquiry (via CLI or API) returns the stored result without calling the LLM again.
- **Durable storage** — SQLite (stdlib `sqlite3`, parameterized queries, no ORM) with per-inquiry records, token usage, and batch runs.
- **Reliability** — per-call timeout, retry with exponential backoff, `Retry-After` honored, retryable-vs-permanent error taxonomy, single-item failure isolation.
- **REST API + CLI on one service layer** — both entry points call the same `InquiryService`; no duplicated business logic.
- **Usage tracking** — token counts and latency persisted when the provider reports them; `null` when it doesn't (nothing is fabricated). Optional cost estimation from a configured price table.
- **139 offline tests** — unit + integration + API tests with a fake SDK; deterministic, no network, no credentials.

## Architecture

```
                ┌────────────┐      ┌──────────────┐
                │    CLI     │      │  FastAPI      │
                │ (src.main) │      │  (src.api)    │
                └─────┬──────┘      └──────┬────────┘
                      │  same entry point  │
                      ▼                    ▼
                ┌─────────────────────────────────┐
                │      InquiryService (src.service)│   idempotency · runs ·
                │      single business entry       │   usage/cost · persistence
                └───────────────┬─────────────────┘
                                ▼
                ┌─────────────────────────────────┐
                │   InquiryProcessor (src.processor)│  per-item error isolation
                └───────────────┬─────────────────┘
                                ▼
                     LLMClient protocol (src.llm_client)
                     ├── OpenAIClient  (official SDK, strict json_schema,
                     │                  timeout, retry + backoff, Retry-After)
                     └── MockLLMClient (deterministic offline rules)
                                ▼
                ┌─────────────────────────────────┐
                │   InquiryStore (src.storage)     │  SQLite: inquiries + runs
                └─────────────────────────────────┘
```

Data flow: input (CSV row / HTTP request) → content-hash identity → pending row in SQLite → provider call with retries → strict schema validation → success/error update → report or HTTP response.

## CLI

```bash
python -m src.main --demo                     # offline demo (mock provider, no key)
python -m src.main --input inquiries.csv      # real run (reads OPENAI_API_KEY)
python -m src.main --demo --db data/demo.db --format csv -o reports
python -m src.main --help / --version
```

The CLI loads the CSV, hands everything to `InquiryService`, then renders JSON/CSV reports from the persisted records and prints the run summary:

```
run: e9c72e5cf0a94ecbb6f1ccf0f32db2a5
processed 8 inquiries: 8 succeeded, 0 failed, 0 duplicates
```

Input CSV: header must contain `customer_name` and `message` (see [`sample_inquiries.csv`](sample_inquiries.csv) — 8 fictional inquiries covering all categories and priorities). Exit codes: `0` ok · `1` input error · `2` config error.

## REST API

Start: `uvicorn src.api:app --host 127.0.0.1 --port 8000` (interactive docs at `/docs`).

| Method | Path | Purpose | Success | Errors |
| --- | --- | --- | --- | --- |
| GET | `/health` | liveness + database + provider | `200` | — |
| POST | `/inquiries` | process one inquiry (idempotent) | `201` | `422` validation |
| POST | `/inquiries/batch` | process up to 1000 inquiries as one run | `200` | `422` validation |
| GET | `/inquiries/{id}` | fetch a stored result | `200` | `404` unknown id |

```bash
# process one inquiry
curl -X POST http://127.0.0.1:8000/inquiries \
  -H "Content-Type: application/json" \
  -d '{"customer_name": "Alice", "message": "I cannot log into my account."}'

# → 201
{
  "id": "39d8b8625a967e10279b37d9a8bf55b8dde67b93a68423652b4dcca829bd3b42",
  "status": "success",
  "summary": "Alice: I cannot log into my account after resetting my password.",
  "category": "Technical Support",
  "priority": "Medium",
  "error_message": null,
  "duplicate": false,
  "created_at": "2026-09-29T15:31:16.641+00:00",
  "updated_at": "2026-09-29T15:31:16.650+00:00"
}
```

Behaviour notes (all covered by tests):

- **Duplicates**: resubmitting identical content returns the stored result with `"duplicate": true` — the LLM is not called again. Failed rows are also remembered: a known-bad inquiry is not re-processed.
- **Batch partial failure**: one bad row never fails the batch. `POST /inquiries/batch` answers `200` with per-row `status` and run-level counts:

```json
{
  "run_id": "28777141f7584cfaa1d42f83ca4b1fab",
  "total": 3, "succeeded": 2, "failed": 1, "duplicates": 0,
  "results": [ ... ]
}
```

- **Safe errors**: provider/storage failures become JSON like `{"error": {"code": "storage_error", "message": "database operation failed"}}` — no tracebacks, no internals, no keys.

There is deliberately **no authentication** on this API; see [Limitations](#limitations).

## Data Model

SQLite schema (created automatically, idempotent):

- **`inquiries`** — `id` (PK, SHA-256 of normalized `customer_name` + `message`), `customer_name`, `message`, `status` (`pending`/`success`/`error`), `summary`, `category`, `priority`, `error_message`, `run_id`, `provider`, `model`, `input_tokens`, `output_tokens`, `total_tokens`, `latency_ms`, `attempts`, `estimated_cost`, `created_at`, `updated_at`.
- **`processing_runs`** — `run_id`, `provider`, `model`, `started_at`, `finished_at`, `total`, `succeeded`, `failed`.

`category` is always one of `Sales | Technical Support | Billing | General Question`; `priority` is `Low | Medium | High` — enforced in the request-side JSON schema *and* by response-side validation.

## LLM Providers

Selected via `LLM_PROVIDER` (or `--provider` / `--demo`):

| Provider | Value | Needs key | Behaviour |
| --- | --- | --- | --- |
| OpenAI | `openai` | yes (`OPENAI_API_KEY`) | `gpt-4o-mini` by default; strict `json_schema` structured outputs; SDK internal retries disabled (this project owns the retry policy) |
| Mock | `mock` | no | deterministic keyword-based classification; identical input → identical verdict; used by `--demo`, tests, and the default Docker image |

`MockLLMClient` reports `usage: null` because no real tokens are consumed — nothing is invented.

## Configuration

All configuration is environment-based (see [`.env.example`](.env.example)); `.env` files are supported but real environment variables always win.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` (CLI) / `mock` (bare `uvicorn` & Docker) | provider selection |
| `OPENAI_API_KEY` | — | API key; never hardcoded, never logged |
| `OPENAI_MODEL` | `gpt-4o-mini` | model name |
| `OPENAI_BASE_URL` | SDK default | point at any OpenAI-compatible endpoint |
| `LLM_TIMEOUT_SECONDS` | `30` | per-call timeout |
| `LLM_MAX_RETRIES` | `3` | retries after the first attempt |
| `LLM_RETRY_INITIAL_DELAY` / `LLM_RETRY_MAX_DELAY` | `1.0` / `10.0` | exponential backoff bounds |
| `AI_INQUIRY_DB_PATH` | `data/inquiries.db` | SQLite location (`:memory:` for tests) |
| `AI_INQUIRY_OUTPUT_DIR` | `output` | report directory |
| `LLM_PRICE_INPUT_PER_MTOK` / `LLM_PRICE_OUTPUT_PER_MTOK` | unset | USD per 1M tokens; **both** required for `estimated_cost` |

## Running Locally

```bash
git clone https://github.com/taozhihaoo/ai-inquiry-processor.git
cd ai-inquiry-processor
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; .venv/bin/activate elsewhere
pip install -r requirements.txt

python -m src.main --demo            # CLI demo
uvicorn src.api:app --port 8000      # API (defaults to mock provider)
```

## Running with Docker

```bash
docker compose up          # mock provider, http://127.0.0.1:8000
curl http://127.0.0.1:8000/health
```

- Runs as a non-root user; SQLite and reports live on named volumes (`/app/data`, `/app/output`).
- `.env` is never copied into the image (`.dockerignore`); pass secrets at runtime only:

```bash
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... docker compose up
```

- A `HEALTHCHECK` hits `/health` every 30s.

## Testing

```bash
pytest            # 139 tests, fully offline: fake SDK + mock provider + tmp SQLite
```

Layers covered: schema validation · CSV validation · retry/backoff (incl. `Retry-After`) · permanent-error fail-fast · idempotency & duplicates · SQLite round-trips & failure handling · service-level batch isolation · full HTTP contract (success, validation 422, duplicate, 404, storage-failure 500, provider-error rows) · CLI→SQLite→JSON-report integration. No test ever calls the real API.

CI (`.github/workflows/ci.yml`) runs an import check plus the full suite on Python 3.11/3.12/3.13 with `OPENAI_API_KEY` explicitly blanked.

## Demo Mode

`--demo` (or the default Docker image / bare `uvicorn`) uses `MockLLMClient`: the complete input → processing → persistence → report flow runs without any credentials and is fully deterministic. Demo and real mode share every code path except the provider class.

## OpenAI API Usage

The integration is **implemented but was not executed in the environment where this README was last verified** (no `OPENAI_API_KEY` present — stated here deliberately, not faked). To run for real:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key       # from https://platform.openai.com/api-keys
python -m src.main --input your_inquiries.csv
```

Usage metadata (`input_tokens` / `output_tokens` / `total_tokens`, latency, attempts) is stored per inquiry when the provider reports it. If you set both price variables, `estimated_cost` is computed from them — it is an **estimate**, not a billing figure, and stays `null` when usage or prices are unavailable.

## Error Handling

| Failure | Handling |
| --- | --- |
| Rate limit (429) / network / 5xx / timeout | retried with exponential backoff; provider `Retry-After` honored |
| Bad credentials, malformed/truncated/refused responses | `LLMPermanentError` — fail fast, no retry |
| One inquiry fails | row marked `error` with reason; batch and API continue |
| Resubmission of a failed inquiry | stored error returned, LLM not re-called (documented semantics) |
| Database failure | wrapped as `StorageError`; API returns an opaque `500 storage_error` |
| Invalid API input | FastAPI/Pydantic `422` with field-level details |

## Security

- API keys are read **only** from environment variables — never hardcoded, logged, or written to reports; `.env` is git-ignored and docker-ignored; `.env.example` ships placeholders only.
- Customer text travels as JSON data in the user message with a fixed system prompt; a test asserts injection attempts cannot alter the instructions.
- `output/` and `data/` (reports, databases) are git-ignored.
- Parameterized SQL only; sample data is fictional.
- No authentication on the API — bind it to localhost or put it behind your own gateway.

## Project Structure

```
ai-inquiry-processor/
├── README.md · LICENSE · requirements.txt · pyproject.toml
├── .env.example · .gitignore · .dockerignore
├── Dockerfile · docker-compose.yml
├── .github/workflows/ci.yml
├── sample_inquiries.csv        # fictional sample data
├── src/
│   ├── config.py               # env → AppConfig (single source of configuration)
│   ├── models.py               # dataclasses, JSON schema, SHA-256 identity
│   ├── llm_client.py           # protocol, OpenAI + mock providers, retry policy
│   ├── csv_loader.py           # CSV validation
│   ├── processor.py            # per-item error isolation
│   ├── storage.py              # SQLite data access (inquiries + runs)
│   ├── service.py              # InquiryService: idempotency, runs, usage/cost
│   ├── report_writer.py        # JSON/CSV reports
│   ├── api.py                  # FastAPI layer (thin, delegates to service)
│   └── main.py                 # CLI (delegates to service)
├── tests/                      # 139 offline tests
├── data/                       # SQLite (created at runtime, git-ignored)
└── output/                     # reports (created at runtime, git-ignored)
```

## Limitations

Stated plainly, so nobody has to guess:

- **Single-process SQLite** — fine for a portfolio service / small teams; concurrent multi-instance writes would need a real database server.
- **No authentication / rate limiting on the API** — intentional scope cut for a lightweight service.
- **No background workers or queues** — batch endpoints process synchronously (bounded at 1000 items).
- **Synchronous LLM calls** — throughput scales with serial latency; no concurrency yet.
- **Cost values are estimates** computed from user-configured prices, never from live billing.
- The OpenAI path is verified against a fake SDK in tests; it has not been executed against the live API in the environment where this README was last updated.
- Not "production-ready", "enterprise-grade", or "battle-tested" — it is a well-tested, honestly-scoped portfolio project.

## License

[MIT](LICENSE)
