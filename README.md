# AI Inquiry Processor

**A lightweight Python CLI that turns a CSV of raw customer inquiries into a structured triage report — summary, category, and priority — powered by LLM structured outputs with strict schema validation, retry/backoff resilience, and a fully offline demo mode.**

`Python 3.11+` · `OpenAI SDK` · `Structured Outputs (JSON schema)` · `pytest` · `100% offline tests`

---

## What it does

Feed it a CSV of customer messages. For every row it calls an LLM to produce a
**summary**, a **category** (`Sales` / `Technical Support` / `Billing` /
`General Question`) and a **priority** (`Low` / `Medium` / `High`), validates
the response against a strict JSON schema, and writes machine-readable reports.

```
input CSV ──▶ csv_loader ──▶ InquiryProcessor ──▶ report_writer ──▶ inquiries_report.json
                                   │                                 └─▶ inquiries_report.csv
                                   │  LLMClient (protocol)
                                   ├─▶ OpenAIClient  — official SDK, json_schema response format,
                                   │                   timeout, retry + exponential backoff
                                   └─▶ MockLLMClient — deterministic offline demo (--demo)
```

One row failing (rate limit, bad response, anything else) **never crashes the
batch**: every input ends up in the report with a `success` or `error` status.

## Tech stack

| Concern | Choice |
| --- | --- |
| Language | Python 3.11+ (dataclasses, `typing.Protocol`) |
| LLM access | Official OpenAI Python SDK behind a provider-agnostic `LLMClient` protocol |
| Structured output | OpenAI *structured outputs* (`response_format` strict `json_schema`) + independent schema re-validation |
| Resilience | Per-call timeout, retry with exponential backoff (capped), retryable-vs-permanent error taxonomy |
| CLI | `argparse` (`python -m src.main` or the `ai-inquiry` console script) |
| Config | Environment variables + optional `.env`, separated from business logic in `src/config.py` |
| Tests | `pytest`, all API interactions mocked — the whole suite runs offline and deterministically |

## Installation

```bash
git clone https://github.com/taozhihaoo/ai-inquiry-processor.git
cd ai-inquiry-processor

python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash) — use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Optional: `pip install -e .` also installs the `ai-inquiry` console command.

## Environment variables

Copy `.env.example` to `.env` and fill in your key — or export the variables directly.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | for real runs | — | API key; read **only** from the environment, never hardcoded |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | Model name |
| `OPENAI_BASE_URL` | no | SDK default | Point at any OpenAI-compatible endpoint |
| `LLM_TIMEOUT_SECONDS` | no | `30` | Per-call timeout |
| `LLM_MAX_RETRIES` | no | `3` | Retries after the first attempt |
| `LLM_RETRY_INITIAL_DELAY` | no | `1.0` | First backoff delay (seconds, doubles up to the max) |
| `LLM_RETRY_MAX_DELAY` | no | `10.0` | Backoff ceiling |

`.env` is developer convenience only — real environment variables always win,
and `.env` is git-ignored.

## Running

```bash
# 1. Offline demo — no API key needed, deterministic mock provider
python -m src.main --demo

# 2. Real run against the OpenAI API (reads OPENAI_API_KEY)
python -m src.main --input your_inquiries.csv

# 3. Useful options
python -m src.main --demo --format csv -o reports --log-level DEBUG
python -m src.main --help
python -m src.main --version
```

Exit codes: `0` success · `1` input error (missing/invalid CSV) · `2` config
error (e.g. `OPENAI_API_KEY` not set — the message reminds you `--demo` works without one).

### Input CSV format

Header must contain `customer_name` and `message` (extra columns are ignored,
a UTF-8 BOM is tolerated, blank rows are skipped):

```csv
customer_name,message
Alice,"I cannot log into my account after resetting my password."
Bob,"I would like to know whether you offer annual billing."
Carla,"Our entire team has been locked out since this morning and we cannot process any orders. This is urgent."
```

The bundled [`sample_inquiries.csv`](sample_inquiries.csv) contains eight
fictional inquiries covering every category and priority level.

### Output example

`output/inquiries_report.json` (abridged):

```json
{
  "report_metadata": {
    "generated_at_utc": "2026-09-29T12:00:00+00:00",
    "tool": "ai-inquiry-processor",
    "tool_version": "1.0.0",
    "provider": "openai",
    "model": "gpt-4o-mini"
  },
  "summary": { "total": 8, "succeeded": 8, "failed": 0, "success_rate": "100%" },
  "results": [
    {
      "row_number": 2,
      "customer_name": "Alice",
      "message": "I cannot log into my account after resetting my password.",
      "status": "success",
      "analysis": {
        "summary": "Alice: I cannot log into my account after resetting my password.",
        "category": "Technical Support",
        "priority": "High"
      },
      "error": null
    }
  ]
}
```

`output/inquiries_report.csv` — one flat row per inquiry:

```csv
row_number,customer_name,status,summary,category,priority,error
2,Alice,success,Alice: I cannot log into my account after resetting my password.,Technical Support,High,
```

Failed rows carry empty analysis fields and a populated `error` column, and
`row_number` traces every result back to the exact source line.

## Design notes

- **Provider abstraction** — everything downstream depends only on the
  `LLMClient` protocol (`analyze(customer_name, message) -> dict`). Swapping
  providers is one class; the processor, reports and CLI never change:

  ```python
  class AnthropicClient:            # drop-in future provider
      def analyze(self, customer_name: str, message: str) -> dict: ...
  ```

- **Structured outputs, twice** — the request pins a strict `json_schema`
  (`response_format`), and the response is *independently* re-validated
  (`InquiryAnalysis.from_dict`): missing fields, wrong types or off-enum
  values become per-row errors instead of corrupt data.
- **Instruction/data separation** — the system prompt is fixed; each inquiry
  travels in the user message as a JSON data payload, so text inside a
  customer message can't reshape the instructions (covered by a test).
- **Error taxonomy** — rate limits / network errors / 5xx are `LLMRetryableError`
  (retried with exponential backoff, capped); bad credentials, truncated or
  malformed responses are `LLMPermanentError` (fail fast, no retry). The OpenAI
  SDK's built-in retry is disabled so this project owns the policy.
- **Config/logic separation** — `src/config.py` is the only module that reads
  the environment; everything else receives plain constructor arguments.

## Testing

The suite never calls the real API: the OpenAI SDK is replaced by an injected
fake, and the provider by the deterministic mock — offline and reproducible.

```bash
pytest                      # or: python -m pytest
pytest tests/test_llm_client.py -v
```

Covered: schema validation, CSV validation, provider retry/backoff behaviour,
per-item failure isolation, report formats, and the CLI end-to-end (demo mode,
missing-key handling, exit codes).

## Security notes

- API keys are read **only** from environment variables (`OPENAI_API_KEY`);
  nothing is hardcoded, logged, or written to reports.
- `.env` is git-ignored; `.env.example` ships placeholders only.
- `output/` reports are git-ignored so generated customer data never lands in
  version control.
- The bundled sample data is fictional; send only data you are allowed to
  process to your configured LLM provider.
- Customer text is passed to the model as data (JSON user content) with a
  fixed system prompt, and prompt-injection attempts are explicitly tested.

## Project layout

```
ai-inquiry-processor/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example          # placeholders only
├── .gitignore
├── sample_inquiries.csv  # fictional sample data
├── src/
│   ├── config.py         # environment → AppConfig (no logic)
│   ├── models.py         # dataclasses + strict schema validation
│   ├── llm_client.py     # LLMClient protocol, OpenAI + mock providers, retry/backoff
│   ├── csv_loader.py     # input validation
│   ├── processor.py      # per-item error isolation
│   ├── report_writer.py  # JSON + CSV reports
│   └── main.py           # CLI wiring
├── tests/                # pytest, fully offline
└── output/               # generated reports (git-ignored, created on demand)
```

## License

[MIT](LICENSE)
