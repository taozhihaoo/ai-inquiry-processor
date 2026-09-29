# AI Inquiry Processor — AI Customer Support Desk

**A lightweight Python service that triages customer inquiries with LLM structured outputs, and a working support desk core on top: customers, tickets, inbox, tags, assignment, notes, history, search, AI analysis with suggested replies, RBAC and an audit log — CLI, REST API, SQLite, 280 offline tests.**

`Python 3.11+` · `FastAPI` · `OpenAI SDK (structured outputs)` · `SQLite` · `pytest (offline)` · `Docker`

> Independent portfolio project. OpenAI integration is fully implemented and tested against a fake SDK — default tests never call the network. Demo mode needs no credentials. Real OpenAI calls require your own key. The built-in authentication is a development/demo mechanism, not production identity management.

**AI-generated fields are suggestions / classifications and may require human review.** The system stores them as advisory data (`ai_*` fields); it never presents them as decided facts and never auto-sends generated replies.

---

## Project Overview

Phase 1 was a CSV → LLM → CSV triage tool. Phase 2 added a service layer, REST API, SQLite persistence and idempotency. Phase 3 evolves it into an **AI Customer Support Desk** core:

```
Customer ──▶ Ticket ──▶ AI Processing (summary / category / urgency / sentiment)
               │              ▲
               │              └── always via LLMClient → OpenAI | Mock
               ├── Tags · Assignment · Status workflow
               ├── Internal Notes · Conversation messages
               └── History events · Audit log
```

## Features

- **Inbox** — list/filter tickets by status (`open/pending/resolved/closed`), priority (`low/normal/high/urgent`), assignee (incl. unassigned), customer and tag; full-text search across subject, customer name/email and message bodies.
- **Tickets with a real workflow** — explicit, enforced status transitions (invalid transitions get `409`), reopen supported, resolution timestamps managed automatically.
- **Customers** — create/find/search; duplicate emails resolve to the same customer; per-customer ticket lists.
- **Tags** — admin-managed vocabulary, linked to tickets (idempotent link/unlink).
- **Agents & assignment** — `agent` and `admin` roles; assign/unassign with permission rules; per-agent queues.
- **Internal notes** — agent-only remarks with authorship, distinct from the customer conversation.
- **Ticket history** — every status/priority/assignment/tag/note/AI change recorded as an event with actor + JSON metadata.
- **AI analysis** — summary, key points, category, suggested priority, urgency, sentiment per ticket (advisory only). Provider-agnostic via the `LLMClient` abstraction with strict `json_schema` outputs and response re-validation.
- **AI suggested reply** — a draft generated from the latest customer message; stored on the ticket, clearly labelled as a draft, never auto-sent.
- **Duplicate detection** — content-hash (SHA-256 over customer + normalized message); re-submitting the same open ticket's content returns `409` with the existing ticket id. Closed tickets can be re-submitted fresh. (The Phase 2 inquiry pipeline keeps its own idempotency: identical inquiries return the stored result, flagged `duplicate`.)
- **RBAC + audit log** — minimal role checks in the service layer; every important action lands in a queryable audit log (admin-only read).
- **Resilience (Phase 2 heritage)** — timeout, retry with exponential backoff, `Retry-After` honored, retryable-vs-permanent error taxonomy, per-item failure isolation, usage/latency tracking (null when the provider reports nothing — never fabricated).

## Architecture

```
            ┌─────────────┐   ┌──────────────────────────────────┐
            │  CLI        │   │  FastAPI (src/api.py)            │
            │ (src.main)  │   │  inquiry endpoints + desk router │
            └──────┬──────┘   └───────┬──────────────────┬───────┘
                   │                  │                  │
                   ▼                  ▼                  ▼
        ┌────────────────────┐  ┌──────────────────────────────────┐
        │ InquiryService     │  │ DeskService (src/desk/service.py)│
        │ (src/service.py)   │  │ workflow · RBAC · dedup · audit  │
        └─────────┬──────────┘  └───────┬──────────────┬───────────┘
                  ▼                     ▼              ▼
        ┌────────────────────┐  ┌─────────────┐  ┌───────────────┐
        │ InquiryProcessor   │  │ desk.ai     │  │ DeskStore     │
        └─────────┬──────────┘  └──────┬──────┘  └──────┬────────┘
                  ▼                    ▼                ▼
        ┌──────────────────────────────────────────────────────────┐
        │ LLMClient protocol → OpenAIClient (strict json_schema,   │
        │ retry/backoff) | MockLLMClient (deterministic, offline)  │
        └──────────────────────────────────────────────────────────┘
                  ▼
        SQLite (one file): inquiries + processing_runs (Phase 2)
                         + customers/agents/tags/tickets/messages/
                           notes/ticket_tags/ticket_events/audit_log (Phase 3)
```

Layering rules: the API never writes SQL and never calls a provider; the service never imports FastAPI; all AI goes through `LLMClient`.

## Data Model

Phase 2 tables (unchanged): `inquiries` (content-hash id, status, analysis, usage), `processing_runs`.

Phase 3 tables (added by idempotent migration; old databases keep working):

| Table | Key fields |
| --- | --- |
| `customers` | `id`, `name`, `email` (unique), `external_id?`, timestamps |
| `agents` | `id`, `name`, `email` (unique), `role` (`agent`/`admin`), `active`, timestamps |
| `tags` | `id`, `name` (unique, case-insensitive) |
| `tickets` | `id`, `customer_id`→customers, `subject`, `status`, `priority`, `assignee_id`→agents?, `content_hash`, `resolved_at?`, `ai_summary`, `ai_key_points` (JSON), `ai_category`, `ai_suggested_priority`, `ai_urgency`, `ai_sentiment`, `ai_reply`, `ai_analyzed_at`, `ai_provider/model/tokens`, timestamps |
| `ticket_messages` | `id`, `ticket_id`→tickets, `author_type` (`customer`/`agent`/`system`), `author_name`, `body`, `content_hash` |
| `internal_notes` | `id`, `ticket_id`, `author_id`, `author_name`, `body` |
| `ticket_tags` | (`ticket_id`, `tag_id`) composite PK |
| `ticket_events` | `id`, `ticket_id`, `event_type`, `actor_id?`, metadata JSON |
| `audit_log` | `id`, `actor_id?`, `action`, `entity_type`, `entity_id`, metadata JSON |

Vocabularies: status `open/pending/resolved/closed` · priority & urgency `low/normal/high/urgent` · sentiment `positive/neutral/negative` · AI category reuses the Phase 2 set `Sales / Technical Support / Billing / General Question`.

Indexes on ticket status, priority, assignee, customer, updated_at, content_hash. All SQL is parameterized; foreign keys enforced.

## Running Locally

```bash
git clone https://github.com/taozhihaoo/ai-inquiry-processor.git
cd ai-inquiry-processor
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # PowerShell; source .venv/bin/activate on bash
pip install -r requirements.txt

python -m src.main --demo                          # CLI triage demo (offline)
uvicorn src.api:app --host 127.0.0.1 --port 8000   # API with desk (mock provider)
# Swagger UI: http://127.0.0.1:8000/docs
```

## Demo / Seed Data

Fictional demo dataset (3 agents, 4 customers, 6 tags, 8 tickets across all statuses/priorities, 3 AI-analyzed, 1 AI reply draft — one seeded ticket deliberately demonstrates duplicate detection):

```bash
python -m src.desk.seed                             # seed data/inquiries.db (skips if populated)
python -m src.desk.seed --db path/to/db.sqlite      # explicit target
python -m src.desk.seed --reset                     # wipe desk tables, reseed (inquiries kept)
```

To explore the API as a seeded admin, pass `X-Agent-Id: <admin id>` — get ids with:

```bash
python -c "from src.desk.storage import DeskStore; [print(a.id, a.role, a.name) for a in DeskStore('data/inquiries.db').list_agents()]"
```

## REST API

Interactive docs at `/docs` (Swagger, Try-it-out). Existing Phase 2 endpoints are unchanged: `GET /health`, `POST /inquiries`, `POST /inquiries/batch`, `GET /inquiries/{id}`.

Desk endpoints (all require the `X-Agent-Id` dev-auth header):

```
POST   /customers                      create (200 existing / 201 new)
GET    /customers?search=              list/search
GET    /customers/{id}                 fetch
GET    /customers/{id}/tickets         customer's tickets

GET    /tickets                        inbox: ?status= ?priority= ?assignee=|none
                                       ?customer_id= ?tag= ?search= ?limit= ?offset=
POST   /tickets                        create (409 on duplicate content)
GET    /tickets/{id}                   detail (ticket + customer + assignee + tags + messages)
PATCH  /tickets/{id}                   status/priority/subject (409 on invalid transition)
POST   /tickets/{id}/assign            { "agent_id": "..." }
POST   /tickets/{id}/unassign
POST   /tickets/{id}/messages          record an inbound customer message
POST   /tickets/{id}/notes             internal note      GET …/notes
POST   /tickets/{id}/tags              { "tag_id": "..." }   DELETE …/tags/{tag_id}
GET    /tickets/{id}/history           ticket events

POST   /tickets/{id}/ai/analyze        summary/key points/category/urgency/sentiment
POST   /tickets/{id}/ai/suggest-reply  draft reply (never auto-sent)

GET    /tags                           POST /tags                (admin)
GET    /agents                         POST /agents              (admin)
GET    /agents/{id}/tickets
GET    /audit                          admin-only, newest first
```

Status codes in use: `200/201/204` success · `400` validation · `401` missing/unknown/inactive agent · `403` insufficient role · `404` unknown entity · `409` duplicate ticket / invalid transition · `422` malformed body · `502` provider failure. Errors are always `{"error": {"code", "message", ...}}` — no tracebacks, no internals.

## AI Capabilities

| Operation | Endpoint/service path | Output (advisory) |
| --- | --- | --- |
| Ticket analysis | `POST /tickets/{id}/ai/analyze` → `DeskService.ai_analyze_ticket` → `desk.ai` → `LLMClient.complete_structured` | summary, 1–5 key points, category, suggested priority, urgency, sentiment |
| Suggested reply | `POST /tickets/{id}/ai/suggest-reply` | draft text generated from the latest customer message; stored as `ai_reply`, labelled as a draft |
| Inquiry triage | Phase 2 pipeline unchanged | summary, category, priority |

All calls pin a strict `json_schema` response format **and** are re-validated against the vocabularies before persistence. The mock provider implements the same schemas deterministically (keyword heuristics), so demos, tests and CI run fully offline; usage fields stay `null` for the mock — no fabricated tokens.

## Roles / Permissions

| Capability | agent | admin |
| --- | --- | --- |
| View tickets/customers/tags/agents, search | ✅ | ✅ |
| Modify ticket (status/priority/subject/assign/tags) | unassigned or own tickets | any ticket |
| Add internal notes / customer messages, use AI | ✅ | ✅ |
| Create customers | ✅ | ✅ |
| Create agents, create tags, read audit log | ❌ | ✅ |

**Development authentication (demo only):** send `X-Agent-Id: <agent id>`. Unknown, missing or deactivated ids get `401`; insufficient role gets `403`. This exists so the RBAC layer is real and testable without building an identity provider. **It is not production authentication** — no OAuth, no JWT, no SSO, no secrets. Put the service behind your own gateway for anything real.

## Audit Log

`GET /audit` (admin) returns entries for ticket created/status/priority changes, assignment changes, tag changes, notes, messages, AI actions, customer and agent creation. Entries carry actor id, action, entity, and whitelisted metadata only — no prompts, no credentials, no authorization headers.

## Search

`GET /tickets?search=<term>` matches ticket subject, customer name, customer email and message bodies via parameterized SQLite `LIKE` with user wildcards escaped. No Elasticsearch, no external services.

## Testing

```bash
pytest        # 280 tests, fully offline: fake SDK, mock LLM, temp SQLite
ruff check .  # lint (CI enforces)
```

Coverage highlights: workflow transitions (valid/invalid/no-op), RBAC (401/403 paths), duplicate detection, assignment permissions, notes/tags/history/audit, AI analyze + suggest reply (advisory-only asserted), migration from a Phase 2 database (old data preserved, new tables added, app boots), seed idempotency, full HTTP contract (`200/201/204/400/401/403/404/409/422`), plus all 139 Phase 2 tests kept green. CI also runs the lint, the migration tests, and executes the seed CLI twice on a fresh database.

## OpenAI API Usage

Implemented but **not executed against the live API** in the environment where this README was last verified (no `OPENAI_API_KEY` present — stated deliberately). To run for real:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key        # https://platform.openai.com/api-keys
```

Set it for the CLI or `docker compose up` (see below). Token usage is stored when the provider reports it and stays `null` otherwise; `estimated_cost` (inquiry pipeline) requires explicitly configured prices and is an **estimate**, never a billing figure.

## Configuration

See [`.env.example`](.env.example). Key variables: `LLM_PROVIDER` (`openai`/`mock`), `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`, `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, `LLM_RETRY_INITIAL_DELAY`, `LLM_RETRY_MAX_DELAY`, `AI_INQUIRY_DB_PATH`, `AI_INQUIRY_OUTPUT_DIR`, optional `LLM_PRICE_*_PER_MTOK`. Defaults: mock provider for bare `uvicorn` and Docker; `openai` for a bare CLI run.

## Running with Docker

```bash
docker compose up          # mock provider; Swagger at http://127.0.0.1:8000/docs
curl http://127.0.0.1:8000/health
python -m src.desk.seed --db <volume path>/data/inquiries.db   # optional demo data
```

Non-root container, named volumes for `/app/data` and `/app/output`, healthcheck on `/health`. `.env` is never baked into the image; inject secrets at runtime:

```bash
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... docker compose up
```

*Docker config is provided but has not been executed in this environment (no Docker daemon available at verification time).*

## Security Notes

- API keys only from environment variables; never hardcoded, logged, or returned.
- All SQL parameterized; foreign keys on; user search input wildcard-escaped.
- Customer text travels as JSON data with a fixed system prompt; a test asserts prompt-injection attempts cannot alter instructions.
- AI payloads re-validated against fixed vocabularies before persistence.
- Error responses are structured and opaque (`{"error": {...}}`) — no tracebacks, no provider internals.
- Audit log stores whitelisted metadata only.
- `data/`, `output/`, `.env` are git-ignored; sample data is fictional.
- Development authentication is explicitly not production auth (see above).

## Project Structure

```
src/
├── config.py · models.py · csv_loader.py · report_writer.py     # shared core
├── llm_client.py        # LLMClient protocol, OpenAI + Mock, retry policy,
│                        # complete_structured() building block for all AI
├── processor.py · service.py · storage.py · main.py             # inquiry pipeline
├── api.py               # FastAPI app: inquiry endpoints + mounted desk router
└── desk/
    ├── models.py        # dataclasses + vocabularies + validation
    ├── workflow.py      # allowed status transitions
    ├── storage.py       # DeskStore: customers/tickets/tags/agents/notes/events/audit
    ├── ai.py            # ticket analysis + suggested reply prompts/schemas/validation
    ├── auth.py          # dev authentication + AgentContext + role checks
    ├── service.py       # DeskService: RBAC, workflow, dedup, history, audit
    ├── api.py           # desk REST router (thin transport)
    └── seed.py          # fictional demo dataset (python -m src.desk.seed)
tests/                   # 280 offline tests (12 Phase 2 files + 6 desk files)
Dockerfile · docker-compose.yml · .github/workflows/ci.yml
```

## Current Limitations

Stated plainly:

- **Development authentication only** — `X-Agent-Id` header, no real identity provider.
- **SQLite, single process** — right for this scale; concurrent multi-instance writes need a real database server.
- **Synchronous AI calls** — no background workers/queues; batch endpoints are bounded (API: 1000 items).
- **Duplicate detection is hash-based** — same customer + normalized text only; no semantic similarity (intentional).
- **No frontend** — Swagger is the UI; seed data makes it demoable.
- **OpenAI path unverified against the live API** in this environment; Docker config not executed here.
- Not "production-ready" or "battle-tested" — a well-tested, honestly-scoped portfolio project.

## Future Extensions

Planned (not implemented — do not assume otherwise):

- **Email ingestion** — inbound email → ticket (the message layer and content-hash dedup are the natural insertion point).
- **Webhook integration** — outbound events on ticket changes (ticket_events is the feed).
- Real authentication (OIDC/JWT), background AI workers, per-customer portals.

## License

[MIT](LICENSE)
