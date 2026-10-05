# AI Engineer Template GENERALS

Reusable scaffold for AI Engineer and Agentic AI projects.

## Stack

- Python
- LangChain
- LangGraph
- OpenAI
- PostgreSQL
- RAG
- Docker
- Git

## Project structure GENERALS

The project is organized into separate layers for configuration, models, tools, agents, RAG, database access and tests.

## Environment GENERALS

Create and activate the virtual environment before installing dependencies.

```cmd
.venv\Scripts\activate
```

## Secrets GENERALS

API keys and other secrets belong in `.env` and must never be committed to Git.


# AI Engineer Template (full)

Reusable foundation for building AI-powered applications with **Python, LangChain, LangGraph, FastAPI, RAG, PostgreSQL, FAISS, Docker and GitHub Actions**.

The repository is designed as a reusable starting point for AI Engineer projects rather than as a single domain-specific application.

The first real-world validation was performed using **Sorbara e Dintorni** as the domain example.

---

## Architecture

```text
                         ┌──────────────┐
                         │   FastAPI    │
                         └──────┬───────┘
                                │
                       Application Layer
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
               Agent                           RAG
                 │                             │
          ┌──────┴──────┐              ┌───────┴──────┐
          │             │              │              │
        Tools          LLM        Embeddings        FAISS
          │                            │
          ▼                            ▼
     PostgreSQL                    Documents
          │
          ▼
     Persistence

Infrastructure:
Docker + Docker Compose + Git + GitHub Actions

Quality:
Pytest + CI
```

---

## Stack

### AI

* Python 3.13
* LangChain
* LangGraph
* OpenAI
* LangChain OpenAI
* RAG
* OpenAI embeddings
* FAISS

### Backend

* FastAPI
* Uvicorn

### Database

* PostgreSQL
* SQLAlchemy
* Psycopg

### Infrastructure

* Docker
* Docker Compose
* GitHub Actions

### Testing

* Pytest
* FastAPI TestClient

---

## Project structure

```text
ai-engineer-template/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── raw/
│   │   └── example.txt
│   └── processed/
│       └── faiss/
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   │
│   ├── agents/
│   │   └── portal_agent.py
│   │
│   ├── config/
│   │   └── settings.py
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   ├── init_db.py
│   │   ├── models.py
│   │   └── repository.py
│   │
│   ├── llm/
│   │   ├── models.py
│   │   └── tool_model.py
│   │
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── ingestion.py
│   │   ├── main.py
│   │   ├── rag.py
│   │   └── vector_store.py
│   │
│   ├── tools/
│   │   └── portal.py
│   │
│   └── main.py
│
├── tests/
│   ├── test_api.py
│   ├── test_database.py
│   ├── test_portal_agent.py
│   └── test_portal_tool.py
│
├── .dockerignore
├── .env
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Core components

### FastAPI

`src/api/main.py`

Provides the HTTP interface.

Current endpoints:

```text
GET  /health
POST /ask
```

Health check:

```json
{
  "status": "ok"
}
```

Example request:

```json
{
  "question": "Qual è lo stato del portale?"
}
```

---

### LangGraph Agent

`src/agents/portal_agent.py`

Implements a basic tool-calling agent using LangGraph.

Flow:

```text
START
  ↓
agent
  ↓
tool required?
 ├── no → END
 └── yes
       ↓
      tool
       ↓
      agent
```

The current `portal_agent` is a domain example. For a new project it can be replaced with agents such as:

* customer support agent
* research agent
* document agent
* sales agent
* operations agent

---

### Tools

`src/tools/`

Contains functions exposed to the agent through LangChain's `@tool`.

Current example:

```text
get_portal_status()
```

In a real project, tools can interact with:

* REST APIs
* databases
* files
* external services
* internal business systems
* Python functions

---

### LLM layer

`src/llm/`

Centralizes LLM configuration.

`models.py` creates the ChatOpenAI model.

`tool_model.py` binds the available tools to the model.

The model is configured through:

```env
OPENAI_MODEL=gpt-5-mini
```

---

## RAG

The RAG implementation lives in:

```text
src/rag/
```

Pipeline:

```text
Documents
    ↓
Ingestion
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS
    ↓
Similarity Search
    ↓
Context
    ↓
LLM
    ↓
Answer
```

### Document ingestion

`src/rag/ingestion.py`

Currently loads `.txt` files from:

```text
data/raw/
```

Documents are split using `RecursiveCharacterTextSplitter`.

Current initial parameters:

```text
chunk_size = 500
chunk_overlap = 50
```

### Embeddings

`src/rag/embeddings.py`

Uses:

```text
text-embedding-3-small
```

### Vector store

`src/rag/vector_store.py`

Uses FAISS and persists the index under:

```text
data/processed/faiss/
```

The vector store is loaded if it already exists, otherwise it is created.

### RAG answer generation

`src/rag/rag.py`

Retrieves the most relevant chunks and provides them as context to the LLM.

The current implementation instructs the model to answer only from the retrieved context and explicitly state when the context is insufficient.

---

## PostgreSQL

Database code lives in:

```text
src/database/
```

### Current model

```text
documents
├── id
├── source
└── content
```

### Initialize database

```powershell
python -m src.database.init_db
```

### Repository

`repository.py` currently provides:

```text
create_document()
get_document()
```

The repository layer can later be extended with additional CRUD and domain-specific operations.

---

## Environment variables

Create `.env` in the project root.

Example:

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5-mini
DATABASE_URL=postgresql+psycopg://agent:agentpass@localhost:5433/ai_engineer
```

Never commit `.env`.

Use `.env.example` as the configuration template.

---

## Local development

### 1. Create/activate virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Configure `.env`

Set:

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5-mini
DATABASE_URL=postgresql+psycopg://agent:agentpass@localhost:5433/ai_engineer
```

### 4. Start PostgreSQL

```powershell
docker compose up -d postgres
```

### 5. Initialize database

```powershell
python -m src.database.init_db
```

### 6. Start FastAPI

```powershell
uvicorn src.api.main:app --reload
```

API:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

---

## API examples

### Health

```powershell
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

### Ask

Windows CMD example:

```powershell
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"Qual è lo stato del portale?\"}"
```

---

## Testing

Run all tests:

```powershell
python -m pytest
```

Current test coverage includes:

```text
test_api.py
test_database.py
test_portal_agent.py
test_portal_tool.py
```

Current verified result:

```text
4 passed
```

---

## Docker

Build:

```powershell
docker compose build
```

Start:

```powershell
docker compose up -d
```

Check services:

```powershell
docker compose ps
```

Logs:

```powershell
docker compose logs -f app
```

Stop:

```powershell
docker compose down
```

Remove containers and database volume:

```powershell
docker compose down -v
```

> `docker compose down -v` deletes the PostgreSQL volume and therefore the stored database data.

### Docker database networking

From Windows:

```text
localhost:5433
```

From the application container:

```text
postgres:5432
```

`postgres` is the Docker Compose service name and is only resolvable inside the Docker network.

---

## Running tests inside Docker

```powershell
docker compose run --rm app python -m pytest
```

Initialize the database from the container:

```powershell
docker compose run --rm app python -m src.database.init_db
```

---

## CI

GitHub Actions workflow:

```text
.github/workflows/ci.yml
```

Runs on:

* pushes to `master`
* pull requests targeting `master`

Pipeline:

```text
Checkout
   ↓
Python 3.13
   ↓
Install dependencies
   ↓
PostgreSQL service
   ↓
Pytest
   ↓
Docker build
```

This provides a basic CI foundation for future production deployment.

---

## Git workflow

Check status:

```powershell
git status
```

View recent commits:

```powershell
git log --oneline -5
```

Stage:

```powershell
git add .
```

Commit:

```powershell
git commit -m "Your message"
```

Push:

```powershell
git push
```

---

# Reusing the template

The repository is designed to separate **generic infrastructure** from **domain-specific logic**.

Keep:

```text
FastAPI
LangChain
LangGraph
LLM layer
RAG layer
Database layer
Docker
Tests
CI
Git structure
```

Replace or extend:

```text
agents
tools
database models
RAG documents
API routes
domain logic
```

For example, a customer-support project could become:

```text
src/
├── agents/
│   └── support_agent.py
│
├── tools/
│   ├── customers.py
│   ├── orders.py
│   └── tickets.py
│
├── rag/
│   └── ...
│
└── api/
    └── main.py
```

The underlying infrastructure remains reusable.

---

## Agent + RAG architecture

A more advanced project can combine both approaches:

```text
                         FastAPI
                            │
                            ▼
                          Agent
                            │
               ┌────────────┼────────────┐
               ▼            ▼            ▼
              RAG          Tools       Memory
               │            │
               ▼            ▼
             FAISS      PostgreSQL
               │            │
               └──────┬─────┘
                      ▼
                     LLM
```

The agent can decide whether a request requires:

* retrieval from the knowledge base
* a tool/API call
* database access
* a direct answer

---

## Current limitations

This template is a solid foundation, not a complete enterprise production stack.

Not yet implemented as production features:

* authentication
* authorization
* rate limiting
* streaming
* advanced observability
* LangSmith tracing configuration
* RAG reranking
* hybrid search
* metadata filtering
* RAG evaluation
* advanced guardrails
* persistent agent memory
* Redis
* background workers
* Kubernetes
* cloud deployment
* secrets management
* Alembic migrations
* production reverse proxy
* structured production logging
* metrics/monitoring

These are planned extensions rather than requirements for the current template.

---

## Roadmap

### V1 — Foundation

* [x] Python environment
* [x] LangChain
* [x] LangGraph
* [x] OpenAI
* [x] Tool calling
* [x] Agent
* [x] RAG
* [x] Embeddings
* [x] FAISS
* [x] PostgreSQL
* [x] SQLAlchemy
* [x] Repository layer
* [x] FastAPI
* [x] Docker
* [x] Docker Compose
* [x] Pytest
* [x] GitHub
* [x] GitHub Actions
* [x] Docker build in CI

### V2 — Advanced RAG

* [ ] Better document loaders
* [ ] Metadata
* [ ] Metadata filtering
* [ ] Hybrid retrieval
* [ ] Reranking
* [ ] Citations
* [ ] RAG evaluation
* [ ] Retrieval evaluation
* [ ] Guardrails

### V3 — Cloud / AI platform

* [ ] Azure
* [ ] Microsoft Foundry
* [ ] Cloud deployment
* [ ] Observability
* [ ] Tracing
* [ ] Production secrets management

### V4 — Production infrastructure

* [ ] Advanced Docker
* [ ] Full CI/CD
* [ ] Kubernetes
* [ ] Scaling
* [ ] Production monitoring

---

## First real-world validation

The architecture was initially validated using **Sorbara e Dintorni** as the domain example.

The domain-specific implementation is intentionally kept separate from the reusable architecture.

Conceptually:

```text
Template
    =
Infrastructure + AI Architecture

Sorbara e Dintorni
    =
Domain / First Real Implementation
```

This allows the same foundation to be reused for completely different AI systems.

---

## Current status

**AI Engineer Template v1.0**

Verified components:

```text
[✓] Python
[✓] LangChain
[✓] LangGraph
[✓] OpenAI
[✓] Tool calling
[✓] Agent
[✓] RAG
[✓] Embeddings
[✓] FAISS
[✓] PostgreSQL
[✓] SQLAlchemy
[✓] FastAPI
[✓] Docker
[✓] Docker Compose
[✓] Pytest
[✓] Git
[✓] GitHub
[✓] GitHub Actions
[✓] Docker build in CI
```

Verified API:

```text
GET  /health
POST /ask
```

Verified local test suite:

```text
4 passed
```

---

## Core principle

The purpose of this repository is not to provide one finished AI application.

Its purpose is to provide a **repeatable engineering foundation**:

```text
API
 ↓
Application Logic
 ↓
Agent / RAG
 ↓
LLM
 ↓
Tools / Vector Store / Database
 ↓
Docker / CI / Testing
```

New projects should start from this foundation and replace the domain logic rather than rebuilding the infrastructure from scratch.
