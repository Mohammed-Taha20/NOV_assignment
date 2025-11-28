# Retail Analytics Copilot - DSPy + LangGraph

A local, free AI agent that answers retail analytics questions using RAG over documents and SQL over the Northwind database.

## Graph Design

**7-Node Hybrid Architecture:**
1. **Router** - Classifies queries as `rag`, `sql`, or `hybrid` using DSPy ChainOfThought
2. **Retriever** - BM25-based document retrieval with chunk IDs for citations
3. **Planner** - Extracts constraints (dates, KPIs, categories) from retrieved docs
4. **NL2SQL** - Generates SQLite queries using DSPy with live schema
5. **Executor** - Safely executes SQL and captures results
6. **Synthesizer** - Produces typed answers matching `format_hint` with DSPy
7. **Repair** - Auto-fixes failed SQL queries (up to 2 attempts)

**Flow:**
- `sql` queries: Router → NL2SQL → Executor → Synthesizer
- `rag` queries: Router → Retriever → Synthesizer
- `hybrid` queries: Router → Retriever → Planner → NL2SQL → Executor → Synthesizer
- Failed SQL triggers: Executor → Repair → Executor (with loop prevention)

## DSPy Optimization

**Module:** NL2SQL  
**Optimizer:** BootstrapFewShot  
**Metric:** Valid SQL generation rate (presence of SELECT statement)

### Results:
| Module | Before | After | Improvement |
|--------|--------|-------|-------------|
| Router | 75% | 100% | +25% |
| NL2SQL | 50% | 100% | +50% |
| Synthesizer | 100% | 100% | 0% |

**Key Improvement:** NL2SQL optimization increased valid SQL generation from 50% to 100% on training examples by providing few-shot demonstrations of successful query patterns.

## Assumptions & Trade-offs

**Cost of Goods Approximation:**  
- As specified in the assignment, when `CostOfGoods` is unavailable, we use `0.7 * UnitPrice`
- This approximation is applied in the gross margin calculation per the KPI definitions

**SQL Safety:**
- Only SELECT queries allowed (security constraint)
- Queries timeout after 90 seconds to prevent hangs
- Maximum 2 repair attempts to prevent infinite loops

**Retrieval:**
- BM25 with 400-character chunks and 50-character overlap
- Top-k=3 chunks per query (balances context vs. noise)
- Citation format: `filename::chunkN` for traceability

**Model Constraints:**
- Phi-3.5-mini-instruct via Ollama (local, no API costs)
- Temperature 0.3 for consistent outputs
- Max 2000 tokens per generation

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama
ollama serve

# 3. Pull model
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M

# 4. Download database
mkdir -p data
curl -L -o data/northwind.sqlite \
  https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db

# 5. Create docs (see assignment PDF for content)
mkdir -p docs
# Add: marketing_calendar.md, kpi_definitions.md, catalog.md, product_policy.md

# 6. Run batch evaluation
python run_agent_hybrid.py \
  --batch sample_questions_hybrid_eval.jsonl \
  --out outputs_hybrid.jsonl
```

## Output Contract

Each answer in `outputs_hybrid.jsonl` includes:
```json
{
  "id": "question_id",
  "final_answer": "typed answer matching format_hint",
  "sql": "last executed SQL or empty string",
  "confidence": 0.75,
  "explanation": "Brief 1-2 sentence explanation",
  "citations": ["Orders", "Products", "marketing_calendar::chunk0"]
}
```

## Project Structure

```
.
├── agent/
│   ├── graph_hybrid.py         # LangGraph workflow
│   ├── dspy_signatures.py      # DSPy modules & optimizers
│   ├── rag/
│   │   └── retrieval.py        # BM25 retriever
│   └── tools/
│       └── sqlite_tool.py      # SQLite interface
├── data/
│   └── northwind.sqlite        # Database
├── docs/
│   ├── marketing_calendar.md   # Campaign dates
│   ├── kpi_definitions.md      # Metric formulas
│   ├── catalog.md              # Product categories
│   └── product_policy.md       # Return policies
├── run_agent_hybrid.py         # CLI entrypoint
├── sample_questions_hybrid_eval.jsonl
└── requirements.txt
```

## Evaluation Questions

The system handles 6 test cases covering:
- RAG-only: Policy lookups (return windows)
- SQL-only: Revenue calculations, top products
- Hybrid: Campaign-specific analytics combining doc constraints + SQL aggregation

All answers include proper citations, confidence scores, and match exact `format_hint` types.
