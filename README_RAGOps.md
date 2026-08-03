# RAGOps: Automated RAG Evaluation & Continuous CI/CD

An enterprise-style Retrieval-Augmented Generation (RAG) project that
automatically builds a synthetic evaluation dataset, benchmarks
retrieval and answer quality, blocks risky changes in CI, and monitors
production feedback and traces.

## 1. Project goals

-   Generate Question--Context--Answer evaluation cases from a document
    corpus.
-   Measure retrieval quality, groundedness, and answer relevance.
-   Run evaluations whenever prompts, embedding models, chunking,
    retrievers, or generation models change.
-   Fail CI when agreed quality gates are violated.
-   Collect thumbs-up/down feedback and traces for offline analysis.

> **Important:** Treat the thresholds below as configurable policy, not
> universal truths. A "hallucination rate below 2%" is meaningful only
> when the metric, test set, sample size, and adjudication process are
> clearly defined. Start with a baseline and calibrate thresholds on a
> representative dataset.

## 2. Recommended architecture

``` mermaid
flowchart TD
    A[Source documents: PDF, DOCX, HTML, TXT] --> B[Ingestion and parsing]
    B --> C[Cleaning, metadata, deduplication]
    C --> D[Chunking]
    D --> E[Embedding model]
    E --> F[(Vector database)]
    C --> G[Corpus snapshot and document IDs]

    G --> H[Synthetic test-set generator]
    H --> I[Question / reference answer / evidence context]
    I --> J[Human review, validation, versioned eval dataset]

    U[User question] --> K[API: FastAPI]
    K --> L[Retriever: top-k + optional reranker]
    F --> L
    L --> M[Prompt assembly with retrieved evidence]
    M --> N[LLM answer generation]
    N --> O[Answer + source citations]
    O --> P[User feedback: thumbs up/down]

    J --> Q[Evaluation runner]
    L --> Q
    N --> Q
    Q --> R[RAGAS / DeepEval metrics]
    R --> S[Evaluation report + trend history]
    S --> T{Quality gates pass?}
    T -->|Yes| V[CI allows merge/deployment]
    T -->|No| W[Fail CI and attach report]

    K --> X[Trace instrumentation]
    L --> X
    N --> X
    P --> X
    X --> Y[Langfuse or Arize Phoenix]
    Y --> Z[Review failures, improve data/prompts/retrieval]
    Z --> B
```

### Runtime request flow

1.  A user sends a question to the FastAPI endpoint.
2.  The retriever fetches relevant chunks from the vector database;
    optionally, a reranker reorders them.
3.  The prompt includes the question, retrieved evidence, and
    instructions to acknowledge missing information.
4.  The LLM generates an answer with source references.
5.  The API returns the answer and citations, while tracing captures
    retrieval results, prompt/model configuration, latency, and token
    usage.
6.  The UI records optional thumbs-up/down feedback tied to a
    trace/request ID.

### Evaluation and CI flow

1.  A pull request changes a prompt, chunking strategy, embedding model,
    retriever, reranker, or LLM configuration.
2.  CI builds or loads a pinned evaluation dataset and runs the same
    benchmark against the candidate pipeline.
3.  Metrics are compared with the baseline and configured absolute
    thresholds.
4.  CI uploads a machine-readable report and fails if a required gate
    fails.
5.  A passing build can proceed to staging/deployment; production
    metrics continue to be monitored.

## 3. Suggested tech stack

Keep the first version small. Pick **one** tool in each overlapping
category instead of integrating everything at once.

  -----------------------------------------------------------------------
  Layer                   Recommended choice      Alternatives / notes
  ----------------------- ----------------------- -----------------------
  Language                Python 3.11+            Pin exact versions in
                                                  the lockfile

  API                     FastAPI + Pydantic      Streamlit can be added
                                                  later for a demo UI

  Document parsing        PyMuPDF, python-docx,   Add OCR only if scanned
                          BeautifulSoup           PDFs are in scope

  Chunking                Custom chunker or       Log chunk size and
                          LlamaIndex/LangChain    overlap as experiment
                          text splitters          parameters

  Embeddings              Sentence Transformers   Version the model and
                          or a hosted embedding   preprocessing
                          API                     

  Vector store            Qdrant                  Chroma for a local
                                                  prototype; pgvector for
                                                  a PostgreSQL-centered
                                                  stack

  Generator LLM           A hosted API or a       Keep provider/model
                          locally served model    configurable

  RAG orchestration       Lightweight Python      Avoid unnecessary
                          modules, or             abstraction in the
                          LangChain/LlamaIndex    first version

  Evaluation              **RAGAS** as the        DeepEval is an
                          primary framework       alternative or
                                                  secondary cross-check

  Tracing / observability **Langfuse**            Arize Phoenix is a
                                                  strong alternative;
                                                  avoid duplicating
                                                  tracing systems
                                                  initially

  Test runner             pytest                  Store deterministic
                                                  unit/integration tests
                                                  separately from LLM
                                                  evaluations

  CI/CD                   GitHub Actions          Run evaluations on pull
                                                  requests and scheduled
                                                  jobs

  Packaging               Docker + Docker Compose Kubernetes only when
                                                  deployment
                                                  scale/operations
                                                  justify it

  Storage                 Versioned JSONL/Parquet Store dataset version,
                          plus artifact storage   corpus hash, model IDs,
                                                  and run metadata
  -----------------------------------------------------------------------

### Recommended MVP stack

**Python + FastAPI + Qdrant + Sentence Transformers + one LLM provider +
RAGAS + Langfuse + pytest + Docker Compose + GitHub Actions.**

Use DeepEval or Phoenix later if you need a second evaluator or
different tracing workflows. Do not start with both RAGAS and DeepEval
plus both Phoenix and Langfuse unless you have a clear comparison goal.

## 4. Repository structure

``` text
ragops/
├── app/
│   ├── api.py                 # FastAPI routes
│   ├── config.py              # Environment-based settings
│   ├── ingestion/
│   │   ├── loaders.py
│   │   ├── cleaning.py
│   │   └── chunking.py
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── generation/
│   │   ├── prompts.py
│   │   └── answer.py
│   ├── evaluation/
│   │   ├── generate_dataset.py
│   │   ├── run_eval.py
│   │   ├── metrics.py
│   │   └── quality_gates.py
│   ├── observability/
│   │   └── tracing.py
│   └── feedback/
│       └── routes.py
├── data/
│   ├── raw/                   # Do not commit private documents
│   └── eval/                  # Sanitized/versioned test fixtures
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_api.py
├── reports/                   # Generated locally; usually ignored by Git
├── scripts/
│   └── build_index.py
├── .github/
│   └── workflows/
│       └── rag-evaluation.yml
├── .env.example
├── .gitignore
├── Dockerfile
├── compose.yaml
├── pyproject.toml
└── README.md
```

## 5. Evaluation dataset design

Generate synthetic cases from **the exact corpus snapshot being
evaluated**, but keep evaluation questions and reference answers
separate from runtime prompts. A useful JSONL record can look like:

``` json
{
  "id": "policy-0001",
  "question": "How long does an employee have to submit the form?",
  "reference_answer": "The form must be submitted within 10 business days.",
  "reference_contexts": [
    "Employees must submit the completed form within 10 business days."
  ],
  "source_document_id": "hr-policy-v3",
  "source_span": "section-4-paragraph-2",
  "difficulty": "single-hop",
  "split": "test"
}
```

Recommended generation process:

1.  Parse and clean source documents; preserve document IDs, headings,
    and source spans.
2.  Generate questions of different types: fact lookup, multi-hop,
    comparison, unanswerable questions, and questions with similar
    distractor passages.
3.  Save reference evidence and source spans, not just an LLM-written
    answer.
4.  Validate that every question is answerable from its cited source
    (unless deliberately marked unanswerable).
5.  Deduplicate semantically similar questions and remove malformed
    cases.
6.  Sample and manually review a subset; track reviewer corrections.
7.  Split by document or source section---not just random question
    rows---to reduce leakage.
8.  Version the corpus snapshot, dataset, generation model, and
    generation prompt.

Do not blindly generate thousands of cases and treat them as ground
truth. Synthetic datasets can contain incorrect answers, weak questions,
duplicated examples, or evidence leakage. Quality review and an explicit
unanswerable set are important.

## 6. Metrics and quality gates

  -----------------------------------------------------------------------
  Metric                  What it helps measure   Practical
                                                  interpretation
  ----------------------- ----------------------- -----------------------
  Context Precision       Whether retrieved       Higher is generally
                          chunks are relevant,    better
                          and how much irrelevant 
                          context is present      

  Context Recall          Whether the retrieved   Higher is generally
                          context contains the    better
                          evidence needed for the 
                          reference answer        

  Faithfulness            Whether answer claims   Higher is generally
                          are supported by        better; a low score can
                          retrieved context       indicate unsupported
                                                  claims

  Answer Relevancy        Whether the response    Higher is generally
                          addresses the user's    better
                          question                

  Retrieval Recall@k /    Ranking and retrieval   Useful for isolating
  MRR / nDCG              quality using labeled   retriever regressions
                          relevant documents      

  Latency / token cost    Operational performance Track p50/p95 latency
                                                  and cost per request
  -----------------------------------------------------------------------

### Example initial policy

-   Context Recall: **at least 0.90** on a representative, reviewed
    benchmark.
-   Faithfulness: establish a baseline, then set a calibrated minimum.
-   Hallucination rate: **at most 0.02** only if using a clearly defined
    claim-level or case-level protocol.
-   Answer Relevancy and Context Precision: compare against baseline and
    set minimums after observing score distributions.
-   Operational checks: API/integration tests pass, and latency/cost do
    not exceed agreed budgets.

**Avoid equating `1 - faithfulness` with a rigorously measured
hallucination rate unless your chosen evaluator and definition
explicitly support that interpretation.** LLM-based judges are
imperfect. Use repeated runs or confidence intervals for noisy metrics,
and review failures before enforcing strict gates.

## 7. CI quality-gate logic

The following is illustrative pseudocode; adapt metric keys to the
actual RAGAS/DeepEval result schema used by your pinned version:

``` python
def quality_gate(metrics: dict) -> None:
    failures = []

    if metrics["context_recall"] < 0.90:
        failures.append("Context Recall is below 0.90")

    if metrics["hallucination_rate"] > 0.02:
        failures.append("Hallucination rate is above 0.02")

    if failures:
        raise SystemExit("RAG evaluation failed:\n- " + "\n- ".join(failures))
```

Before enforcing a hard gate, define whether metrics are averaged per
question, aggregated per claim, or calculated another way. Include
minimum sample sizes, missing-score handling, and an agreed rule for
flaky judge calls. Use a small smoke-test set on every PR and a larger
benchmark on scheduled runs or before release.

### GitHub Actions workflow outline

``` yaml
name: RAG Evaluation

on:
  pull_request:
  workflow_dispatch:
  schedule:
    - cron: "0 3 * * 1"

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install project
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Unit and integration tests
        run: pytest
      - name: Run RAG benchmark
        env:
          LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
          # Add evaluator/provider secrets as required.
        run: python -m app.evaluation.run_eval --dataset data/eval/test.jsonl --output reports/eval.json
      - name: Enforce quality gates
        run: python -m app.evaluation.quality_gates --report reports/eval.json
      - name: Upload evaluation report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: rag-evaluation-report
          path: reports/
```

This workflow is a template, not a drop-in workflow: the package extras,
command-line arguments, report format, and secrets must match the
implementation. Pin third-party GitHub Actions to reviewed commit SHAs
for a stricter supply-chain posture.

## 8. Production monitoring and feedback

Capture, with suitable redaction and access controls:

-   Request/trace ID and timestamp.
-   Prompt version, model/provider ID, embedding model ID, chunking
    configuration, and top-k.
-   Retrieved chunk IDs, source IDs, rank/scores, and retrieval latency.
-   Prompt template version, response, token usage, and generation
    latency.
-   User feedback (thumbs up/down) associated with the request/trace ID.
-   Evaluation outcomes and issue labels from reviewed failures.

Add a feedback endpoint such as `POST /feedback` with `request_id`,
`rating`, and optional `comment`. Validate input and avoid storing
secrets or unnecessary personal data. Define retention, access control,
and redaction policies before logging production prompts or documents.

Use monitoring to create a review queue, not automatic fine-tuning by
default. Inspect failure clusters first: missing source data, poor chunk
boundaries, embedding mismatch, weak ranking, prompt defects, model
changes, or evaluator errors. Update the benchmark only through a
reviewed and versioned process.

## 9. API endpoints to implement

  -----------------------------------------------------------------------
  Method                  Endpoint                Purpose
  ----------------------- ----------------------- -----------------------
  `POST`                  `/ingest`               Ingest an approved
                                                  corpus or trigger a
                                                  controlled indexing job

  `POST`                  `/query`                Run retrieval and
                                                  generate an answer with
                                                  citations

  `POST`                  `/feedback`             Save thumbs-up/down
                                                  feedback for a request

  `POST`                  `/evaluate`             Trigger an authorized
                                                  evaluation job

  `GET`                   `/health`               Health/readiness checks

  `GET`                   `/metrics`              Optional operational
                                                  metrics endpoint
  -----------------------------------------------------------------------

For production, protect ingestion and evaluation endpoints with
authentication and authorization. Do not expose unrestricted document
ingestion or expensive evaluation jobs publicly.

## 10. Build roadmap

### Phase 1 --- Working RAG baseline

-   [ ] Load a small, permitted document corpus.
-   [ ] Clean, chunk, embed, and index documents.
-   [ ] Build `/query` with retrieved-source citations.
-   [ ] Add unit tests for parsing, chunking, retrieval, and API
    validation.

### Phase 2 --- Evaluation harness

-   [ ] Generate a small synthetic dataset with source evidence.
-   [ ] Review and version the test set.
-   [ ] Run RAGAS metrics and export JSON/CSV reports.
-   [ ] Add retrieval metrics such as Recall@k where labels are
    available.

### Phase 3 --- CI/CD

-   [ ] Add GitHub Actions for tests and a fast PR benchmark.
-   [ ] Add a configurable quality-gate script.
-   [ ] Upload reports even when a gate fails.
-   [ ] Add a larger scheduled benchmark and compare against a baseline.

### Phase 4 --- Observability

-   [ ] Instrument traces with Langfuse (or Phoenix).
-   [ ] Record retrieved chunk IDs, model/prompt versions, latency, and
    token usage.
-   [ ] Add thumbs-up/down feedback tied to request IDs.
-   [ ] Build a reviewed failure-analysis loop.

### Phase 5 --- Deployment hardening

-   [ ] Build and test a Docker image.
-   [ ] Use environment variables or a secret manager for credentials.
-   [ ] Add authentication, rate limits, redaction, retention, and
    access controls.
-   [ ] Deploy to a target platform; adopt Kubernetes only if required
    by the use case.

## 11. Security and reliability checklist

-   Never commit `.env`, API keys, private corpora, or raw user
    conversations.
-   Provide `.env.example` with placeholder values only.
-   Pin dependencies and model versions; record configuration in each
    evaluation report.
-   Treat source documents as untrusted input; defend against prompt
    injection in retrieved content.
-   Keep retrieved text clearly separated from system instructions.
-   Test unanswerable questions and instruct the model to say when
    evidence is insufficient.
-   Use least-privilege credentials, authenticated admin endpoints, and
    request limits.
-   Redact sensitive content from traces and establish
    retention/deletion policies.
-   Record failed cases and metric distributions, not just a single
    aggregate score.
-   Ensure CI uses a stable, representative dataset and does not
    silently regenerate its test set on every run.

## 12. Definition of done

The project is ready for a portfolio demonstration when:

1.  A user can query an indexed corpus and receive an answer with source
    citations.
2.  A versioned evaluation dataset can be generated, reviewed, and
    rerun.
3.  Evaluation reports include Context Precision, Context Recall,
    Faithfulness, and Answer Relevancy.
4.  A deliberately regressed test run fails the CI quality gate.
5.  Traces link the question, retrieved chunks, generated answer,
    model/prompt versions, and feedback.
6.  A README explains setup, evaluation commands, thresholds,
    limitations, and a sample report.

## 13. Suggested portfolio demo

Demonstrate one controlled change at a time:

-   Baseline: current chunk size, embedding model, and prompt.
-   Experiment A: change chunk size or overlap.
-   Experiment B: swap embedding models.
-   Experiment C: modify the system prompt.
-   Show the metric comparison, retrieved evidence, failed examples, and
    CI pass/fail result.

A strong project is not merely a RAG chatbot. Its differentiator is
**reproducible evaluation, explainable regressions, and enforced quality
gates**.
