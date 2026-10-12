# RAG Chunking Strategy Comparison

This project compares two chunking strategies on the supplied Type 2 Diabetes reference document, retrieves evidence for the assignment's ten questions, and optionally asks an LLM to answer from that evidence.

## Project contents

- `src/chunking.py` — fixed-size and structure-aware chunking
- `src/retrieval.py` — sentence-transformer embeddings and top-k cosine retrieval
- `src/generation.py` — optional OpenAI LLM generation grounded in retrieved chunks
- `src/evaluate.py` — runs both strategies and saves evaluation CSVs
- `data/` — source document
- `results/` — generated CSV outputs

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows activation:
```bash
.venv\Scripts\activate
```

macOS/Linux activation:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

The first run downloads `sentence-transformers/all-MiniLM-L6-v2`, so internet access is needed.

## Run retrieval-only mode

```bash
python src/evaluate.py
```

## Run with LLM answers

Set your OpenAI API key in the environment. Do not commit a real API key to GitHub.

Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="your_api_key_here"
python src/evaluate.py --llm
```

macOS/Linux:
```bash
export OPENAI_API_KEY="your_api_key_here"
python src/evaluate.py --llm
```

The default model is `gpt-4.1-mini`; override it with the `OPENAI_MODEL` environment variable if needed. LLM calls require API access and may incur charges. The LLM is told to answer only from retrieved evidence and to admit when the reference is insufficient.

## Strategies

### A — Naive
500-character windows with 50-character overlap. Easy to implement, but can split a sentence, list, or table.

### B — Structure-aware
Attempts to preserve Markdown headings and paragraph boundaries with a target maximum of 1,000 characters. Long paragraphs use a fallback split. This is structural chunking, not embedding-based semantic chunking.

## Embeddings and retrieval

Both chunk sets use `sentence-transformers/all-MiniLM-L6-v2`. Vectors are normalized and ranked with NumPy dot products (cosine similarity). The top three chunks are retrieved for each question by default.

## Evaluation procedure

`results/evaluation_results.csv` contains one row per retrieved chunk and includes the retrieved text, similarity score, and optional LLM answer. Review the evidence manually:
- `yes`: sufficient correct evidence is present
- `partial`: evidence is relevant but incomplete
- `no`: required evidence is absent

For retrieval accuracy, count a question as successful when at least one of its top-k chunks is labelled `yes`:
`retrieval accuracy = successful questions / 10 × 100`

Do not treat similarity score as proof of correctness. If `--llm` is enabled, manually check the generated answer against the reference and fill `answer_correct`. `results/chunk_statistics.csv` contains chunk counts and sizes; fill the accuracy columns after the review. Do not invent scores before running the code.

## Expected source limitation

Question 9 asks for the recommended blood glucose target range for most adults. The supplied reference gives diagnostic thresholds but does not explicitly provide this treatment target. A grounded LLM response should acknowledge that the source does not contain enough information.

## Comparison report

Add a question-by-question table and a 200–400-word analysis after reviewing actual outputs. Discuss measured accuracy, chunk counts, average chunk size, questions where each strategy did better, and the simplicity/context-preservation trade-off.

## Safety and scope

This is an educational demonstration, not medical advice. The LLM is restricted to the retrieved reference passages, but its output must still be checked. Retrieval-only evaluation is available without an API key; LLM mode requires an OpenAI API key.
