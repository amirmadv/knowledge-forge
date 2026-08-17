# Retrieval Evaluation

KnowledgeForge can evaluate the hybrid retrieval pipeline against a small, versioned gold dataset.
The evaluation is intentionally offline at the metric layer: it measures the note identifiers returned
by the configured retriever and does not ask the chat model to judge answer quality.

## Dataset format

The dataset is JSON. It may be a top-level array or an object containing a `cases` array.
Each case contains a query and one or more relevant note slugs:

```json
{
  "cases": [
    {
      "query": "linear regression",
      "relevant": ["linear-regression"]
    },
    {
      "query": "gradient descent",
      "relevant": ["gradient-descent", "optimization"]
    }
  ]
}
```

`relevant` must contain note slugs, not display titles. The dataset should therefore live alongside
the vault knowledge it evaluates and should be updated deliberately when note slugs change.

## Run an evaluation

```powershell
knowledgeforge-ai evaluate .\evals\retrieval.json
```

The command reports:

- `Precision@K`: fraction of returned top-K notes that are relevant.
- `Recall@K`: fraction of all relevant notes found in top-K.
- `MRR`: mean reciprocal rank of the first relevant result.

Change K and configure minimum quality thresholds:

```powershell
knowledgeforge-ai evaluate .\evals\retrieval.json `
  --k 5 `
  --min-precision 0.60 `
  --min-recall 0.80 `
  --min-mrr 0.75
```

By default, a failed quality gate exits with code `2`, which makes the command suitable for CI.
Use `--no-fail-on-gate` when you want a report without failing the calling process.

For per-query diagnostics:

```powershell
knowledgeforge-ai evaluate .\evals\retrieval.json --details
```

## Quality-gate principle

Thresholds should be established from a representative evaluation set rather than chosen to make
one run pass. A retrieval change should be considered an improvement only when it maintains or
improves the agreed quality floor without creating unacceptable regressions in another metric.
