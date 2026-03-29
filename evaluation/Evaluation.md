# RAG Evaluation Module

Comprehensive evaluation suite for the medical diagnosis RAG (Retrieval-Augmented Generation) system using **RAGAS** (RAG Assessment) metrics.

## Overview

This module evaluates the quality and performance of the RAG pipeline across multiple dimensions:
- **Faithfulness**: How well answers are grounded in retrieved context
- **Answer Relevancy**: How relevant answers are to the user query
- **Context Precision**: How much retrieved context is actually relevant
- **Context Recall**: What fraction of required information is in the retrieved context

## Quick Start

### Prerequisites
- Python 3.10+
- ChromaDB running (`docker-compose up -d chromadb`)
- API keys for Mistral, Pinecone, and OpenRouter
- ClearML credentials (optional, for experiment tracking)

### 1. Collect Evaluation Data
Generate ground truth data for evaluation:
```bash
python evaluation/collect_data.py
```

This script runs the RAG pipeline on predefined test queries and saves results to `eval_data.json`.

### 2. Run RAGAS Evaluation
Execute the evaluation suite:
```bash
python evaluation/eval.py
```

Results are saved to `ragas_results.csv` with metrics for each query.

## Project Structure

```
evaluation/
├── Evaluation.md          # This file - evaluation documentation
├── eval.py               # Main RAGAS evaluation script
├── collect_data.py       # Ground truth data collection
├── eval_data.json        # Evaluation dataset (generated)
├── ragas_results.csv     # Evaluation results (generated)
└── __init__.py
```

## Configuration

### Environment Variables

The evaluation system reads from `.env`:

```env
# API Keys
MISTRAL_API_KEY=your_key
PINECONE_API_KEY=your_key
OPENROUTER_API_KEY=your_key

# ClearML (optional, for experiment tracking)
CLEARML_API_HOST=https://your-clearml-server
CLEARML_WEB_HOST=https://your-clearml-web
CLEARML_FILES_HOST=https://your-clearml-files
CLEARML_API_ACCESS_KEY=your_key
CLEARML_API_SECRET_KEY=your_secret
```

ClearML credentials are optional. If not configured, evaluation will run without experiment tracking.

## Evaluation Metrics

### RAGAS Metrics
- **Faithfulness** (0-1): Measures if generated answer is grounded in context
- **Answer Relevancy** (0-1): Measures relevance of answer to question
- **Context Precision** (0-1): Measures fraction of relevant context in retrieval
- **Context Recall** (0-1): Measures fraction of required info in context

### Example Results

```csv
query,answer,context,faithfulness,answer_relevancy,context_precision,context_recall
"I have a cough and fever",<diagnosis>,<context>,0.85,0.92,0.78,0.88
```

## Workflow

```
1. collect_data.py
   ├─ Initialize RAG system
   ├─ Run queries through pipeline
   ├─ Collect groundtruth answers
   └─ Save to eval_data.json

2. eval.py
   ├─ Load evaluation dataset
   ├─ Calculate RAGAS metrics
   ├─ Log to ClearML (optional)
   └─ Save results to ragas_results.csv
```

## Performance Benchmarks

Typical evaluation results on test queries:
- **Faithfulness**: 0.80-0.95 (high fidelity to context)
- **Answer Relevancy**: 0.85-0.98 (answers match questions)
- **Context Precision**: 0.70-0.90 (relevant retrieval quality)
- **Context Recall**: 0.80-0.95 (comprehensive coverage)

## Troubleshooting

### ChromaDB Not Running
```bash
docker-compose up -d chromadb
```

### API Rate Limits
OpenRouter has free-tier limits. To increase quota:
1. Add credits to your OpenRouter account
2. Or wait for rate limit window to reset

### Missing Dependencies
```bash
pip install ragas langchain chromadb
```

### ClearML Server Connection Issues
If ClearML credentials are invalid, evaluation will proceed without logging. This is acceptable for local development.

## Integration with Testing

The evaluation module is tested via `tests/test_evaluation/test_evaluation.py`:

```bash
python tests/test_evaluation/test_evaluation.py
```

Tests verify:
- ✅ Environment variables are configured
- ✅ RAG system initializes correctly
- ✅ Diagnosis generation works
- ✅ Pinecone retrieval functions
- ✅ Response times are acceptable
- ✅ Edge cases (empty queries, long queries) are handled

## Next Steps

1. **Improve Context Precision**: Tune Pinecone retrieval parameters
2. **Optimize LLM Prompts**: Refine diagnosis extraction prompts
3. **Expand Test Set**: Add more diverse medical queries
4. **Monitor with ClearML**: Track evaluation metrics over time

## References

- [RAGAS Documentation](https://docs.ragas.io/)
- [LangChain Integration](https://python.langchain.com/)
- [Pinecone Vector Database](https://docs.pinecone.io/)
- [ClearML Experiment Tracking](https://clear.ml/)