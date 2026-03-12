import sys
import os
import json
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_precision, 
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings

# clearml additions

from clearml import Task

task = Task.init(
    project_name="ragas_evaluation",
    task_name="mistral_ragas_metrics"
)

logger = task.get_logger()

# 1. SETUP ENVIRONMENT
load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
api_key = os.getenv('MISTRAL_API_KEY')

# 2. MISTRAL PATCH (Prevents the 'TypeError: dict + dict' crash)
class SafeUsage(dict):
    def __init__(self):
        super().__init__({"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0})
    def __add__(self, other): return self
    def __radd__(self, other): return self

class MistralSafeLLM(ChatMistralAI):
    def _generate(self, *args, **kwargs):
        res = super()._generate(*args, **kwargs)
        res.llm_output = {"token_usage": SafeUsage()} 
        return res
    async def _agenerate(self, *args, **kwargs):
        res = await super()._agenerate(*args, **kwargs)
        res.llm_output = {"token_usage": SafeUsage()}
        return res

# 3. INITIALIZE MODELS
llm = LangchainLLMWrapper(MistralSafeLLM(
    model="mistral-small-latest", 
    mistral_api_key=api_key,
    temperature=0
))

embeddings = LangchainEmbeddingsWrapper(MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=api_key
))

# 4. LOAD DATA (Keep original names for the evaluation phase)
with open("evaluation/eval_data.json", "r") as f:
    data = json.load(f)

# RAGAS 0.1.21 REQUIRES these specific keys to run: question, answer, contexts, ground_truth
dataset = Dataset.from_dict(data)

# 5. CONFIGURE METRICS
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
for m in metrics:
    m.llm = llm
    if hasattr(m, 'embeddings'):
        m.embeddings = embeddings

# 6. RUN EVALUATION
print("--- Starting RAGAS Evaluation (4 Metrics) ---")
result = evaluate(
    dataset,
    metrics=metrics,
    llm=llm,
    embeddings=embeddings,
    run_config=RunConfig(timeout=300, max_workers=1)
)

# 7. RENAME COLUMNS FOR THE FINAL CSV
print("\n=== FINAL RAGAS SCORES ===")
print(result)

df = result.to_pandas()

# log to clearml

faithfulness_score = df["faithfulness"].mean()
answer_relevancy_score = df["answer_relevancy"].mean()
context_precision_score = df["context_precision"].mean()
context_recall_score = df["context_recall"].mean()

logger.report_scalar("ragas_metrics", "faithfulness", faithfulness_score, iteration=0)
logger.report_scalar("ragas_metrics", "answer_relevancy", answer_relevancy_score, iteration=0)
logger.report_scalar("ragas_metrics", "context_precision", context_precision_score, iteration=0)
logger.report_scalar("ragas_metrics", "context_recall", context_recall_score, iteration=0)

# Now we rename the columns to satisfy your project's naming requirement
df.rename(columns={
    "question": "user_input",
    "answer": "response",
    "contexts": "retrieved_contexts",
    "ground_truth": "reference"
}, inplace=True)

df.to_csv("evaluation/ragas_results.csv", index=False)

# upload artifact to the task

task.upload_artifact(
    name="ragas_results",
    artifact_object="evaluation/ragas_results.csv"
)

print(f"✅ Results saved to evaluation/ragas_results.csv with custom column names.")
