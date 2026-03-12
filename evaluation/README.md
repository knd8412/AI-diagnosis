Run collect_data.py to collect the groundtruth data for the RAGAS metrics to be run over. 

Then run eval.py to get RAGAS results saved to ragas_results.csv. 

Added automated health check script in evaluation/tests/run_tests.py to verify API connectivity, Pinecone retrieval, and LLM response formatting. in order to use it simply run  the code from root:
python evaluation/tests/run_tests.py