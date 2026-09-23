import json
import sys
from src.orchestrator.pipeline import RAGPipeline

def run_benchmark():
    with open("eval/test_dataset.json", "r") as f:
        dataset = json.load(f)
    
    pipeline = RAGPipeline()
    results = []
    
    total_relevance_score = 0
    total_faithfulness_score = 0
    num_queries = len(dataset)
    
    for item in dataset:
        query = item["query"]
        expected = item["expected"]
        print(f"Testing: {query}")
        
        answer, trace = pipeline.run(query)
        
        # Calculate Relevance: the final retrieval attempt's score
        # If no attempts, score is 0. Otherwise, take the score of the successful/last attempt.
        rel_score = 0
        if trace["retrieval_attempts"]:
            # We look at the attempt that actually broke the loop or the last one
            last_attempt = trace["retrieval_attempts"][-1]
            rel_score = last_attempt.get("relevance_score", 0)
            
        # Calculate Faithfulness (Groundedness): the final generation attempt's score
        faith_score = 0
        if trace["generation_attempts"]:
            last_gen = trace["generation_attempts"][-1]
            faith_score = last_gen.get("groundedness_score", 0)
            
        total_relevance_score += rel_score
        total_faithfulness_score += faith_score
        
        passed = expected.lower() in answer.lower()
        results.append({
            "query": query,
            "expected": expected,
            "actual": answer,
            "passed": passed,
            "relevance": rel_score,
            "faithfulness": faith_score
        })
        print(f"Result: {'PASS' if passed else 'FAIL'} | Rel: {rel_score} | Faith: {faith_score}")
        
    avg_relevance = total_relevance_score / num_queries if num_queries > 0 else 0
    avg_faithfulness = total_faithfulness_score / num_queries if num_queries > 0 else 0
    
    with open("eval/benchmark_results.json", "w") as f:
        json.dump({
            "results": results,
            "metrics": {
                "avg_relevance": avg_relevance,
                "avg_faithfulness": avg_faithfulness
            }
        }, f, indent=2)
    
    print("\n=== Benchmark Summary ===")
    print(f"Average Relevance: {avg_relevance:.2f}%")
    print(f"Average Faithfulness: {avg_faithfulness:.2f}%")
    
    # Fail build if thresholds are not met
    if avg_relevance < 85 or avg_faithfulness < 80:
        print("\nFAILURE: Benchmark thresholds not met!")
        print(f"Required: Relevance >= 85%, Faithfulness >= 80%")
        sys.exit(1)
    
    print("\nSUCCESS: Benchmark thresholds met.")

if __name__ == "__main__":
    run_benchmark()
