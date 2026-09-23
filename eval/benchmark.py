import json
from src.orchestrator.pipeline import RAGPipeline

def run_benchmark():
    with open("eval/test_dataset.json", "r") as f:
        dataset = json.load(f)
    
    pipeline = RAGPipeline()
    results = []
    
    for item in dataset:
        query = item["query"]
        expected = item["expected"]
        print(f"Testing: {query}")
        answer = pipeline.run(query)
        
        # Very simple keyword-based check for demonstration
        passed = expected.lower() in answer.lower()
        results.append({
            "query": query,
            "expected": expected,
            "actual": answer,
            "passed": passed
        })
        print(f"Result: {'PASS' if passed else 'FAIL'}")
        
    with open("eval/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nBenchmark complete. Results saved to eval/benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()
