import json
import os
import glob
from collections import defaultdict

def summarize_costs():
    logs_dir = "logs"
    trace_files = glob.glob(os.path.join(logs_dir, "trace_*.json"))
    
    if not trace_files:
        print("No trace files found in logs directory.")
        return

    total_runs = len(trace_files)
    aggregate_metrics = {
        "total_llm_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_cost": 0.0
    }
    
    per_query_metrics = []

    for tf in trace_files:
        with open(tf, 'r') as f:
            trace = json.load(f)
            metrics = trace.get("metrics", {})
            if not metrics:
                continue
                
            aggregate_metrics["total_llm_calls"] += metrics.get("total_llm_calls", 0)
            aggregate_metrics["total_prompt_tokens"] += metrics.get("total_prompt_tokens", 0)
            aggregate_metrics["total_completion_tokens"] += metrics.get("total_completion_tokens", 0)
            aggregate_metrics["total_cost"] += metrics.get("total_cost", 0.0)
            
            per_query_metrics.append({
                "query": trace.get("query"),
                "cost": metrics.get("total_cost", 0.0),
                "tokens": metrics.get("total_prompt_tokens", 0) + metrics.get("total_completion_tokens", 0)
            })

    print("=== RAG Pipeline Cost Summary ===")
    print(f"Total Queries Processed: {total_runs}")
    print(f"Total LLM Calls:         {aggregate_metrics['total_llm_calls']}")
    print(f"Total Prompt Tokens:    {aggregate_metrics['total_prompt_tokens']:,}")
    print(f"Total Completion Tokens: {aggregate_metrics['total_completion_tokens']:,}")
    print(f"Estimated Total Cost:   ${aggregate_metrics['total_cost']:.6f}")
    print("-" * 32)
    print(f"Average Cost per Query: ${aggregate_metrics['total_cost']/total_runs if total_runs > 0 else 0:.6f}")
    print(f"Average Tokens per Query: { (aggregate_metrics['total_prompt_tokens'] + aggregate_metrics['total_completion_tokens'])/total_runs if total_runs > 0 else 0:.2f}")
    print("=================================")

if __name__ == "__main__":
    summarize_costs()
