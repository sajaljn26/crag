# Advanced Self-Correcting Hybrid RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline featuring hybrid search, cross-encoder reranking, three-way corrective grading, and automated safety guards.

## 🚀 Key Features

### 🔍 Advanced Retrieval & Reranking
- **Hybrid Search**: Combines **ChromaDB** vector similarity with **BM25** keyword search to maximize recall across both semantic and exact matches.
- **Multi-Query Expansion**: Generates 3 diverse, semantically equivalent phrasings of the user query to overcome retrieval gaps.
- **Cross-Encoder Reranking**: Uses a `ms-marco-MiniLM` cross-encoder to re-score the top 10 hybrid candidates, keeping only the top 5 high-precision documents.

### 🛠️ Corrective RAG (CRAG) Logic
The pipeline uses a three-way classification system for retrieved documents:
- **Correct**: Documents are sufficient $\to$ Proceed to generation.
- **Ambiguous**: Documents are partially related $\to$ Rewrite query $\to$ Retry retrieval.
- **Incorrect**: Documents are irrelevant $\to$ Trigger **Web Search Fallback** (via DuckDuckGo).

### 🛡️ Safety & Reliability
- **Confidence-Based Gating**: Retries and fallbacks are triggered based on numeric confidence scores (0-100) rather than binary flags, with thresholds configurable in `src/config.py`.
- **Prompt Injection Defense**: An input validation layer scans retrieved documents for override patterns (e.g., "ignore previous instructions") and filters out malicious content before it reaches the generator.
- **Faithfulness Grading**: Every generated answer is scored for groundedness. If it falls below the threshold, the pipeline triggers a corrective regeneration.

### 📈 Observability & Ops
- **Cost Tracking**: Real-time tracking of prompt/completion tokens and estimated USD cost per query, logged in every trace.
- **Detailed Tracing**: Full JSON logs for every run, including expanded queries, reranking scores, and grader reasoning.
- **CI Eval Gating**: GitHub Actions workflow that runs benchmarks on every push, failing the build if Relevance (<85%) or Faithfulness (<80%) drops.

---

## 📐 Architecture

**Query Flow:**
1. **Query Expansion**: $\text{User Query} \to \text{3 Variants} + \text{Original}$.
2. **Hybrid Retrieval**: $\text{Variants} \to (\text{Vector Search} \cup \text{BM25}) \to \text{Merged Candidates}$.
3. **Reranking**: $\text{Candidates} \to \text{Cross-Encoder} \to \text{Top 5 Docs}$.
4. **Relevance Grading**: $\text{Top 5 Docs} \to \text{LLM Grader} \to \text{Score (0-100) \& Label}$.
5. **Logic Gate**:
    - If $\text{Score} \ge \text{Threshold} \to$ **Generate**.
    - If $\text{Label} = \text{"incorrect"} \to$ **Web Search} \to$ **Generate**.
    - If $\text{Label} = \text{"ambiguous"} \to$ **Rewrite Query} \to$ **Step 2**.
6. **Generation**: $\text{Context} \to \text{LLM} \to \text{Answer}$.
7. **Groundedness Grading**: $\text{Answer} \to \text{LLM Grader} \to \text{Score}$.
    - If $\text{Score} < \text{Threshold} \to$ **Regenerate**.

---

## 📊 Evaluation Results

Based on the latest benchmark run across the test dataset:

| Metric | Result | Status |
| :--- | :--- | :--- |
| **Average Relevance** | 92.5% | ✅ Pass |
| **Average Faithfulness** | 88.0% | ✅ Pass |
| **Avg. Tokens per Query** | ~1,200 | ℹ️ |
| **Avg. Cost per Query** | \$0.00024 | ℹ️ |

---

## ⚙️ Setup & Usage

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
Set your API key in a `.env` file:
```env
GROQ_API_KEY=your_api_key_here
```

### Running the Pipeline
```bash
python main.py
```

### Running Benchmarks
```bash
python eval/benchmark.py
```

### Analyzing Costs
```bash
python eval/cost_summary.py
```
