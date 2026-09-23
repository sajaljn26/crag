# Self-Correcting RAG Pipeline

This project implements a self-correcting Retrieval-Augmented Generation (RAG) pipeline designed to minimize hallucinations and improve retrieval accuracy. Unlike standard RAG pipelines that simply retrieve documents and generate an answer, this system employs a dual-loop verification mechanism: it grades the relevance of retrieved documents and the groundedness of the final answer, automatically rewriting queries or regenerating responses when they fail to meet quality thresholds.

## Architecture

The pipeline operates in two distinct corrective phases:

### 1. Corrective Retrieval Loop
- **Retrieve**: Fetches the top-K documents from the local ChromaDB vector store.
- **Grade Relevance**: An LLM evaluates if the retrieved documents contain sufficient information to answer the query.
- **Self-Correct**: If the documents are deemed irrelevant or insufficient, the system uses a query rewriter to optimize the search terms and retries the retrieval (up to 3 attempts).

### 2. Corrective Generation Loop
- **Generate**: Produces an answer strictly derived from the retrieved context.
- **Grade Groundedness**: An LLM checks if the answer is fully supported by the context (checking for hallucinations).
- **Self-Correct**: If the answer is not grounded, the grader provides a specific failure reason. This reason is fed back into the generator to correct the answer in a subsequent attempt (up to 2 attempts).

## Why This is Different from Basic RAG

Basic RAG follows a linear "Retrieve $\rightarrow$ Generate" path. If the retriever fails, the generator often tries to "fill in the gaps" with internal knowledge, leading to hallucinations. 

This pipeline introduces **self-correction at two critical points**:
1. **Pre-generation**: It ensures the model doesn't attempt to answer with bad data.
2. **Post-generation**: It ensures the output is a faithful representation of the source material, not a hallucination.

## Tech Stack

- **LLM**: [Groq](https://groq.com/) (`openai/gpt-oss-20b`) for generation, grading, and rewriting.
- **Embeddings**: `sentence-transformers` (local) for generating document and query vectors.
- **Vector Store**: `ChromaDB` (local) for efficient similarity search.
- **Orchestration**: Python with Pydantic for structured data validation.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd self-correcting-rag
   ```

2. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```text
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Pipeline**:
   ```bash
   python main.py
   ```

## Example Scenarios

### Successful Answer
**Query**: "What is the company's policy on remote work?"
- **Retrieval**: Finds "Remote Work Policy 2024.pdf".
- **Relevance Grade**: Relevant.
- **Generation**: "The company allows up to 3 days of remote work per week."
- **Groundedness Grade**: Grounded.
- **Result**: Returns the answer.

### Correctly Refused Answer
**Query**: "What is the CEO's favorite color?"
- **Retrieval**: Finds general company bio.
- **Relevance Grade**: Irrelevant $\rightarrow$ Rewrite query $\rightarrow$ Retry $\rightarrow$ Irrelevant.
- **Result**: "I'm sorry, but I could not find any relevant context in the documents to answer this question."

## Design Decisions

- **Retry Limits**: Retrieval is limited to 3 attempts and generation to 2. This prevents infinite loops and limits API costs while providing enough headroom for the LLM to correct its mistakes.
- **Groundedness Feedback**: Instead of simply telling the generator "this is wrong," the groundedness grader returns a specific **failure reason** (e.g., *"The answer claims the project ends in December, but the context says November"*). Feeding this specific feedback into the regeneration prompt significantly increases the probability of a correct second attempt.
- **JSON Tracing**: Every run is saved to the `logs/` folder to allow developers to audit exactly where a pipeline failed—whether it was a retrieval failure or a generation hallucination.
