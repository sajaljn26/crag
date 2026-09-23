RELEVANCE_PROMPT = """
You are an expert grader. Your task is to evaluate if the retrieved documents are relevant to the user query.
If the documents contain information that can help answer the query, return 'YES'. 
Otherwise, return 'NO'.

Query: {query}
Documents: {context}

Response (YES/NO):"""

GROUNDEDNESS_PROMPT = """
You are an expert grader. Your task is to determine if the provided answer is grounded in the retrieved context.
An answer is grounded if every claim it makes is supported by the context. 
If the answer contains information NOT present in the context, it is NOT grounded.

Context: {context}
Answer: {answer}

Return your response in the following JSON format:
{{
  "is_grounded": "YES" or "NO",
  "reason": "explanation of why it is or is not grounded"
}}
"""

REWRITE_PROMPT = """
The previous retrieval attempt failed to find relevant documents. 
Rewrite the following user query to make it more effective for vector search. 
Maintain the original intent but expand it with potential keywords or synonyms.

Original Query: {query}

Rewritten Query:"""

GENERATION_PROMPT = """
You are a helpful assistant. Answer the user's question strictly using the provided context.
If the context does not contain the answer, state that you do not know.
Do not use outside knowledge.

Context: {context}
Question: {query}

Answer:"""

REGENERATION_PROMPT = """
The previous answer was graded as not grounded.
Failure Reason: {reason}

Please rewrite the answer using ONLY the provided context. Ensure that every claim is directly supported by the context.

Context: {context}
Question: {query}
Previous Answer: {previous_answer}

Corrected Answer:"""
