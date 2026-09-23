RELEVANCE_PROMPT = """
You are an expert grader. Your task is to evaluate if the retrieved documents are relevant to the user query.

Be extremely strict. If the documents do not contain the specific information needed to answer the query, they are not relevant.

Return your response in the following JSON format:
{{
  "score": (0-100, where 100 is perfectly relevant and 0 is completely irrelevant),
  "label": ("correct", "ambiguous", or "incorrect"),
  "reasoning": "explanation for the score and label"
}}

Query: {query}
Documents: {context}
"""

GROUNDEDNESS_PROMPT = """
You are an expert grader. Your task is to determine if the provided answer is grounded in the retrieved context.
An answer is grounded if every claim it makes is supported by the context. 

Return your response in the following JSON format:
{{
  "score": (0-100, where 100 is perfectly grounded and 0 is completely hallucinated),
  "reasoning": "explanation of why it is or is not grounded"
}}

Context: {context}
Answer: {answer}
"""

REWRITE_PROMPT = """
The previous retrieval attempt failed to find relevant documents. 
Rewrite the following user query to make it more effective for vector search. 
Maintain the original intent but expand it with potential keywords or synonyms.

Original Query: {query}

Rewritten Query:"""

EXPANSION_PROMPT = """
You are a search query expansion expert. Given a user query, generate 3 diverse but semantically equivalent phrasings of the query to improve retrieval recall.

Return the phrasings as a simple list, one per line, without numbering or bullets.

Query: {query}
"""

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
