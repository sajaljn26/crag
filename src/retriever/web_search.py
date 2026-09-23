from duckduckgo_search import DDGS

class WebSearchTool:
    def search(self, query: str, max_results: int = 3) -> list[str]:
        print(f"Performing web search for: {query}...")
        try:
            with DDGS() as ddgs:
                results = [r['body'] for r in ddgs.text(query, max_results=max_results)]
                return results
        except Exception as e:
            print(f"Web search failed: {e}")
            return []
