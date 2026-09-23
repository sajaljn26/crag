import re
import logging

class InjectionDetector:
    def __init__(self):
        # Common prompt injection patterns
        self.patterns = [
            r"ignore (all )?previous instructions",
            r"disregard (all )?previous instructions",
            r"forget (everything|all instructions)",
            r"now you are (a|an)",
            r"you are now (a|an)",
            r"act as (a|an)",
            r"system prompt",
            r"override (the )?system",
            r"bypass (the )?filters",
            r"new instructions:",
            r"stop acting as",
            r"you must now",
            r"ignore the following",
            r"instead of answering",
            r"from now on",
        ]
        self.regex = re.compile("|".join(self.patterns), re.IGNORECASE)

    def is_injection(self, text: str) -> bool:
        """Check if a piece of text contains potential prompt injection patterns."""
        if not text:
            return False
        return bool(self.regex.search(text))

    def filter_documents(self, documents: list[str]) -> tuple[list[str], list[str]]:
        """
        Filters a list of documents.
        Returns a tuple of (clean_documents, flagged_documents).
        """
        clean = []
        flagged = []
        for doc in documents:
            if self.is_injection(doc):
                flagged.append(doc)
            else:
                clean.append(doc)
        return clean, flagged
