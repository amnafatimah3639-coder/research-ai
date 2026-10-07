
"""A small CrewAI tool wrapping the DDGS web-search library."""

from typing import Type

from crewai.tools import BaseTool
from ddgs import DDGS
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    query: str = Field(..., description="A focused web search query.")


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo web search"
    description: str = (
        "Search the web using DDGS, which includes a DuckDuckGo backend. "
        "Use a specific query and return source titles, URLs, and snippets."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        results = DDGS(timeout=12).text(query, backend="duckduckgo", max_results=6)
        if not results:
            return "No results were returned. Try a shorter or more specific search query."
        rows = []
        for number, item in enumerate(results, start=1):
            title = item.get("title", "Untitled source")
            url = item.get("href", "")
            snippet = item.get("body", "")
            rows.append(f"[{number}] {title}\nURL: {url}\nSnippet: {snippet}")
        return "\n\n".join(rows)
