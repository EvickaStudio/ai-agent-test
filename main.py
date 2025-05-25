import os
import time
from datetime import timedelta, date
from typing import List, Type

import arxiv
from crewai import Agent, Crew, Task
from crewai.tools import BaseTool
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool,
)
from pydantic import BaseModel, Field
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.openai_api_key
os.environ["OPENAI_MODEL_NAME"] = settings.openai_model_name
os.environ["SERPER_API_KEY"] = settings.serper_api_key


# Tool Definitions
class FetchArxivPapersInput(BaseModel):
    """Input schema for FetchArxivPapersTool."""

    model_config = {"arbitrary_types_allowed": True}

    target_date: date = Field(..., description="Target date to fetch papers for.")


class FetchArxivPapersTool(BaseTool):
    name: str = "fetch_arxiv_papers"
    description: str = "Fetches all ArXiv papers from selected categories submitted on the target date."
    args_schema: Type[BaseModel] = FetchArxivPapersInput

    def _run(self, target_date: date) -> List[dict]:
        AI_CATEGORIES = ["cs.CL"]

        start_date_str = target_date.strftime(
            "%Y%m%d000000"
        )  # Ensure HHMMSS for Arxiv query
        end_date_obj = target_date + timedelta(days=1)
        end_date_str = end_date_obj.strftime(
            "%Y%m%d000000"
        )  # Ensure HHMMSS for Arxiv query

        client = arxiv.Client(
            page_size=100,
            delay_seconds=3,
        )

        all_papers = []

        for category in AI_CATEGORIES:
            print(f"Fetching papers for category: {category}")

            search_query = (
                f"cat:{category} AND submittedDate:[{start_date_str} TO {end_date_str}]"
            )

            search = arxiv.Search(
                query=search_query,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                max_results=None,
            )

            category_papers = []
            for result in client.results(search):
                category_papers.append(
                    {
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "summary": result.summary,
                        "published": result.published,
                        "url": result.entry_id,
                    }
                )
                time.sleep(3)

            print(f"Fetched {len(category_papers)} papers from {category}")
            all_papers.extend(category_papers)

        return all_papers


# Initialize Tools
docs_tool = DirectoryReadTool(directory="./your-directory")
file_tool = FileReadTool()
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()
arxiv_search_tool = FetchArxivPapersTool()


# Agent Definitions
researcher = Agent(
    role="Senior Researcher",
    goal="Find the top 10 papers from the search results from ArXiv on {date}. "
    "Rank them appropriately.",
    backstory="You are a senior researcher with a deep understanding of all topics in AI and AI research. "
    "You are able to identify the best research papers based on the title and abstract.",
    verbose=True,
    tools=[arxiv_search_tool],
)

frontend_engineer = Agent(
    role="Senior Frontend & AI Engineer",
    goal="Compile the results into a HTML file.",
    backstory="You are a competent frontend engineer writing HTML and CSS with decades of experience. "
    "You have also been working with AI for decades and understand it well.",
    verbose=True,
)


# Task Definitions
research_task = Task(
    description=(
        "Find the top 10 research papers from the search results from ArXiv on {date}."
    ),
    expected_output=(
        "A list of top 10 research papers with the following information in the following format:\n"
        "- Title\n"
        "- Authors\n"
        "- Abstract\n"
        "- Link to the paper"
    ),
    agent=researcher,
    human_input=True,
)

reporting_task = Task(
    description="Compile the results into a detailed report in a HTML file.",
    expected_output=(
        "An HTML file with the results in the following format:\n"
        "Top 10 AI Research Papers published on {date}\n"
        "- Title (which on clicking opens the paper in a new tab)\n"
        "- Authors\n"
        "- Short summary of the abstract (2-4 sentences)"
    ),
    agent=frontend_engineer,
    context=[research_task],
    output_file="./output/ai_research_report.html",
    human_input=True,
)


# Crew Definition
arxiv_research_crew = Crew(
    agents=[researcher, frontend_engineer],
    tasks=[research_task, reporting_task],
    verbose=True,
)


# Main Execution
if __name__ == "__main__":
    crew_inputs = {"date": "2025-03-12"}
    result = arxiv_research_crew.kickoff(inputs=crew_inputs)
