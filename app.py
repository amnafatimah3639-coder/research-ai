"""A small Streamlit interface for a one-agent CrewAI research report."""

import streamlit as st


st.set_page_config(page_title="Research Report Builder", page_icon="📚", layout="centered")
st.title("Research Report Builder")
st.caption("Enter a topic to search the web and draft a sourced report.")


def get_groq_key() -> str | None:
    """Read the key from Streamlit secrets, with an environment fallback."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    if not key:
        import os

        key = os.environ.get("GROQ_API_KEY")
    return key


topic = st.text_area(
    "Research topic",
    placeholder="For example: How urban trees reduce heat in cities",
    height=100,
    key="research_topic_input",
)
run = st.button("Generate report", type="primary", disabled=not topic.strip())
run = st.button(
    "Generate report",
    type="primary",
    disabled=not topic.strip(),
    key="generate_report_button",
)

if run:
    api_key = get_groq_key()
    if not api_key:
        st.error(
            "Groq API key not found. Add GROQ_API_KEY to .streamlit/secrets.toml "
            "for local use, or to your app's Streamlit Community Cloud secrets."
        )
        st.stop()

    with st.spinner("Searching the web and preparing your report…"):
        try:
            # Keep framework imports inside the action so a broken cloud
            # dependency install produces a useful message in the app.
            from crewai import Agent, Crew, LLM, Process, Task

            from search_tool import DuckDuckGoSearchTool

            llm = LLM(
                model="groq/openai/gpt-oss-120b",
                api_key=api_key,
                temperature=0.2,
            )
            researcher = Agent(
                role="Research analyst",
                goal="Create a clear, balanced report grounded in web search results.",
                backstory=(
                    "You are a careful research analyst. You distinguish evidence from "
                    "interpretation and never invent sources, dates, statistics, or quotes."
                ),
                llm=llm,
                tools=[DuckDuckGoSearchTool()],
                verbose=False,
                allow_delegation=False,
                max_iter=5,
            )
            task = Task(
                description=(
                    "Research this topic: {topic}\n\n"
                    "Use the web search tool for several focused queries. Write a beginner-friendly "
                    "Markdown report with: a concise title; an executive summary; 3–5 thematic "
                    "sections; key takeaways; limitations or disagreements in the evidence; and "
                    "a Sources section with the exact source titles and URLs returned by search. "
                    "Cite claims inline using numbered references like [1]. Use only information "
                    "supported by retrieved results. If evidence is thin or sources conflict, say so. "
                    "Do not claim to have read a full page when only a search snippet was returned."
                ),
                expected_output=(
                    "A structured Markdown research report with inline numbered citations and a "
                    "matching list of source titles and clickable URLs."
                ),
                agent=researcher,
            )
            crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=False)
            result = crew.kickoff(inputs={"topic": topic.strip()})
            st.markdown(str(result))
            st.caption("Review the linked sources before relying on the report.")
        except ModuleNotFoundError as exc:
            missing = exc.name or "an unknown package"
            st.error(f"A required Python package is missing: `{missing}`.")
            st.markdown(
                "Check that `requirements.txt` is committed in the repository root "
                "(or beside `app.py`), then reboot the Community Cloud app."
            )
            st.caption("The full diagnostic is available in Manage app → Logs.")
        except Exception as exc:
            st.error("The report could not be generated.")
            st.code(str(exc))
            st.info(
                "Check your API key, internet connection, package installation, and Groq account "
                "limits. See README.md for common fixes."
            )
