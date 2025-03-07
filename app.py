import os
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
import streamlit as st
import matplotlib.pyplot as plt
import logging
from tavily import TavilyClient
from langchain_ollama import ChatOllama
import wikipediaapi
from transformers import pipeline
import yaml
import spacy
from getpass import getpass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    logger.error("TAVILY API key is missing in the environment.")
else:
    tavily = TavilyClient(api_key=TAVILY_API_KEY)

llm = ChatOllama(model="llama3")

def load_config() -> Dict[str, Any]:

    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

from spacy.lang.en.stop_words import STOP_WORDS


nlp = spacy.load("en_core_web_sm")

def extract_keywords(claim: str) -> List[str]:
    doc = nlp(claim)

    blacklist = {"claim", "fact", "check", "news", "article", "report", "study", "research"}

    keywords = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ["NOUN", "PROPN", "NUM", "ADJ"]
        and token.lemma_.lower() not in STOP_WORDS
        and token.lemma_.lower() not in blacklist
        and len(token.lemma_) >= 3
    ]

    keywords = list(set(keywords))

    return keywords

def fetch_wikipedia_articles(query: str) -> List[Dict[str, str]]:
      #It will Fetch articles from Wikipedia.
    user_agent = "MyResearchApp/1.0 ( https://github.com/ssonali6/Fact-Checking-AI-Assistant)"
    wiki_wiki = wikipediaapi.Wikipedia(language='en', user_agent=user_agent)
    try:
        page = wiki_wiki.page(query)
        if page.exists():
            return [{"title": page.title, "url": page.fullurl, "content": page.summary}]

        search_results = wiki_wiki.search(query)
        if search_results:

            page = wiki_wiki.page(search_results[0])
            if page.exists():
                return [{"title": page.title, "url": page.fullurl, "content": page.summary}]

        return []
    except Exception as e:
        logger.error(f"Error fetching Wikipedia articles: {e}")
        return []

class ResearchAgent:

    @staticmethod
    def fetch_tavily_articles(claim: str, search_params: Dict[str, Any]) -> Dict[str, Any]:  #It fetches articles from Tavily based on claim keywords.

        keywords = extract_keywords(claim)
        if len(keywords) < 2:
            query = claim
        else:
            query = f"{claim} {' '.join(keywords)}"

        logger.info(f"🔍 Tavily query: {query}")

        try:
            response = tavily.search(
                query=query,
                search_depth="advanced",
                max_results=10,
                include_domains=search_params.get("domains", []),
            )
            articles = response.get("results", [])[:5]

            if not response.get("results"):
                logger.warning("⚠️ No results found for this claim.")
                return {"articles": []}

            articles = [
                {
                    "title": res.get("title", "No title"),
                    "url": res.get("url", "No URL"),
                    "content": res.get("content", "No content available"),
                }
                for res in response["results"][:5]
            ]
            logger.info(f"✅ Retrieved {len(articles)} relevant articles from Tavily.")
            return {"articles": articles}
        except HTTPError as e:
            logger.error(f"⚠️ Tavily API error: {e}")
            return {"articles": []}
        except Exception as e:
            logger.error(f"⚠️ An unexpected error occurred: {e}")
            return {"articles": []}

    @staticmethod
    def fetch_articles(state: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch relevant news articles for the given claim using Tavily and Wikipedia."""
        claim = state["claim"]
        config = load_config()
        search_params = config.get("search_params", {})
        logger.info(f"🔍 Searching for articles related to: {claim}")

        tavily_results = ResearchAgent.fetch_tavily_articles(claim, search_params)

        wikipedia_results = fetch_wikipedia_articles(claim)

        articles = wikipedia_results + (tavily_results.get("articles") or [])
        return {"articles": articles[:5]}

class FactCheckAgent:

    @staticmethod
    def weight_evidence(state: Dict[str, Any]) -> Dict[str, Any]: #It Assigns reliability scores to sources based on a trusted list.

        articles = state["articles"]
        weighted_evidence = []

        config = load_config()
        trusted_sources = config.get("trusted_sources", [])

        if not trusted_sources:
            logger.warning("No trusted sources found in config.yaml. Using default list.")
            trusted_sources = ["mit.edu", "bbc.com", "theprint.in", "thewire.in", "timesofindia.indiatimes.com",
                               "reuters.com", "nature.com", "sciencedirect.com", "who.int", "nasa.gov"]

        for article in articles:
            source = article["url"]
            reliability_score = 1
            if any(trusted_source in source for trusted_source in trusted_sources):
                reliability_score = 2

            weighted_evidence.append({
                "title": article["title"],
                "content": article["content"],
                "url": article["url"],
                "weight": reliability_score,
            })

        weighted_evidence.sort(key=lambda x: x["weight"], reverse=True)

        logger.info(f"✅ {len(weighted_evidence)} articles weighted & sorted.")
        return {"weighted_evidence": weighted_evidence}


    @staticmethod
    def fact_check(state: Dict[str, Any]) -> Dict[str, Any]:  #It uses an LLM (Llama 3) to verify claim truthfulness.

        claim = state["claim"]
        weighted_evidence = state["weighted_evidence"]

        if not weighted_evidence:
            return {"result": "⚠️ No evidence found. Unable to fact-check."}

        formatted_evidence = []
        references = []
        for i, evidence in enumerate(weighted_evidence, start=1):
            formatted_evidence.append(f"[{i}] {evidence['content']}")
            references.append(f"[{i}] {evidence['title']} ({evidence['url']})")

        formatted_evidence = "\n\n".join(formatted_evidence)
        reference_list = "\n".join(references)

        prompt = (
    f"Claim: \"{claim}\"\n\n"
    f"Evidence:\n"
    f"{formatted_evidence}\n\n"
    f"Analyze the claim based on evidence and return one of:\n"
    f"✅ True\n❌ False\n⚠ Partially True"
    )


        try:
            response = llm.invoke(prompt)
            result = f"{response.content}\n\nReferences:\n{reference_list}"
            return {"result": result}
        except Exception as e:
            logger.error(f"Error during fact-checking: {e}")
            return {"result": f"⚠️ An error occurred: {e}"}


    @staticmethod
    def draft_answer(state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a draft answer using a multi-step process."""
        claim = state["claim"]
        evidence = state["weighted_evidence"]

        rough_draft_prompt = f"""
        Summarize the following evidence to support or refute the claim:
        Claim: "{claim}"
        Evidence: {evidence}
        """
        rough_draft = llm.invoke(rough_draft_prompt).content


        refined_draft_prompt = f"""
        Refine the following draft to make it more coherent and accurate:
        Draft: {rough_draft}
        """
        refined_draft = llm.invoke(refined_draft_prompt).content
        return {"draft": refined_draft}


    @staticmethod
    def detect_bias(text: str) -> str:
      """Detect bias in the given text."""
      bias_detector = pipeline("sentiment-analysis", truncation=True)
      try:
          result = bias_detector(text[:512])[0]
          if result["label"] == "NEGATIVE":
              return "⚠️ Potential bias detected: Negative sentiment."
          elif result["label"] == "POSITIVE":
              return "⚠️ Potential bias detected: Positive sentiment."
          return "✅ No significant bias detected."
      except Exception as e:
          logger.error(f"Error during bias detection: {e}")
          return f"⚠️ An error occurred during bias detection: {e}"

class GraphState(TypedDict):
    claim: str
    articles: List[Dict[str, str]]
    weighted_evidence: List[Dict[str, Any]]
    result: str

def create_fact_checking_pipeline():
   #It creates the LangGraph pipeline for fact-checking.
    pipeline = StateGraph(state_schema=GraphState)

    pipeline.add_node("research_node", ResearchAgent.fetch_articles)
    pipeline.add_node("weight_node", FactCheckAgent.weight_evidence)
    pipeline.add_node("fact_check_node", FactCheckAgent.fact_check)
    pipeline.add_node("draft_node", FactCheckAgent.draft_answer)

    pipeline.add_edge("research_node", "weight_node")
    pipeline.add_edge("weight_node", "fact_check_node")
    pipeline.add_edge("fact_check_node", "draft_node")
    pipeline.add_edge("draft_node", END)

    pipeline.set_entry_point("research_node")
    pipeline.set_finish_point("draft_node")

    compiled_pipeline = pipeline.compile()

    return compiled_pipeline

def generate_graph(sources: List[str], scores: List[int], filename: str = "/content/source_reliability.png") -> str:
    try:
        plt.figure(figsize=(10, 6))
        plt.barh(sources, scores, color=['green' if w == 2 else 'blue' for w in scores])
        plt.xlabel("Reliability Score")
        plt.ylabel("Source")
        plt.title("Source Reliability Scores")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        logger.info(f"✅ Source reliability chart saved as '{filename}'.")
        return filename
    except Exception as e:
        logger.error(f"Error generating graph: {e}")
        raise e

def generate_report(state: Dict[str, Any], graph_filename: str, filename: str = "report.md"):
    claim = state["claim"]
    articles = state["articles"]
    result = state["result"]
    weighted_evidence = state.get("weighted_evidence", [])

    summary = f"""
    ## Summary of Research Findings
    - **Total Articles Analyzed:** {len(articles)}
    - **Trusted Sources Used:** {len([e for e in weighted_evidence if e["weight"] == 2])}
    - **Fact-Check Verdict:** {result.splitlines()[0]}  # Extract the verdict (e.g., ✅ True)
    """

    with open(filename, "w") as file:
        file.write(f"# Fact-Check Report\n\n")
        file.write(f"## Claim\n{claim}\n\n")
        file.write(summary)
        file.write(f"## Research Findings\n")
        for i, article in enumerate(articles, start=1):
            file.write(f"{i}. [{article['title']}]({article['url']})\n")
        file.write(f"\n## Fact-Check Result\n{result}\n")

        if weighted_evidence:
            file.write("\n## Source Reliability\n")
            file.write(f"![Source Reliability]({graph_filename})\n")


def run_streamlit_app():    #Creates an interactive web app for fact-checking.

    st.title("Fact-Checking AI Assistant")
    claim = st.text_input("Enter a claim to fact-check:")

    if claim:
        st.session_state.articles = []
        st.session_state.weighted_evidence = []
        st.session_state.result = ""
        st.session_state.draft = ""
        st.session_state.graph_filename = ""

        with st.spinner("🔍 Researching..."):
            state = GraphState(claim=claim, articles=[], weighted_evidence=[], result="", draft="")

            pipeline = create_fact_checking_pipeline()
            state = pipeline.invoke(state)

            if "draft" not in state:
                draft_result = FactCheckAgent.draft_answer(state)
                state.update(draft_result)

            if "weighted_evidence" in state:
                state["sources"] = [e["title"] for e in state["weighted_evidence"]]
                state["scores"] = [e["weight"] for e in state["weighted_evidence"]]

            if "sources" in state and "scores" in state:
                st.session_state.graph_filename = generate_graph(state["sources"], state["scores"])
            else:
                st.warning("⚠️ No weighted evidence found. Unable to generate graph.")

        st.write("## Fact-Check Result")
        st.write(state["result"])

        st.write("## Draft Answer")
        if "draft" in state and state["draft"]:
            st.write(state["draft"])
        else:
            st.write("No draft available.")

        bias_result = FactCheckAgent.detect_bias(state["result"])
        st.write("## Bias Detection")
        st.write(bias_result)

        if state.get("weighted_evidence"):
            st.write("## Source Reliability Scores")
            reliability_table = [
                {"Source": e["title"], "Reliability Score": e["weight"]}
                for e in state["weighted_evidence"]
            ]
            st.table(reliability_table)

            st.write("## Source Reliability Chart")
            if "graph_filename" in st.session_state and os.path.exists(st.session_state.graph_filename):
                st.image(st.session_state.graph_filename)
            else:
                st.write("⚠️ Source reliability chart not found.")

            generate_report(state, st.session_state.graph_filename, "report.md")
            with open("report.md", "r") as file:
                st.download_button("Download Report", file, file_name="report.md")

    if st.button("Clear Results"):

        st.session_state.claim = ""
        st.session_state.articles = []
        st.session_state.weighted_evidence = []
        st.session_state.result = ""
        st.session_state.draft = ""
        st.session_state.graph_filename = ""

        st.rerun()

if __name__ == "__main__":
    run_streamlit_app()

