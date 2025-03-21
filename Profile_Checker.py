# conda activate chat-with-website
# streamlit run homepage.py

import os
import re
import streamlit as st
import requests
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a persistent session for all API calls
session = requests.Session()

# Initialize session state variables
if "repos" not in st.session_state:
    st.session_state.repos = None
if "selected_repo" not in st.session_state:
    st.session_state.selected_repo = None
if "readme_content" not in st.session_state:
    st.session_state.readme_content = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "all_summaries" not in st.session_state:
    st.session_state.all_summaries = {}
if "extracted_keywords" not in st.session_state:
    st.session_state.extracted_keywords = None
if "final_query" not in st.session_state:
    st.session_state.final_query = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None

# --- Configuration ---
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# --- Helper Functions ---


def extract_top_keywords_llm(text):
    """
    Use the LLM to analyze the concatenated summaries and return the top 8 keywords
    as a comma-separated list (with no extra explanation).
    """
    prompt = f"""
    Given the following concatenated README summaries from various repositories, extract the top 8 keywords that best capture the candidate's unique cybersecurity focus. Provide only the keywords as a comma-separated list, with no additional text or explanation.

    Summaries:
    {text}

    Keywords:
    """
    try:
        llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0.0,
            base_url="http://localhost:8888",
        )
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | llm
        response = chain.invoke({})
        # Expect a comma-separated list; split and clean the keywords.
        keywords_str = response.content.strip()
        keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
        return keywords
    except Exception as e:
        return []


def search_github_repositories(query):
    """
    Use the GitHub Search API to search for repositories matching the query.
    """
    url = "https://api.github.com/search/repositories"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": 5,  # Retrieve top 5 results
    }
    response = session.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    return None


# --- Existing Functions ---


def fetch_user_repos(owner):
    """Fetch all repositories for a given GitHub username."""
    url = f"https://api.github.com/users/{owner}/repos"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    response = session.get(url, headers=headers)
    if response.status_code != 200:
        return None
    return response.json()


def fetch_readme(owner, repo):
    """Fetch the README file for a given repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
        "Accept": "application/vnd.github.v3.raw",
    }
    response = session.get(url, headers=headers)
    if response.status_code != 200:
        return None
    return response.text


def generate_readme_summary(repo_name, readme_text):
    """Generate a summary of the README content using Ollama LLM."""
    prompt = f"""
    Repository: {repo_name}
    README Content:
    {readme_text}
    Please provide a concise and informative summary of this project's README. The summary should describe what the project is about, its main features, and any key insights.
    """
    try:
        llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0.0,
            base_url="http://localhost:8888",
        )
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | llm
        response = chain.invoke({})
        return response
    except Exception as e:
        return None


def summarize_repo(repo_name):
    st.session_state.selected_repo = repo_name
    owner = st.session_state.github_username
    with st.spinner(f"Fetching README for {repo_name}..."):
        st.session_state.readme_content = fetch_readme(owner, repo_name)
    if st.session_state.readme_content and len(st.session_state.readme_content) > 20:
        with st.spinner("Generating summary using LLM..."):
            st.session_state.summary = generate_readme_summary(
                repo_name, st.session_state.readme_content
            )


def summarize_all_repos():
    owner = st.session_state.github_username
    repos = st.session_state.repos
    if not repos:
        return
    st.session_state.all_summaries = {}
    for repo in repos:
        repo_name = repo.get("name", "Unnamed")
        with st.spinner(f"Fetching README for {repo_name}..."):
            readme_text = fetch_readme(owner, repo_name)
        if readme_text:
            with st.spinner(f"Generating summary for {repo_name}..."):
                summary = generate_readme_summary(repo_name, readme_text)
            if summary:
                st.session_state.all_summaries[repo_name] = summary


def introduce_him(all_concatenated_summaries):
    """Using the concatenated summaries, introduce who the developer is."""
    prompt = f"""
    Below are concatenated summaries of the README files from a developer's GitHub repositories:

    {all_concatenated_summaries}

    Based on these summaries, please provide a detailed introduction of the person behind these projects. Address:
    1. Who is this person and what can we infer about their background?
    2. What are the common themes or topics of their projects?
    3. What technical abilities, skills, or expertise are demonstrated?
    4. What hobbies, interests, or non-technical passions might be inferred?
    5. What unique or special attributes set this person apart?

    Provide a well-structured and comprehensive introduction.
    """
    try:
        llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0.0,
            base_url="http://localhost:8888",
        )
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | llm
        response = chain.invoke({})
        return response
    except Exception as e:
        return None


# --- Streamlit UI ---

st.title("RepoNarrator")

# Input GitHub username
st.text_input("GitHub Username", value="ldw129", key="github_username")

col1, col2 = st.columns(2)
with col1:
    if st.button("Load Repositories"):
        owner = st.session_state.github_username
        with st.spinner(f"Fetching repositories for {owner}..."):
            st.session_state.repos = fetch_user_repos(owner)
        if st.session_state.repos:
            st.success(f"Found {len(st.session_state.repos)} repositories")
with col2:
    if st.button("Summarize All ReadMe"):
        summarize_all_repos()

# Display repositories if loaded
if st.session_state.repos:
    st.subheader(f"Repositories for {st.session_state.github_username}")
    repo_container = st.container()
    with repo_container:
        for repo in st.session_state.repos:
            repo_name = repo.get("name", "Unnamed")
            with st.expander(f"**{repo_name}**"):
                if st.button("Summarize README", key=f"btn_{repo_name}"):
                    summarize_repo(repo_name)
                if (
                    st.session_state.selected_repo == repo_name
                    and st.session_state.summary is not None
                ):
                    st.subheader("README Summary")
                    st.write(st.session_state.summary.content)

# Display concatenated summaries and introduction
all_concatenated_summaries = ""
if st.session_state.all_summaries:
    st.subheader("Summaries for All Repositories")
    for repo_name, summary in st.session_state.all_summaries.items():
        st.write(f"### {repo_name}")
        st.write(summary.content)
        all_concatenated_summaries += f"### {repo_name}\n{summary.content}\n\n"

if len(all_concatenated_summaries) > 10:
    response = introduce_him(all_concatenated_summaries)
    st.write(f"### Introduction of who {st.session_state.github_username} is")
    st.write(response.content)

# --- Search by Keywords Section ---
st.write("## Search for Related Projects")
if st.button("Search by Keywords"):
    if st.session_state.all_summaries:
        all_concatenated = ""
        for repo_name, summary in st.session_state.all_summaries.items():
            all_concatenated += f" {summary.content} "
        # Extract top 8 keywords using the LLM
        keywords = extract_top_keywords_llm(all_concatenated)
        if keywords:
            st.session_state.extracted_keywords = keywords
        else:
            st.write("Could not extract keywords from the summaries.")
    else:
        st.write("No summaries available. Please summarize repositories first.")

# If keywords have been extracted, display a dropdown for selection and a Run Search button.
if st.session_state.get("extracted_keywords"):
    selected_keyword = st.selectbox(
        "Select a keyword to search:",
        st.session_state.extracted_keywords,
        key="selected_keyword",
    )
    if st.button("Run Search", key="run_search_button"):
        # Build the final query using the selected keyword and append additional qualifiers.
        final_query = f"{selected_keyword} topic:cybersecurity language:python"
        st.session_state.final_query = final_query
        with st.spinner("Searching GitHub repositories..."):
            results = search_github_repositories(final_query)
        st.session_state.search_results = {
            "final_query": final_query,
            "results": results,
        }
        st.write("### GitHub Search Results")
        st.write("**Final Query:**", final_query)
        if results and results.get("items"):
            for repo in results.get("items", []):
                st.write(f"**{repo.get('full_name')}**")
                st.write(f"Description: {repo.get('description')}")
                st.write(
                    f"Stars: {repo.get('stargazers_count')}, Updated: {repo.get('updated_at')}"
                )
                st.write("---")
        else:
            st.write("No search results found.")
