# conda activate chat-with-website
# streamlit run homepage.py

import os
import streamlit as st
import requests
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# --- Configuration ---
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")


# --- Functions ---
def fetch_user_repos(owner):
    """Fetch all repositories for a given GitHub username."""
    url = f"https://api.github.com/users/{owner}/repos"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(
            f"Error fetching repositories: {response.status_code} - {response.text}"
        )
        return None
    repos = response.json()
    return repos


def fetch_readme(owner, repo):
    """Fetch the README file for a given repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
        "Accept": "application/vnd.github.v3.raw",  # Retrieve raw README content
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(f"Error fetching README: {response.status_code} - {response.text}")
        return None
    readme = response.text
    return readme


def generate_readme_summary(repo_name, readme_text):
    """Generate a summary of the README content using Ollama LLM."""
    prompt = f"""
    Repository: {repo_name}
    README Content:
    {readme_text}
    Please provide a concise and informative summary of this project's README. The summary should describe what the project is about, its main features, and any key insights that can be gleaned from the README.
    """
    try:
        llm = ChatOllama(
            model="llama3.3:70b",
            temperature=0.0,
            base_url="http://localhost:8888",
        )
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | llm
        response = chain.invoke({})
        return response
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None


# --- Button click handlers ---
def load_repos():
    owner = st.session_state.github_username
    with st.spinner(f"Fetching repositories for {owner}..."):
        st.session_state.repos = fetch_user_repos(owner)
    if st.session_state.repos:
        st.success(f"Found {len(st.session_state.repos)} repositories")


def summarize_repo(repo_name):
    st.session_state.selected_repo = repo_name
    owner = st.session_state.github_username
    with st.spinner(f"Fetching README for {repo_name}..."):
        st.session_state.readme_content = fetch_readme(owner, repo_name)
    if st.session_state.readme_content:
        with st.spinner("Generating summary using LLM..."):
            st.session_state.summary = generate_readme_summary(
                repo_name, st.session_state.readme_content
            )


def summarize_all_repos():
    owner = st.session_state.github_username
    repos = st.session_state.repos
    if not repos:
        st.error("No repositories loaded.")
        return
    # Reset all_summaries before starting
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


def introduce_him(info):
    """Using what the summaries of the different repo, introduce who he is."""
    prompt = f"""
    Below are concatenated summaries of the README files from a developer's GitHub repositories:

    {all_concatenated_summaries}

    Based on these summaries, please provide a detailed introduction of the person behind these projects. In your analysis, please address the following:
    1. Who is this person and what can we infer about their background?
    2. What are the common themes or topics of their projects?
    3. What technical abilities, skills, or expertise are demonstrated through these projects?
    4. What hobbies, interests, or non-technical passions might be inferred?
    5. Are there any unique or special attributes that set this person apart?

    Please provide a well-structured and comprehensive introduction that weaves these elements together.
    """

    try:
        llm = ChatOllama(
            model="llama3.3:70b",
            temperature=0.0,
            base_url="http://localhost:8888",
        )
        # llm = ChatOllama(model="llama3.2:latest", temperature=0.0)
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | llm
        response = chain.invoke({})
        return response
    except Exception as e:
        st.error(f"Error generating introduction statement: {str(e)}")
        return None


# --- Streamlit UI ---
st.title("RepoNarrator")

# Input for GitHub username
st.text_input("GitHub Username", value="ldw129", key="github_username")

# Create two columns for the two buttons: Load Repositories and Summarize All ReadMe
col1, col2 = st.columns(2)
with col1:
    if st.button("Load Repositories"):
        load_repos()
with col2:
    if st.button("Summarize All ReadMe"):
        summarize_all_repos()

# Display repositories if they've been loaded
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
                    response = st.session_state.summary.content
                    st.write(response)

# Display Summaries for All Repositories if available
all_concatenated_summaries = ""
if st.session_state.all_summaries:
    st.subheader("Summaries for All Repositories")
    for repo_name, summary in st.session_state.all_summaries.items():
        st.write(f"### {repo_name}")
        st.write(summary.content)
        all_concatenated_summaries += f"### {repo_name}\n{summary.content}\n\n"

if len(all_concatenated_summaries) > 10:
    response = introduce_him(all_concatenated_summaries)
    st.write(response.content)
