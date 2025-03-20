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
    st.session_state.readme_content = {}
if "code_content" not in st.session_state:
    st.session_state.code_content = {}
if "summaries" not in st.session_state:
    st.session_state.summaries = {}  # Combined README and code summaries

# --- Configuration ---
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")


# --- Functions ---
def search_user_repos(owner):
    """Search for repositories owned by a GitHub user using the Search API."""
    url = "https://api.github.com/search/repositories"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
    }
    params = {
        "q": f"user:{owner}",
        "per_page": 100,
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        st.error(
            f"Error searching repositories: {response.status_code} - {response.text}"
        )
        return None
    results = response.json()
    return results["items"]


def fetch_readme(owner, repo):
    """Fetch the README file for a given repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
        "Accept": "application/vnd.github.v3.raw",
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(f"Error fetching README: {response.status_code} - {response.text}")
        return None
    return response.text


def search_repo_code(owner, repo, keyword="main.py"):
    """Search for a key code file in a specific repository."""
    url = "https://api.github.com/search/code"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
    }
    params = {
        "q": f"{keyword} repo:{owner}/{repo}",
        "per_page": 1,  # Get the top match
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200 or response.json()["total_count"] == 0:
        return None
    results = response.json()
    return results["items"][0]  # Return the first match


def fetch_code_content(url):
    """Fetch the raw content of a code file from its raw URL."""
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
        "Accept": "application/vnd.github.v3.raw",
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(
            f"Error fetching code content: {response.status_code} - {response.text}"
        )
        return None
    return response.text


def generate_combined_summary(repo_name, readme_text, code_file=None, code_text=None):
    """Generate a combined summary of README and code content."""
    prompt = f"""
    Repository: {repo_name}
    README Content:
    {readme_text}
    {'Code File: ' + code_file if code_file else ''}
    {code_text if code_text else 'No key code file found.'}
    Please provide a concise documentation summary for this repository. Include:
    1. An overview of the project based on the README.
    2. Key features or insights from the README.
    3. A description of the provided code file (if available), including its purpose and functionality.
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
def load_and_document():
    owner = st.session_state.github_username
    with st.spinner(f"Searching repositories for {owner}..."):
        st.session_state.repos = search_user_repos(owner)
    if not st.session_state.repos:
        st.error("No repositories found.")
        return

    st.session_state.summaries = {}
    st.session_state.readme_content = {}
    st.session_state.code_content = {}

    for repo in st.session_state.repos:
        repo_name = repo.get("name", "Unnamed")
        with st.spinner(f"Processing {repo_name}..."):
            # Fetch README
            readme_text = fetch_readme(owner, repo_name)
            if readme_text:
                st.session_state.readme_content[repo_name] = readme_text

            # Search for a key code file
            code_result = search_repo_code(
                owner, repo_name, st.session_state.code_keyword
            )
            code_file = None
            code_text = None
            if code_result:
                code_file = code_result["name"]
                raw_url = (
                    code_result["html_url"]
                    .replace("github.com", "raw.githubusercontent.com")
                    .replace("/blob/", "/")
                )
                code_text = fetch_code_content(raw_url)
                if code_text:
                    st.session_state.code_content[repo_name] = code_text

            # Generate combined summary
            summary = generate_combined_summary(
                repo_name, readme_text, code_file, code_text
            )
            if summary:
                st.session_state.summaries[repo_name] = summary

    st.success(f"Documented {len(st.session_state.summaries)} repositories")


# --- Streamlit UI ---
st.title("RepoNarrator: Automated Documentation Generator")

# Input for GitHub username and code keyword
st.text_input("GitHub Username", value="ldw129", key="github_username")
st.text_input(
    "Code Keyword (e.g., 'main.py', 'app.py')", value="main.py", key="code_keyword"
)

# Button to load and document
if st.button("Generate Documentation"):
    load_and_document()

# Display documentation
if st.session_state.repos and st.session_state.summaries:
    st.subheader(f"Documentation for {st.session_state.github_username}'s Repositories")
    for repo in st.session_state.repos:
        repo_name = repo.get("name", "Unnamed")
        with st.expander(f"**{repo_name}**"):
            if repo_name in st.session_state.summaries:
                st.write("### Summary")
                st.write(st.session_state.summaries[repo_name].content)
                if repo_name in st.session_state.readme_content:
                    st.write("### README Content")
                    st.text(st.session_state.readme_content[repo_name])
                if repo_name in st.session_state.code_content:
                    st.write("### Code Content")
                    st.code(st.session_state.code_content[repo_name])
            else:
                st.write("No documentation generated for this repository.")
