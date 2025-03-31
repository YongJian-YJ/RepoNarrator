import os
import re
import streamlit as st
import requests
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time

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

# --- Configuration ---
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# --- Helper Functions ---


def auto_escape_prompt(prompt: str, exclude: list = None) -> str:
    """
    Automatically escape any curly-brace expressions in the prompt
    that are meant to be literal text rather than variable placeholders.

    If a substring inside { ... } is not in the exclude list,
    it will be replaced with double braces.
    """
    if exclude is None:
        exclude = []

    def replacer(match):
        content = match.group(1)
        if content in exclude:
            return match.group(0)
        else:
            return "{{" + content + "}}"

    return re.sub(r"\{([^{}]+)\}", replacer, prompt)


def fetch_user_repos(owner):
    """Fetch all repositories for a given GitHub username."""
    url = f"https://api.github.com/users/{owner}/repos"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    try:
        response = session.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch repos for {owner}: {response.status_code}")
            return None
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repos: {e}")
        return None


def fetch_readme(owner, repo):
    """Fetch the README file for a given repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
        "Accept": "application/vnd.github.v3.raw",
    }
    try:
        response = session.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching README: {e}")
        return None


def generate_readme_summary(repo_name, readme_text):
    """Generate a summary of the README content using Ollama LLM."""
    prompt = f"""
    Repository: {repo_name}
    README Content:
    {readme_text}
    Please provide a concise and informative summary of this project's README.
    The summary should describe what the project is about, its main features, and any key insights.
    For example, analyze expressions like {{ciphertext_username.hex()}}, {{ciphertext_password.hex()}}, 
    {{secret_password.decode('utf-8')}} and even {{e}} as literal text.
    """
    prompt = auto_escape_prompt(prompt)
    try:
        llm = ChatOllama(
            model="gemma3:1b",
            temperature=0.0,
        )
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | llm
        response = chain.invoke({})
        return response
    except Exception as e:
        print(f"Error generating README summary: {e}")
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
    prompt = auto_escape_prompt(prompt)
    try:
        llm = ChatOllama(
            model="gemma3:1b",
            temperature=0.0,
        )
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | llm
        response = chain.invoke({})
        return response
    except Exception as e:
        print(f"Error generating introduction: {e}")
        return None


# --- Functions for Code Usage Fingerprint ---


def detect_language(file_name):
    """Detect programming language based on file extension."""
    extension_map = {".py": "python", ".c": "c", ".java": "java", ".js": "javascript"}
    _, ext = os.path.splitext(file_name.lower())
    return extension_map.get(ext, "unknown")


def fetch_code_file(owner, repo_name, path="", visited=None, depth=0, max_depth=5):
    """
    Recursively fetch the content of the first code file with a function in the repository.
    Returns: (code_text, language, (snippet, function_block)) or (None, None, None)
    """
    if depth > max_depth:
        print(f"Reached max depth ({max_depth}) at {path}. Stopping recursion.")
        return None, None, None

    if visited is None:
        visited = set()

    url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    try:
        response = session.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch contents at {url}: {response.status_code}")
            return None, None, None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching contents at {url}: {e}")
        time.sleep(1)
        return None, None, None

    contents = response.json()
    supported_extensions = [".py", ".c", ".java", ".js"]

    print(f"Checking directory: {path if path else 'root'} (depth: {depth})")

    for item in contents:
        item_path = item["path"]
        item_type = item["type"]

        print(f"Processing item: {item_path} (type: {item_type})")

        if item_path in visited:
            print(f"Skipping already visited: {item_path}")
            continue

        visited.add(item_path)

        if item_type == "file" and any(
            item["name"].lower().endswith(ext) for ext in supported_extensions
        ):
            print(f"Found file: {item_path}")
            try:
                download_url = item["download_url"]
                code_response = session.get(download_url, headers=headers, timeout=10)
                if code_response.status_code == 200:
                    language = detect_language(item["name"])
                    code_text = code_response.text
                    snippet, function_block = extract_useful_function(code_text)
                    if snippet and function_block:
                        print(f"Found useful function in {item_path}: {snippet}")
                        return code_text, language, (snippet, function_block)
                    else:
                        print(f"No useful function found in {item_path}")
                else:
                    print(
                        f"Failed to download {item_path}: {code_response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {item_path}: {e}")
                time.sleep(1)
            finally:
                if "code_response" in locals():
                    code_response.close()

        elif item_type in ("dir", "submodule", "symlink"):
            print(f"Entering subdirectory: {item_path} (depth: {depth + 1})")
            sub_result, sub_language, sub_function = fetch_code_file(
                owner, repo_name, item_path, visited, depth + 1, max_depth
            )
            if sub_result and sub_language and sub_function:
                return sub_result, sub_language, sub_function
            time.sleep(0.5)

    return None, None, None


def extract_useful_function(code_text, max_chars=1000):
    """Extract a useful function definition and its block from the code."""
    if not code_text:
        return None, None

    # Updated regex to capture "async def" as well
    function_keywords = r"\b(async\s+def|def|function|int|void|float|double|char|public|private|protected)\b"
    lines = code_text.split("\n")
    function_block = []
    in_function = False
    indent_level = 0
    brace_count = 0
    snippet = None

    # Adjusted common utility function names (removed 'main')
    common_functions = {
        "xor_bytes",
        "to_string",
        "to_int",
        "get",
        "set",
        "add",
        "subtract",
        "multiply",
        "divide",
        "print",
        "log",
        "init",
        "setup",
        "loop",
    }

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not in_function and re.search(function_keywords, stripped_line):
            # Updated regex includes async def
            match = re.search(
                r"\b(async\s+def|def|function|int|void|float|double|char|public|private|protected)\s+(\w+)\s*\(",
                stripped_line,
            )
            if match:
                function_name = match.group(2)
                print(f"Detected function: {function_name}")
                if function_name.lower() in common_functions:
                    print(f"Skipping common function: {function_name}")
                    continue

                in_function = True
                snippet = stripped_line
                function_block.append(line)
                if "{" in stripped_line:
                    brace_count = line.count("{") - line.count("}")
                else:
                    indent_level = len(line) - len(line.lstrip())
        elif in_function:
            function_block.append(line)
            if brace_count > 0:
                brace_count += line.count("{") - line.count("}")
                if brace_count <= 0:
                    break
            else:
                current_indent = len(line) - len(line.lstrip())
                if stripped_line == "" or (
                    current_indent <= indent_level and stripped_line
                ):
                    break
        if in_function and len("\n".join(function_block)) > max_chars:
            break

    if not function_block:
        print("No function block extracted.")
        return None, None

    function_text = "\n".join(function_block)
    complexity_indicators = [
        r"\bif\b",
        r"\bfor\b",
        r"\bwhile\b",
        r"\btry\b",
        r"\bexcept\b",
        r"\bthrow\b",
        r"\breturn\b",
        r"\braise\b",
        r"\bassert\b",
    ]
    has_complexity = any(
        re.search(indicator, function_text, re.IGNORECASE)
        for indicator in complexity_indicators
    )
    line_count = len(function_block)
    print(
        f"Function complexity: has_complexity={has_complexity}, line_count={line_count}"
    )
    # Accept functions that are at least 2 lines long even if they lack complexity
    if not has_complexity and line_count < 2:
        print("Function not considered useful: lacks complexity or is too short.")
        return None, None

    return snippet, function_text


def search_snippet(snippet, language="python"):
    """Search for the snippet across GitHub using the Search API."""
    if not snippet:
        return 0
    url = "https://api.github.com/search/code"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    query = f'"{snippet}" language:{language}'
    params = {"q": query, "per_page": 1}
    try:
        response = session.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            result = response.json()
            return result["total_count"]
        else:
            print(f"Failed to search snippet: {response.status_code}")
            return 0
    except requests.exceptions.RequestException as e:
        print(f"Error searching snippet: {e}")
        return 0


def generate_code_fingerprint(selected_repo_names):
    """
    Generate a code usage fingerprint and summary using LLM for selected repositories.
    Returns a tuple: (fingerprint_summary, snippet_details)
    """
    owner = st.session_state.github_username
    repos = st.session_state.repos
    if not repos:
        st.write("No repositories loaded.")
        return "No repositories loaded.", []

    # Filter repos based on selected names
    selected_repos = [repo for repo in repos if repo["name"] in selected_repo_names]
    if not selected_repos:
        st.write("No valid repositories selected.")
        return "No valid repositories selected.", []

    snippet_details = []
    search_results = []
    full_files = []

    for repo in selected_repos[:3]:
        repo_name = repo["name"]
        print(f"\nProcessing repository: {repo_name}")
        code_text, language, function_data = fetch_code_file(owner, repo_name)
        if code_text and language != "unknown" and function_data:
            snippet, _ = function_data
            matches = search_snippet(snippet, language)
            snippet_details.append(
                {
                    "function_name": (
                        snippet.split("(")[0].strip() if "(" in snippet else snippet
                    ),
                    "repo_name": repo_name,
                    "matches": matches,
                    "code_snippet": code_text,
                    "purpose": "This function serves a specific role in the repository. (Auto-extracted)",
                    "usage": "Usage information not available automatically.",
                    "notes": "No additional notes available.",
                }
            )
            search_results.append((repo_name, snippet, matches))
            full_files.append((repo_name, code_text))
        else:
            st.warning(
                f"No useful function found in {repo_name} after searching all directories. Skipping this repository."
            )

    search_summary = (
        "\n".join(
            [
                f"- Repository '{repo}': Code snippet used for search: '{snippet}' | Similarity found in {matches} repositories"
                for repo, snippet, matches in search_results
            ]
        )
        if search_results
        else "No snippets found."
    )

    code_summary = (
        "\n\n".join(
            [f"Full file from {repo}:\n{sample}" for repo, sample in full_files]
        )
        if full_files
        else "No files with useful functions available."
    )

    prompt = f"""
    Below are the search results for code snippets extracted from the user's repositories. Each entry includes:
    - The repository name.
    - The code snippet that was used for searching.
    - The number of repositories where a similar code snippet was found.
    - (Optionally, if available, the sources or repository names where similar code is found.)

    Search Summary:
    {search_summary}

    Below are the full files from the user's repositories containing the extracted functions,
    which are used to evaluate coding style, structure, and documentation practices:

    {code_summary}

    Please provide a detailed analysis covering:
    1. The originality of the user's code based on the provided search results.
    2. A breakdown of the specific code snippets used for searching and their relevance.
    3. The number of similarities found for each snippet.
    4. (Optionally) Any information on the sources where similar code was found.
    5. An overall assessment of the coding style, structure, and documentation practices.

    Provide a well-structured, comprehensive summary.
    """
    prompt = auto_escape_prompt(prompt)
    try:
        llm = ChatOllama(
            model="gemma3:1b",
            temperature=0.0,
        )
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | llm
        response = chain.invoke({})
        return response.content, snippet_details
    except Exception as e:
        print(f"Error generating fingerprint summary: {e}")
        return f"Error generating summary: {str(e)}", snippet_details


# --- Streamlit UI ---

st.title("RepoNarrator")

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

st.write("## Code Usage Fingerprint")
if st.session_state.repos:
    repo_names = [repo["name"] for repo in st.session_state.repos]
    default_selection = repo_names[:3] if len(repo_names) >= 3 else repo_names
    selected_repos = st.multiselect(
        "Select up to 3 repositories to analyze:",
        options=repo_names,
        default=default_selection,
        max_selections=3,
        key="selected_repos",
    )
    if len(selected_repos) == 3:
        st.info(
            "You’ve selected the maximum of 3 repositories. Deselect one to choose another."
        )
    if st.button("Generate Code Fingerprint"):
        with st.spinner("Analyzing code usage..."):
            fingerprint_summary, snippet_details = generate_code_fingerprint(
                selected_repos
            )
        st.write("### Code Usage Fingerprint Summary")
        st.write(fingerprint_summary)

        st.subheader("Code Snippets Searched from GitHub Database")
        if snippet_details:
            for detail in snippet_details:
                func_name = detail.get("function_name", "Unnamed Function")
                repo = detail.get("repo_name", "Unknown Repo")
                matches = detail.get("matches", "N/A")
                purpose = detail.get("purpose", "No description provided.")
                usage_info = detail.get("usage", "No usage info provided.")
                notes = detail.get("notes", "No additional notes available.")
                code_snippet = detail.get("code_snippet", "No code available.")

                st.write(f"**{func_name} Function (from {repo}):**")
                st.write(f"**Purpose:** {purpose}")
                st.write(f"**Usage:** {usage_info}")
                st.write(f"**Similarity:** Found in {matches} repositories")
                st.write(f"**Notes:** {notes}")

                with st.expander("Show full code snippet"):
                    st.code(code_snippet, language="python")
        else:
            st.write("No code snippets available for display.")
else:
    st.write("Please load repositories first to analyze code.")
