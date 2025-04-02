# conda activate chat-with-website
# streamlit run Profile_checker.py

import os
import re
import streamlit as st
import requests
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time

# Load Model
# llm = ChatOllama(
#     model="llama3.1:8b",
#     temperature=0.0,
#     base_url="http://localhost:8888",
# )

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0.0,
)

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


def fetch_code_files(
    owner, repo_name, path="", visited=None, depth=0, max_depth=3, max_functions=2
):
    """
    Memory-efficient version of fetch_code_files with better error handling and resource management.
    """
    if depth > max_depth:
        print(f"Reached max depth ({max_depth}) at {path}. Stopping recursion.")
        return []

    if visited is None:
        visited = set()

    results = []

    url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

    # Explicitly close connections and manage resources
    try:
        # Use a context manager for the response to ensure resources are released
        with session.get(url, headers=headers, timeout=10) as response:
            if response.status_code != 200:
                print(f"Failed to fetch contents at {url}: {response.status_code}")
                return results

            # Parse contents immediately and clear response data
            try:
                contents = response.json()
            except Exception as e:
                print(f"Error parsing JSON from {url}: {e}")
                return results
    except requests.exceptions.RequestException as e:
        print(f"Error fetching contents at {url}: {e}")
        time.sleep(2)  # Longer backoff time
        return results

    # Garbage collect to free memory
    import gc

    gc.collect()

    # Process only a limited number of files at once
    supported_extensions = [".py", ".c", ".java", ".js"]
    file_items = [
        item
        for item in contents
        if item["type"] == "file"
        and any(item["name"].lower().endswith(ext) for ext in supported_extensions)
    ]

    # Limit number of files processed per directory
    file_items = file_items[:5]  # Process at most 5 files per directory

    for item in file_items:
        item_path = item["path"]

        if item_path in visited:
            continue

        visited.add(item_path)

        # Get file metadata but don't download yet
        download_url = item.get("download_url")
        if not download_url:
            continue

        # Process the file
        try:
            with session.get(
                download_url, headers=headers, timeout=10
            ) as code_response:
                if code_response.status_code != 200:
                    continue

                language = detect_language(item["name"])
                code_text = code_response.text

                # Extract functions with a size limit
                file_functions = extract_all_functions(
                    code_text, max_functions=2, max_chars=500
                )

                for snippet, function_block in file_functions:
                    results.append((code_text, language, (snippet, function_block)))

                    # Break early if we have enough functions
                    if len(results) >= max_functions:
                        return results
        except Exception as e:
            print(f"Error processing {item_path}: {e}")

        # Force garbage collection after each file
        gc.collect()
        time.sleep(0.5)  # Add delay between requests

    # Then process a limited number of directories
    if len(results) < max_functions:
        dir_items = [
            item for item in contents if item["type"] in ("dir", "submodule", "symlink")
        ]

        # Limit directories processed
        dir_items = dir_items[:2]  # Process at most 2 subdirectories per level

        for item in dir_items:
            item_path = item["path"]

            sub_results = fetch_code_files(
                owner,
                repo_name,
                item_path,
                visited,
                depth + 1,
                max_depth,
                max_functions - len(results),
            )

            results.extend(sub_results)
            if len(results) >= max_functions:
                return results[:max_functions]

            gc.collect()  # Force garbage collection
            time.sleep(1)  # Longer delay between directory processing

    return results


def extract_all_functions(code_text, max_functions=3, max_chars=1000):
    """Extract multiple useful function definitions from the code."""
    if not code_text:
        return []

    # Updated regex to capture "async def" as well
    function_keywords = r"\b(async\s+def|def|function|int|void|float|double|char|public|private|protected)\b"
    lines = code_text.split("\n")
    results = []
    i = 0

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

    while i < len(lines) and len(results) < max_functions:
        line = lines[i]
        stripped_line = line.strip()

        if re.search(function_keywords, stripped_line):
            match = re.search(
                r"\b(async\s+def|def|function|int|void|float|double|char|public|private|protected)\s+(\w+)\s*\(",
                stripped_line,
            )

            if match:
                function_name = match.group(2)
                print(f"Detected function: {function_name}")

                if function_name.lower() in common_functions:
                    print(f"Skipping common function: {function_name}")
                    i += 1
                    continue

                # Extract function block
                function_block = []
                function_block.append(line)
                snippet = stripped_line

                # Track braces or indentation
                if "{" in stripped_line:
                    brace_count = line.count("{") - line.count("}")
                    j = i + 1
                    while j < len(lines) and brace_count > 0:
                        next_line = lines[j]
                        function_block.append(next_line)
                        brace_count += next_line.count("{") - next_line.count("}")
                        j += 1
                        if len("\n".join(function_block)) > max_chars:
                            break
                else:
                    indent_level = len(line) - len(line.lstrip())
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j]
                        next_stripped = next_line.strip()
                        current_indent = len(next_line) - len(next_line.lstrip())

                        if next_stripped == "":
                            function_block.append(next_line)
                        elif current_indent <= indent_level and next_stripped:
                            break
                        else:
                            function_block.append(next_line)

                        j += 1
                        if len("\n".join(function_block)) > max_chars:
                            break

                function_text = "\n".join(function_block)

                # Check if the function has complexity
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

                # Accept functions that are at least 2 lines long even if they lack complexity
                if has_complexity or line_count >= 2:
                    results.append((snippet, function_text))
                    print(
                        f"Found useful function: {function_name}, length: {line_count} lines"
                    )

                # Skip to after this function
                i = j
                continue

        i += 1

    return results


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


def analyze_code_style(code_text):
    """Analyze the code style based on various metrics."""
    if not code_text:
        return {}

    # Count lines with comments (excluding empty comment lines)
    comment_lines = sum(
        1
        for line in code_text.split("\n")
        if line.strip().startswith(("#", "//")) and len(line.strip()) > 2
    )

    # Count total non-empty lines
    total_lines = sum(1 for line in code_text.split("\n") if line.strip())

    # Check for docstrings
    has_docstring = '"""' in code_text or "'''" in code_text

    # Check for descriptive variable names (more than 1 character)
    variable_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*="
    variables = re.findall(variable_pattern, code_text)
    descriptive_vars = sum(1 for var in variables if len(var) > 1)
    single_char_vars = sum(1 for var in variables if len(var) == 1)

    # Calculate indentation consistency
    indentation_pattern = r"^(\s*)[^\s]"
    indentations = [
        len(re.match(indentation_pattern, line).group(1))
        for line in code_text.split("\n")
        if line.strip() and re.match(indentation_pattern, line)
    ]

    indent_sizes = set(indentations) - {0}
    consistent_indentation = len(indent_sizes) <= 2

    # Line length (80 chars is standard in many styles)
    long_lines = sum(1 for line in code_text.split("\n") if len(line) > 100)

    return {
        "comment_ratio": round(comment_lines / total_lines, 2) if total_lines else 0,
        "has_docstring": has_docstring,
        "descriptive_vars": descriptive_vars,
        "single_char_vars": single_char_vars,
        "consistent_indentation": consistent_indentation,
        "long_lines_ratio": round(long_lines / total_lines, 2) if total_lines else 0,
        "total_lines": total_lines,
    }


def analyze_function_detail(detail):
    """
    Analyze an individual function using the LLM.
    The prompt asks for code originality, style, technical depth, and developer insights.
    """
    function_name = detail.get("function_name", "Unnamed Function")
    code_snippet = detail.get("code_snippet", "No code available.")
    language = detail.get("language", "python")
    prompt = f"""
    Please analyze the following function:

    Function Name: {function_name}
    Language: {language}

    Code Snippet:
    {code_snippet}

    Provide a concise analysis covering:
    1. Code originality – is this implementation unique or a standard pattern?
    2. Coding style and readability – quality of comments, variable naming, and structure.
    3. Technical depth – use of language features, complexity, and efficiency.
    4. Developer insight – what does this function reveal about the developer’s skills and approach?
    """
    prompt = auto_escape_prompt(prompt)
    try:
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | llm
        response = chain.invoke({})
        return response.content
    except Exception as e:
        return f"Error analyzing function {function_name}: {str(e)}"


def generate_code_fingerprint(selected_repo_names):
    """Memory-efficient version of the code fingerprint generator with inline per-function analysis."""
    owner = st.session_state.github_username
    repos = st.session_state.repos
    if not repos:
        return "No repositories loaded.", []

    # Limit selection size
    selected_repo_names = selected_repo_names[:3]  # Limit to 3 repos max
    selected_repos = [repo for repo in repos if repo["name"] in selected_repo_names]

    snippet_details = []
    search_results = []
    unique_repo_code = {}

    max_functions_per_repo = 2

    for repo in selected_repos:
        repo_name = repo["name"]
        with st.spinner(f"Analyzing {repo_name}..."):
            results = fetch_code_files(
                owner,
                repo_name,
                max_depth=2,  # Reduce max depth
                max_functions=max_functions_per_repo,
            )

            if not results:
                st.warning(f"No functions found in {repo_name}.")
                continue

            seen_snippets = set()
            for code_text, language, function_data in results:
                snippet, function_block = function_data
                if snippet in seen_snippets:
                    continue
                seen_snippets.add(snippet)

                if repo_name not in unique_repo_code:
                    unique_repo_code[repo_name] = code_text

                try:
                    matches = search_snippet(snippet, language)
                except Exception as e:
                    print(f"Error searching snippet: {e}")
                    matches = 0

                style_metrics = analyze_code_style(function_block)
                function_name = (
                    snippet.split("(")[0].strip() if "(" in snippet else snippet
                )
                snippet_details.append(
                    {
                        "function_name": function_name,
                        "repo_name": repo_name,
                        "matches": matches,
                        "code_snippet": function_block[:500],  # Limit size
                        "language": language,
                        "style_metrics": style_metrics,
                    }
                )
                search_results.append((repo_name, snippet, matches, language))

                import gc

                gc.collect()

    # Generate UI summaries for display (as before)
    if search_results:
        search_summary_items = []
        for repo, snippet, matches, lang in search_results:
            function_name = snippet.split("(")[0].strip() if "(" in snippet else snippet
            originality = (
                "Highly original"
                if matches < 5
                else ("Common" if matches < 20 else "Very common")
            )
            search_summary_items.append(
                f"- **{repo}**: Function `{function_name}` ({lang}) - {matches} matches - {originality}"
            )
        search_summary = "\n".join(search_summary_items)
    else:
        search_summary = "No snippets found."

    style_summary = "## Code Style Analysis:\n\n"
    for repo_name, code_text in unique_repo_code.items():
        metrics = analyze_code_style(code_text)
        style_summary += f"### {repo_name}:\n"
        style_summary += f"- Documentation: {'Has docstrings' if metrics.get('has_docstring') else 'No docstrings'}\n"
        style_summary += f"- Comment ratio: {metrics.get('comment_ratio', 0):.2f}\n"
        style_summary += f"- Lines of code: {metrics.get('total_lines', 0)}\n\n"

    # Now, perform inline analysis for each extracted function
    function_analysis_results = []
    for detail in snippet_details:
        analysis = analyze_function_detail(detail)
        detail["analysis"] = analysis  # Store the analysis with the snippet details
        function_analysis_results.append(analysis)

    # Combine individual analyses into a final summary for display.
    combined_analysis = "### Function Analysis:\n\n"
    for i, detail in enumerate(snippet_details):
        combined_analysis += (
            f"#### {i+1}. {detail['function_name']} (from {detail['repo_name']})\n"
        )
        combined_analysis += detail.get("analysis", "No analysis available.") + "\n\n"

    # Return the combined analysis and snippet details for further display.
    return combined_analysis, snippet_details


# --- Streamlit UI ---
st.title("RepoNarrator")
st.text_input("GitHub Username", value="ldw129", key="github_username")

# Create three columns with better proportions for button alignment
col1, col2, col3 = st.columns([1, 1, 1])

# Put Load Repos in first column (left)
with col1:
    if st.button("Load Repos", use_container_width=True):
        owner = st.session_state.github_username
        with st.spinner(f"Fetching repositories for {owner}..."):
            st.session_state.repos = fetch_user_repos(owner)
        if st.session_state.repos:
            st.success(f"Found {len(st.session_state.repos)} repositories")

# Leave middle column empty to create space
with col2:
    pass

# Put Profile Summary in third column (right)
with col3:
    if st.button("Create Portfolio Summary", use_container_width=True):
        summarize_all_repos()

# Rest of your code remains the same
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
        "Select up to 5 repositories to analyze:",  # Increased from 3 to 5
        options=repo_names,
        default=default_selection,
        max_selections=5,  # Increased from 3 to 5
        key="selected_repos",
    )
    if len(selected_repos) == 5:  # Updated from 3 to 5
        st.info(
            "You've selected the maximum of 5 repositories. Deselect one to choose another."
        )
    if st.button("Generate Code Fingerprint"):
        with st.spinner("Analyzing code usage..."):
            fingerprint_summary, snippet_details = generate_code_fingerprint(
                selected_repos
            )
        st.write("### Code Usage Fingerprint Summary")
        st.markdown(fingerprint_summary)  # Using markdown to render formatting

        st.subheader("Code Snippets Searched from GitHub Database")
        if snippet_details:
            for i, detail in enumerate(snippet_details):
                func_name = detail.get("function_name", "Unnamed Function")
                repo = detail.get("repo_name", "Unknown Repo")
                matches = detail.get("matches", "N/A")
                purpose = detail.get("purpose", "No description provided.")
                usage_info = detail.get("usage", "No usage info provided.")
                notes = detail.get("notes", "No additional notes available.")
                code_snippet = detail.get("code_snippet", "No code available.")
                lang = detail.get("language", "python")

                # Add style metrics visual representation
                style_metrics = detail.get("style_metrics", {})

                # Create a more visually appealing display for each function
                st.markdown(f"#### {i+1}. **{func_name}** Function")
                st.markdown(f"**From:** {repo} | **Language:** {lang.capitalize()}")

                # Display a more visual metric for similarity
                if matches == 0:
                    st.success(
                        f"🔰 **Unique:** No similar code found in other repositories"
                    )
                elif matches < 5:
                    st.success(
                        f"✅ **Mostly Original:** Found in only {matches} repositories"
                    )
                elif matches < 20:
                    st.warning(f"⚠️ **Common Pattern:** Found in {matches} repositories")
                else:
                    st.error(f"🔄 **Very Common:** Found in {matches} repositories")

                # Code style metrics as a horizontal layout
                if style_metrics:
                    cols = st.columns(3)
                    with cols[0]:
                        doc_status = (
                            "✅" if style_metrics.get("has_docstring", False) else "❌"
                        )
                        st.markdown(f"**Documentation:** {doc_status}")

                    with cols[1]:
                        comment_ratio = style_metrics.get("comment_ratio", 0)
                        comment_icon = (
                            "✅"
                            if comment_ratio > 0.1
                            else ("⚠️" if comment_ratio > 0 else "❌")
                        )
                        st.markdown(
                            f"**Comments:** {comment_icon} ({comment_ratio:.0%})"
                        )

                    with cols[2]:
                        desc_vars = style_metrics.get("descriptive_vars", 0)
                        single_vars = style_metrics.get("single_char_vars", 0)
                        naming_icon = (
                            "✅"
                            if desc_vars > single_vars * 2
                            else ("⚠️" if desc_vars > single_vars else "❌")
                        )
                        st.markdown(f"**Variable Names:** {naming_icon}")

                # Code snippet in an expander
                with st.expander("Show full code snippet"):
                    st.code(code_snippet, language=lang)

                st.markdown("---")  # Divider between functions
        else:
            st.write("No code snippets available for display.")
else:
    st.write("Please load repositories first to analyze code.")
