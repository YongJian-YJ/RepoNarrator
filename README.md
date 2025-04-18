# RepoNarrator
<h2>📺 Video Demo</h2>
<a href="https://www.youtube.com/watch?v=WPnzze30Deo" target="_blank">
  <img src="https://img.youtube.com/vi/WPnzze30Deo/0.jpg" alt="Watch the video" width="600">
</a>

# 🧠 RepoNarrator

**RepoNarrator** is a developer profiling tool that summarizes GitHub repositories to generate insightful narratives and evaluate coding behavior. It uses Large Language Models (LLMs) to interpret README files, assess function-level code quality, and analyze code originality — giving recruiters and reviewers a deeper understanding of the person behind the code.

---

## 🚀 Features

- 📄 **README Summarization**: Aggregates and summarizes repository READMEs to generate a personal profile.
- 👨‍💻 **Developer Profiling**: Uses an LLM to infer background, skills, project themes, and passions.
- 🔍 **Code Analysis & Originality Check**:
  - Recursive function extraction from selected repositories.
  - LLM-based evaluation of coding practices and code style.
  - GitHub Search API integration to assess function originality across GitHub.
  - Visualized insights on documentation, variable naming, and inline comments.
- 🧠 **Chain-of-Thought Prompting**: Ensures structured, step-by-step reasoning in LLM responses.
- 🌐 **GitHub API Integration**: Fetches repository metadata, README files, and code content.

---

## 🧪 Code Analysis & Originality Check

In addition to summarizing READMEs, RepoNarrator allows users to **select up to five repositories** for in-depth code evaluation. This includes:

### 🧬 Function-Level Analysis

- Extracts all functions recursively from each selected repository.
- Sends code snippets to an LLM that evaluates:
  - 📚 **Documentation**: Presence of clear docstrings.
  - 💬 **Comments**: Use of inline code explanations.
  - 🔤 **Variable Naming**: Readability and descriptiveness of variable names.
  - ✨ **Coding Style**: Consistency and cleanliness of code.
  - 🧠 **Technical Depth**: Complexity and abstraction in implementation.

### 🔍 Code Originality

- Uses the GitHub Search API to search for exact matches of each function.
- Reports how many public repositories contain that exact function, helping assess:
  - ❗ Plagiarism risk
  - ✅ Code uniqueness
  - 🧑‍🎓 Reuse of common code snippets
