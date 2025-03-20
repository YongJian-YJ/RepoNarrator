# conda activate chat-with-website
# streamlit run History.py

import streamlit as st
import os
import json

st.set_page_config(page_title="Conversation History", layout="wide")
st.title("Conversation History")

HISTORY_FILE = "data/history.json"


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


def update_history_tag(conv_timestamp, conv_question, new_tag):
    """Update the tag of a conversation identified by timestamp and question."""
    updated = False
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        for conv in history:
            if (
                conv.get("timestamp") == conv_timestamp
                and conv.get("question") == conv_question
            ):
                conv["tag"] = new_tag
                updated = True
        if updated:
            save_history(history)
    return updated


def delete_history_record(conv_timestamp, conv_question):
    """Delete a conversation identified by timestamp and question."""
    deleted = False
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        new_history = [
            conv
            for conv in history
            if not (
                conv.get("timestamp") == conv_timestamp
                and conv.get("question") == conv_question
            )
        ]
        if len(new_history) < len(history):
            save_history(new_history)
            deleted = True
    return deleted


def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)


# Load conversation history
history = load_history()

# Build a set of all tags present in the history
all_tags = set()
for conv in history:
    tag = conv.get("tag", "").strip()
    if tag:
        all_tags.add(tag)
all_tags = sorted(list(all_tags))
all_tags.insert(0, "All")  # "All" option to show all conversations

# Sidebar: History Options and Tag Filter
st.sidebar.header("History Options")
selected_tag = st.sidebar.selectbox("Filter by tag", options=all_tags)

if st.sidebar.button("Clear History"):
    clear_history()
    st.sidebar.success("History cleared!")
    try:
        st.experimental_rerun()
    except AttributeError:
        pass

st.write("Below is your conversation history:")

# Filter history by selected tag if applicable
if selected_tag != "All":
    filtered_history = [
        conv for conv in history if conv.get("tag", "").strip() == selected_tag
    ]
else:
    filtered_history = history

if filtered_history:
    # Display conversations sorted by timestamp descending (latest first)
    sorted_history = sorted(
        filtered_history, key=lambda x: x.get("timestamp", ""), reverse=True
    )
    for conv in sorted_history:
        timestamp = conv.get("timestamp", "Unknown time")
        question = conv.get("question", "")
        answer = conv.get("answer", "")
        current_tag = conv.get("tag", "")

        with st.expander(f"{timestamp} - Q: {question}", expanded=False):
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**Answer:** {answer}")
            # Text input to update the tag
            new_tag = st.text_input(
                "Tag this conversation:",
                value=current_tag,
                key=f"tag_{timestamp}_{question}",
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Tag", key=f"save_{timestamp}_{question}"):
                    if update_history_tag(timestamp, question, new_tag):
                        st.success("Tag updated!")
                    else:
                        st.error("Could not update tag.")
            with col2:
                if st.button("Delete History", key=f"delete_{timestamp}_{question}"):
                    if delete_history_record(timestamp, question):
                        st.success("Conversation deleted!")
                        try:
                            st.experimental_rerun()
                        except AttributeError:
                            pass
                    else:
                        st.error("Failed to delete conversation.")
else:
    st.info("No conversation history available for the selected tag.")
