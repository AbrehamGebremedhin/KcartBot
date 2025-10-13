"""Streamlit chat client for the KcartBot FastAPI service."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

API_DEFAULT = os.getenv("KCARTBOT_API_URL", "http://localhost:8000/api/v1/chat")
SESSION_ID_KEY = "kcartbot_session_id"
HISTORY_KEY = "kcartbot_chat_history"
TRACE_KEY = "kcartbot_trace_log"


@dataclass
class ChatTurn:
    """Represents a single assistant reply with optional metadata."""

    user: str
    bot: Optional[str]
    raw_payload: Dict[str, Any]


def _initialise_state() -> None:
    """Seed Streamlit session state with defaults if missing."""
    if SESSION_ID_KEY not in st.session_state:
        st.session_state[SESSION_ID_KEY] = str(uuid.uuid4())
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []
    if TRACE_KEY not in st.session_state:
        st.session_state[TRACE_KEY] = []


def _post_chat_message(api_url: str, session_id: str, message: str) -> Dict[str, Any]:
    """Send a chat request to the FastAPI backend and return JSON."""
    payload = {
        "session_id": session_id,
        "message": message,
    }
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise RuntimeError(f"API error: {exc.response.status_code if exc.response else 'unknown'} {detail}") from exc
    except requests.RequestException as exc:  # network or timeout
        raise RuntimeError(f"Request failed: {exc}") from exc


def _reset_session(reseed_session_id: bool) -> None:
    """Clear history and optionally issue a new session identifier."""
    st.session_state[HISTORY_KEY] = []
    st.session_state[TRACE_KEY] = []
    if reseed_session_id:
        st.session_state[SESSION_ID_KEY] = str(uuid.uuid4())


def _render_sidebar() -> str:
    """Render sidebar controls and return the configured API URL."""
    st.sidebar.header("Connection")
    api_url = st.sidebar.text_input("Chat endpoint", value=API_DEFAULT, help="POST endpoint for /api/v1/chat")

    st.sidebar.header("Session")
    st.sidebar.caption("Keep the same session id while continuing a conversation.")
    st.sidebar.write(st.session_state[SESSION_ID_KEY])
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("New ID", help="Generate a fresh session identifier"):
            st.session_state[SESSION_ID_KEY] = str(uuid.uuid4())
    with col2:
        if st.button("Reset chat", help="Clear history and keep session id", key="reset_history"):
            _reset_session(reseed_session_id=False)

    if st.sidebar.button("Reset all", help="Clear history and issue a new session id", key="reset_all"):
        _reset_session(reseed_session_id=True)

    st.sidebar.header("Debug")
    st.sidebar.checkbox("Show tool traces", key="show_traces", value=False)

    return api_url


def _render_history() -> None:
    """Display the running chat transcript."""
    for turn in st.session_state[HISTORY_KEY]:
        with st.chat_message("user"):
            st.markdown(turn.user)
        with st.chat_message("assistant"):
            st.markdown(turn.bot or "(No response received)")


def _render_traces() -> None:
    """Optionally show raw trace payloads for each turn."""
    if not st.session_state.get("show_traces"):
        return
    st.divider()
    st.subheader("Tool traces")
    for idx, trace in enumerate(st.session_state[TRACE_KEY], start=1):
        with st.expander(f"Turn {idx}"):
            st.json(trace or {})


def main() -> None:
    """Application entry point for Streamlit."""
    st.set_page_config(page_title="KcartBot Client", page_icon="🤖", layout="wide")
    _initialise_state()

    api_url = _render_sidebar()

    st.title("KcartBot Conversational Console")
    st.caption("Gemini-powered assistant backed by Milvus RAG and PostgreSQL operations.")

    _render_history()

    if prompt := st.chat_input("Ask a question or issue a command..."):
        st.session_state[HISTORY_KEY].append(ChatTurn(user=prompt, bot=None, raw_payload={}))
        with st.spinner("Contacting KcartBot..."):
            try:
                payload = _post_chat_message(api_url, st.session_state[SESSION_ID_KEY], prompt)
            except RuntimeError as err:
                st.error(str(err))
                st.session_state[HISTORY_KEY][-1].bot = "(Request failed)"
            else:
                st.session_state[HISTORY_KEY][-1].bot = payload.get("response", "")
                st.session_state[HISTORY_KEY][-1].raw_payload = payload
                st.session_state[TRACE_KEY].append(payload.get("trace"))

    st.rerun()

    _render_traces()


if __name__ == "__main__":
    main()
