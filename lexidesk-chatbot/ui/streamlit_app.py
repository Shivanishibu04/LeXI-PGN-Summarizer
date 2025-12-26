# ui/streamlit_app.py
"""
Streamlit chat frontend (simple).
Run: streamlit run ui/streamlit_app.py
"""
import streamlit as st
import requests

API_URL = st.text_input("API URL", value="http://localhost:8000/qa")

st.title("LeXiDesk Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

q = st.text_input("Ask question about uploaded docs:")

if st.button("Ask") and q.strip():
    payload = {"question": q, "top_k": 5}
    try:
        r = requests.post(API_URL, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        st.session_state.history.append((q, data.get("answer", ""), data.get("sources", [])))
    except Exception as e:
        st.error(str(e))

for q,a,sources in reversed(st.session_state.history):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
    st.markdown("**Sources:**")
    for src in sources:
        st.markdown(f"- {src['doc_id']} (p{src['page']})")
    st.markdown("---")
