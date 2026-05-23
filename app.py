import streamlit as st
from rag import ask_question

st.set_page_config(page_title="Document Q&A RAG")

st.title("📚 Document Q&A with Citations")

query = st.text_input("Ask a question")

if st.button("Ask"):

    result = ask_question(query)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Citations")

    for cite in result["citations"]:
        st.write(f"Source: {cite['source']}")
        st.write(f"Page: {cite['page']}")
        st.code(cite['snippet'])