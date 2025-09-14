import streamlit as st
from src.generation.rag_chain import RAGChain

st.title("Hello, Streamlit!")

if st.session_state.get("rag_chain") is None:
    st.session_state.rag_chain = RAGChain(persist_directory="./data/chromadb",
                                      collection_name="aiaa_docs",)

def response_generator(query):
    response = st.session_state.rag_chain.ask_hybrid(query)
    yield response.answer

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Say something"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})



