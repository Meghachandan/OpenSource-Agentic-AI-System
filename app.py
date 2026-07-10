import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="Groq Chatbot",
    page_icon="🤖",
)

st.title("🤖 Chat with Groq")

st.sidebar.header("Settings")

# Read API Key from Streamlit Secrets
api_key = st.secrets["GROQ_API_KEY"]

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hi! I'm a chatbot powered by Groq. How can I help you?")
    ]

# Display Chat History
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat Input
prompt = st.chat_input("Ask me anything...")

if prompt:

    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0.7,
    )

    with st.chat_message("assistant"):
        response = llm.invoke(st.session_state.messages)
        st.write(response.content)

    st.session_state.messages.append(
        AIMessage(content=response.content)
    )
