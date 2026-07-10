import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Groq Chatbot", page_icon="🤖")

st.title("🤖Chat with Groq")

# Sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hi! I'm a chatbot powered by Groq. How can I help you?")
    ]

# Display chat history
for message in st.session_state.messages:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.write(message.content)

# User Input
prompt = st.chat_input("Ask me anything...")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.write(prompt)

    if not api_key:
        st.error("Please enter your Groq API Key.")
        st.stop()

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="Llama3-8b-8192",
        streaming=True,
    )

    with st.chat_message("assistant"):
        response = llm.invoke(st.session_state.messages)
        st.write(response.content)

    st.session_state.messages.append(
        AIMessage(content=response.content)
    )
