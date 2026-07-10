import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(
    page_title="OpenSource Agentic AI",
    page_icon="🤖"
)

st.title("🤖 OpenSource Agentic AI System")

groq_api_key = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0,
    streaming=True,
)

from langchain.tools import Tool
import wikipedia

wikipedia.set_lang("en")

def wikipedia_search(query):
    try:
        page = wikipedia.page(query, auto_suggest=True)
        return page.summary[:500]
    except Exception as e:
        return f"Wikipedia Error: {str(e)}"

wiki = Tool(
    name="Wikipedia",
    func=wikipedia_search,
    description="Search Wikipedia for general knowledge."
)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=300,
)

from langchain.tools import Tool

def arxiv_search(query):
    try:
        return ArxivQueryRun(api_wrapper=arxiv_wrapper).run(query)
    except Exception as e:
        return f"ArXiv Error: {str(e)}"

arxiv = Tool(
    name="Arxiv",
    func=arxiv_search,
    description="Search research papers from ArXiv."
)
search = DuckDuckGoSearchRun(name="Search")

tools = [search, wiki, arxiv]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I am your AI Assistant. Ask me anything."
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask anything..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):

        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False
        )

        try:
            response = agent.run(
                prompt,
                callbacks=[st_cb]
            )
        except Exception as e:
            response = f"Error: {e}"

        st.write(response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )
