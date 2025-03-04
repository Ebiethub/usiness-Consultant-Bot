import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
import os

# Set up environment variables for Groq API
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# st.secrets["GROQ_API_KEY"]

# Initialize LLM with Groq API
llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-specdec",
        temperature=0.7,
        max_tokens=4000
    )

# Streamlit UI Setup
st.set_page_config(page_title="AI Business Consultant Bot", layout="wide")
st.title("🚀 AI Business Consultant Bot")
st.subheader("Get expert business advice instantly!")

# Initialize conversation memory
memory = ConversationBufferMemory()

# Load chat history from session state if available
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a professional AI business consultant. Provide expert-level business guidance, financial insights, and strategic planning advice.")
    ]

# Display previous messages
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# User input field
user_input = st.chat_input("Ask your business question...")

if user_input:
    # Append user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)
    
    # Generate AI response
    response = llm(messages=st.session_state.messages)
    
    # Append AI response to chat history
    st.session_state.messages.append(AIMessage(content=response.content))
    
    # Display AI response
    st.chat_message("assistant").write(response.content)

# Add a footer
st.markdown("\n💡 *Powered by Groq API & LangChain* \n")
