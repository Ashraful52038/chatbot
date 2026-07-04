import streamlit as st
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import add_messages
import uuid

# ==================== BACKEND SETUP ====================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

checkpointer = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

# ==================== FRONTEND UI ====================

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 AI Chat Assistant")
st.caption("Powered by Groq Llama 3.3")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar with controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    
    st.divider()
    
    # Display thread info
    st.info(f"🆔 Thread ID: {st.session_state.thread_id[:8]}...")
    
    # Temperature control
    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more focused"
    )
    
    # Model info
    st.divider()
    st.caption("🤖 Model: Llama 3.3 70B")
    st.caption("💾 Memory: Session-based")

# Main chat interface
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input at the bottom
if prompt := st.chat_input("Type your message here..."):
    # Add user message to session
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare the state with conversation history
                # Convert session messages to LangChain format
                langchain_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))
                
                # If no history, initialize with current message
                if not langchain_messages:
                    langchain_messages = [HumanMessage(content=prompt)]
                
                # Invoke the chatbot
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                state = {'messages': langchain_messages}
                
                response = chatbot.invoke(state, config=config)
                bot_response = response['messages'][-1].content
                
                # Display bot response
                st.markdown(bot_response)
                
                # Add bot response to session
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Auto-scroll to bottom (using JavaScript)
st.markdown("""
    <script>
    function scrollToBottom() {
        const chatContainer = document.querySelector('.stChatMessage');
        if (chatContainer) {
            chatContainer.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
    }
    setTimeout(scrollToBottom, 100);
    </script>
""", unsafe_allow_html=True)

# Footer
st.divider()
st.caption("💡 Type your message and press Enter to chat with the AI assistant.")