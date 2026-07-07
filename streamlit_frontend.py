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
from datetime import datetime

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

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for multiple threads
if "threads" not in st.session_state:
    # Format: {thread_id: {"name": "Chat 1", "messages": [], "created_at": timestamp}}
    st.session_state.threads = {}
    
if "current_thread_id" not in st.session_state:
    # Create first thread
    first_thread_id = str(uuid.uuid4())
    st.session_state.threads[first_thread_id] = {
        "name": "New Chat",
        "messages": [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    st.session_state.current_thread_id = first_thread_id

# ==================== SIDEBAR ====================

with st.sidebar:
    st.title("💬 Chat History")
    
    # New Chat button
    if st.button("➕ New Chat", use_container_width=True):
        new_thread_id = str(uuid.uuid4())
        st.session_state.threads[new_thread_id] = {
            "name": f"Chat {len(st.session_state.threads) + 1}",
            "messages": [],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        st.session_state.current_thread_id = new_thread_id
        st.rerun()
    
    st.divider()
    
    # Display all threads
    for thread_id, thread_data in st.session_state.threads.items():
        # Highlight current thread
        is_current = thread_id == st.session_state.current_thread_id
        button_label = f"📌 {thread_data['name']}" if is_current else f"💬 {thread_data['name']}"
        
        # Show message count and timestamp
        msg_count = len(thread_data['messages'])
        col1, col2 = st.columns([4, 1])
        
        with col1:
            if st.button(
                button_label,
                key=f"thread_{thread_id}",
                use_container_width=True,
                type="primary" if is_current else "secondary"
            ):
                if not is_current:
                    st.session_state.current_thread_id = thread_id
                    st.rerun()
        
        with col2:
            # Delete thread button (except if it's the only one)
            if len(st.session_state.threads) > 1:
                if st.button("🗑️", key=f"delete_{thread_id}"):
                    # Don't delete if it's the current thread
                    if thread_id == st.session_state.current_thread_id:
                        # Switch to first available thread
                        other_threads = [tid for tid in st.session_state.threads.keys() if tid != thread_id]
                        if other_threads:
                            st.session_state.current_thread_id = other_threads[0]
                    del st.session_state.threads[thread_id]
                    st.rerun()
        
        # Show small info
        st.caption(f"📝 {msg_count} messages • {thread_data['created_at']}")
        st.divider()
    
    # Clear all chats option
    if len(st.session_state.threads) > 0:
        if st.button("🗑️ Clear All Chats", use_container_width=True, type="secondary"):
            # Keep only one new thread
            new_thread_id = str(uuid.uuid4())
            st.session_state.threads = {
                new_thread_id: {
                    "name": "New Chat",
                    "messages": [],
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
            }
            st.session_state.current_thread_id = new_thread_id
            st.rerun()

# ==================== MAIN CHAT AREA ====================

# Get current thread messages
current_thread = st.session_state.threads[st.session_state.current_thread_id]
messages = current_thread["messages"]

# Title with thread name
st.title(f"🤖 {current_thread['name']}")
st.caption(f"Thread ID: {st.session_state.current_thread_id[:8]}...")

# Display chat messages
chat_container = st.container()

with chat_container:
    if not messages:
        st.info("💡 Start a new conversation! Type your message below.")
    else:
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# Chat input at the bottom
if prompt := st.chat_input("Type your message here..."):
    # Add user message to current thread
    messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare the state with conversation history
                langchain_messages = []
                for msg in messages:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))
                
                # Invoke the chatbot
                config = {"configurable": {"thread_id": st.session_state.current_thread_id}}
                state = {'messages': langchain_messages}
                
                response = chatbot.invoke(state, config=config)
                bot_response = response['messages'][-1].content
                
                # Display bot response
                st.markdown(bot_response)
                
                # Add bot response to thread
                messages.append({"role": "assistant", "content": bot_response})
                
                # Auto-update thread name if it's still "New Chat" and has messages
                if current_thread["name"] == "New Chat" and len(messages) >= 2:
                    # Use first few words of first user message as title
                    first_user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
                    if first_user_msg:
                        words = first_user_msg.split()[:5]
                        current_thread["name"] = " ".join(words) + "..."
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                messages.append({"role": "assistant", "content": error_msg})
    
    # Force rerun to update UI
    st.rerun()

# Footer
st.divider()
st.caption("💡 Type your message and press Enter to chat with the AI assistant.")