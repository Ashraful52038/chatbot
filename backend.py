from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import MemorySaver


from langgraph.graph import add_messages

class chatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # ✅ model দিতে হবে!
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

def chat_node(state: chatState):
    #take user query from state
    messages = state['messages']
    #send to LLM
    response = llm.invoke(messages)
    return {'messages': [response]}

checkpointer = MemorySaver()

graph = StateGraph(chatState)

#add nodes
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)


config = {"configurable": {"thread_id": "1"}}

initial_state = {
    'messages': [HumanMessage(content="Hello, how are you?")]
}

chatbot.invoke(initial_state, config=config)['messages'][-1].content

chatbot.get_state(config=config)