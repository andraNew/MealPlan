import operator
from typing import Annotated, List, TypedDict, Sequence, Literal
from langchain_anthropic import ChatAnthropic
import os
import dotenv

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

dotenv.load_dotenv()
model = ChatAnthropic(
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-haiku-4-5-20251001")

class GraphState(TypedDict):
    input : str
    destination: str
    output: str

def logic_node(state: GraphState) -> GraphState:
    """
    Reads the input and decides whether it's a request for a lunch or breakfast receipe.
    """
    system_prompt = SystemMessage(content=(

        "You are a touting assistant for a food app. Read the user's input and understand \
        if the message is about breakfast or lunch receipe. If the user is asking for a \
        breakfast receipe, respond with 'breakfast', otherwise respond with 'lunch'. \
        If the message doesn't clearly fit either, defaut to lunch \
        Respond with only one word: 'breakfast' or 'lunch'."
    )) 
    human_message = HumanMessage(content=state["input"])

    response = model.invoke([system_prompt, human_message])
    decision = response.content.strip().lower()

    if 'breakfast' in decision:
        destination = 'breakfast'
    elif 'lunch' in decision:
        destination = 'lunch'
    else:
        destination = 'lunch'
    
    return {**state, "destination": destination}
def breakfast_node(state: GraphState) -> GraphState:
    """
    Generates a breakfast receipe based on the user's input.
    """
    system_prompt = SystemMessage(content=(
        "You are a helpful assistant tasked with generating a breakfast receipe. \
        Use the user's input to create a suitable breakfast receipe."
    ))
    human_message = HumanMessage(content=state["input"])

    response = model.invoke([system_prompt, human_message])
    output = response.content.strip()

    return {**state, "output": f"Breakfast Recipe: {output}"}

def lunch_node(state: GraphState) -> GraphState:
    """
    Generates a lunch receipe based on the user's input.
    """
    system_prompt = SystemMessage(content=(
        "You are a helpful assistant tasked with generating a lunch receipe. \
        Use the user's input to create a suitable lunch receipe."
    ))
    human_message = HumanMessage(content=state["input"])

    response = model.invoke([system_prompt, human_message])
    output = response.content.strip()

    return {**state, "output": f"Lunch Recipe: {output}"}  

def route_decision(state: GraphState) -> Literal["breakfast_node", "lunch_node"]:
    return state["destination"]


def create_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("router", logic_node)
    workflow.add_node("breakfast_node", breakfast_node)
    workflow.add_node("lunch_node", lunch_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "breakfast": "breakfast_node",
            "lunch": "lunch_node",
        },
    )
    
    workflow.add_edge("breakfast_node", END)
    workflow.add_edge("lunch_node", END)

    return workflow.compile()

def main():
    print("--- Dimplr Logic CLI (Type 'exit' to quit) ---")
    app = create_graph()

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        state = {"input": user_input, "destination": "", "output": ""}

        try:
            output = app.invoke(state)
            print(f"System: {output['output']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()