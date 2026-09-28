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
    messages: List[BaseMessage]
    mode: Literal["route", "adjust"]

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

def _run_recipe_node(state: GraphState, meal: str) -> GraphState:
    """Shared logic for both recipe nodes: keeps history so follow-up adjustments have context."""
    system_prompt = SystemMessage(content=(
        f"You are a helpful assistant tasked with generating and adjusting {meal} "
        "recipes. Use the full conversation so far, including any previous recipe "
        "you gave, to produce an updated or new recipe that matches the user's "
        "latest request."
    ))
 
    history = state.get("messages", [])
    updated_history = history + [HumanMessage(content=state["input"])]
 
    response = model.invoke([system_prompt] + updated_history)
    output = response.content.strip()
 
    updated_history = updated_history + [AIMessage(content=output)]
 
    return {
        **state,
        "output": f"{meal.capitalize()} Recipe: {output}",
        "messages": updated_history,
        "destination": f"{meal}_node",
    }

def breakfast_node(state: GraphState) -> GraphState:
    return _run_recipe_node(state, "breakfast")
 
def lunch_node(state: GraphState) -> GraphState:
    return _run_recipe_node(state, "lunch") 

def route_decision(state: GraphState) -> Literal["breakfast_node", "lunch_node"]:
    return state["destination"]

def entry_route(state: GraphState) -> Literal["router", "breakfast_node", "lunch_node"]:
    """On a fresh request, go through the router. On an adjustment, go straight
    back to whichever node handled the recipe last time."""
    if state.get("mode") == "adjust" and state.get("destination"):
        return state["destination"]
    return "router"

def create_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("router", logic_node)
    workflow.add_node("breakfast_node", breakfast_node)
    workflow.add_node("lunch_node", lunch_node)

    #workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        START,
        entry_route,
        {
            "router": "router",
            "breakfast_node": "breakfast_node",
            "lunch_node": "lunch_node",
        },
    )

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

    state: GraphState = {
        "input": "",
        "destination": "",
        "output": "",
        "messages": [],
        "mode": "route",
    }

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        state["input"] = user_input

        try:
            output = app.invoke(state)
            print(f"System: {output['output']}")
            state = output  # Update state for next iteration

            follow_up= input("Do you want to adjust the recipe? (yes/no): ").strip().lower()
            state["mode"] = "adjust" if follow_up.startswith("y") else "route"

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()