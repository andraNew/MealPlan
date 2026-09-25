import operator
from typing import Annotated, List, TypedDict, Sequence

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END

model = ChatAnthropic(
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-haiku-4-5-20251001")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def logic_node(state: AgentState) -> AgentState:
    """
    A simple logic node that appends a new message to the state.
    """
    last_message = state["messages"][-1]
    user_text = last_message.content 

    if user_text == "reply":
        response_text = "well done"
    else:
        response_text = f"You said: '{last_message.content}'"
    
    return {"messages": [AIMessage(content=response_text)]}

def llm_call(state: MessagesState):
    """LLM decides what is the intention of the user and what is the next step in the workflow."""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with deciding if the user wants a lunch receipe or breakfast receipe."
                        
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("logic_node", logic_node)
    workflow.set_entry_point("logic_node")
    workflow.add_edge("logic_node", END)

    return workflow.compile()

def main():
    print("--- Dimplr Logic CLI (Type 'exit' to quit) ---")
    app = create_graph()

    # Initialize state
    state = {"messages": []}

    while True:
        user_input = input("User: ")

        # Exit condition
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        if not user_input.strip():
            print("Please enter a valid message.")
            continue

        # Add user message to state
        state["messages"] = [HumanMessage(content=user_input)]

        try:
            output = app.invoke(state)
            state["messages"] = output["messages"]
            last_message = output["messages"][-1]
            print(f"System: {last_message.content}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()