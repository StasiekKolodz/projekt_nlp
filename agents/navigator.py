from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from tools.drone_tools import takeoff, fly_to, land
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage

class NavigatorAgent:
    def __init__(self):
        self.tools = [
            Tool(
                name="Takeoff",
                func=takeoff,
                description=(
                    "takeoff and fly up to a given altitude.\n"
                    "Arguments:\n"
                    "  • altitude (float): target height in meters, must be > 0."
                )
            ),
            Tool(
                name="FlyTo",
                func=fly_to,
                description=(
                    "Move the drone relative to its current position.\n"
                    "Arguments (all floats, in meters):\n"
                    "  • north: positive moves north, negative moves south\n"
                    "  • east:  positive moves east,  negative moves west\n"
                    "  • down:  positive moves downward, negative moves upward"
                )
            ),
            Tool(
                name="Land",
                func=land,
                description=(
                    "Descend and land safely—no arguments required."
                )
            ),
        ]

        self.navigator = create_react_agent(
            ChatOpenAI(model="gpt-4"),
            self.tools,
        )


    def run(self, task):
        # Accepts either dict or string, always wraps as a message
        if isinstance(task, dict):
            step = task.get("step", "")
            vision = task.get("vision_context", "")
            content = f"Krok misji: {step}\nKontekst wizji: {vision}"
        else:
            content = str(task)
        return self.navigator.invoke({"messages": [HumanMessage(content=content)]})


