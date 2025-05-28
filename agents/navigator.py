from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage
import threading
import time

class NavigatorAgent:
    def __init__(self, message_pool=None):
        self.message_pool = message_pool
        self.tools = [
            Tool(
                name="Takeoff",
                func=self.takeoff,
                description=(
                    "takeoff and fly up to a given altitude.\n"
                    "Arguments:\n"
                    "  • altitude (float): target height in meters, must be > 0."
                )
            ),
            Tool(
                name="FlyTo",
                func=self.fly_to,
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
                func=self.land,
                description=(
                    "Descend and land safely—no arguments required."
                )
            ),
        ]

        self.navigator = create_react_agent(
            ChatOpenAI(model="gpt-4"),
            self.tools,
        )
        self.current_step = None

    def takeoff(self, altitude=2.0):
        if not str(altitude).isnumeric() or float(altitude) <= 0:
            return "Invalid altitude. Must be a positive number."

        altitude = float(altitude)

        msg = self.message_pool.build_message(
            "drone_action",
            {"step": self.current_step, "action": "takeoff", "parameters": altitude}
        )
        self.message_pool.post(msg)

        return f"Drone taking off to {altitude} meters."

    def fly_to(self, north=0.0, east=0.0, down=0.0):
        if not all(str(x).isnumeric() for x in [north, east, down]):
            return "Invalid coordinates. All values must be numbers."

        msg = self.message_pool.build_message(
            "drone_action",
            {"step": self.current_step, "action": "fly_to", "parameters": [float(north), float(east), float(down)]}
        )
        self.message_pool.post(msg)
        return f"Drone flying to (N:{north}, E:{east}, D:{down})."

    def land(self, arg=None):
        msg = self.message_pool.build_message(
            "drone_action",
            {"step": self.current_step, "action": "land"}
        )
        self.message_pool.post(msg)
        return "Drone landing."

    def read_messages(self):
        while True:
            messages = self.message_pool.get_all()
            # print(f"🗨️ Received {len(messages)} messages from the pool.")
            for msg in messages:
                if msg["msg_type"] == "mission_steps":
                    if msg["content"].get("vision_context") is not None:
                        vision_context = msg["content"]["vision_context"]
                        for step in msg["content"]["mission_plan"]:
                            self.current_step = step
                            print(f"🚀 Executing step: {step}")
                            content = f"Krok misji: {step} \nKontekst wizji: {vision_context}"
                            result = self.navigator.invoke({"messages": [HumanMessage(content=content)]})
                            # print(f"Result: {result}")
                            self.message_pool.remove_type("mission_steps")
            time.sleep(2)

    def run_task(self, task):
        # Accepts either dict or string, always wraps as a message
        if isinstance(task, dict):
            step = task.get("step", "")
            vision = task.get("vision_context", "")
            content = f"Krok misji: {step}\nKontekst wizji: {vision}"
        else:
            content = str(task)
        return self.navigator.invoke({"messages": [HumanMessage(content=content)]})

    def start(self):
        navigator_thread = threading.Thread(target=self.read_messages, daemon=True)
        navigator_thread.start()
        print("🧭 Navigator agent started and listening for tasks...")