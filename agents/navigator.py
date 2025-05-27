from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage

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

    def takeoff(self, altitude=2.0):
        if not isinstance(altitude, (int, float)) or altitude <= 0:
            return "Invalid altitude. Must be a positive number."

        self.message_pool.post({
            "type": "drone_action",
            "data": {"action": "takeoff", "altitude": altitude}
        })
        return f"Drone taking off to {altitude} meters."

    def fly_to(self, north=0.0, east=0.0, down=0.0):
        if not all(isinstance(x, (int, float)) for x in [north, east, down]):
            return "Invalid coordinates. All values must be numbers."

        self.message_pool.post({
            "type": "drone_action",
            "data": {"action": "fly_to", "north": north, "east": east, "down": down}
        })
        return f"Drone flying to (N:{north}, E:{east}, D:{down})."

    def land(self, arg=None):
        self.message_pool.post({
            "type": "drone_action",
            "data": {"action": "land"}
        })
        return "Drone landing."

    def read_messages(self):
        while True:
            messages = self.message_pool.get_all()
            for msg in messages:
                if msg["msg_type"] == "mission_steps":
                    mission_plan = self.plan_mission(msg["content"])
                    msg = self.message_pool.build_message(
                            "mission_steps",
                            mission_plan
                        )
                    self.message_pool.post(msg)
                    self.message_pool.remove_type("plan_mission")
                    print(f"📜 Mission Plan: {mission_plan}")
        
                if msg["msg_type"] == "print_user":
                    print(f"Message: {msg['content']}")
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

    def run(self):
        pass