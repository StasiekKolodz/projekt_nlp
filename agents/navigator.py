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
            )
            # ,
            # Tool(
            #     name="None",
            #     func=self.do_nothing,
            #     description=(
            #         "Do nothing. Use this if no action is needed at the moment."
            #     )
            # )
        ]

        self.navigator = create_react_agent(
            ChatOpenAI(model="gpt-4o-mini"),
            self.tools,
        )
        self.current_step = None
        self.current_vision = None

    # def do_nothing(self, arg=None):
    #     return "No action taken. Please specify a valid command."

    def takeoff(self, altitude=2.0):
        if not str(altitude).isnumeric() or float(altitude) <= 0:
            return "Invalid altitude. Must be a positive number."

        altitude = float(altitude)

        msg = self.message_pool.build_message(
            "drone_action",
            {"step": self.current_step,
            "vision_context": self.current_vision,
            "action": "takeoff",
            "parameters": altitude,
            "executed": False,
            "logged": False}
        )
        self.message_pool.post(msg)
        # print(f"[NAVIGATOR] Posting takeoff message: {msg}")

        return f"Drone taking off to {altitude} meters."

    def fly_to(self, north=0.0, east=0.0, down=0.0):
        if not all(str(x).isnumeric() for x in [north, east, down]):
            return "Invalid coordinates. All values must be numbers."

        msg = self.message_pool.build_message(
            "drone_action",
            {"step": self.current_step,
            "vision_context": self.current_vision,
            "action": "fly_to",
            "parameters": [float(north), float(east), float(down)],
            "executed": False,
            "logged": False}
        )
        # print(f"[NAVIGATOR] Posting fly_to message: {msg}")
        self.message_pool.post(msg)
        return f"Drone flying to (N:{north}, E:{east}, D:{down})."

    def land(self, arg=None):
        msg = self.message_pool.build_message(
            "drone_action",
            {"step": self.current_step,
             "vision_context": self.current_vision,
            "action": "land",
            "executed": False,
            "logged": False}
        )
        self.message_pool.post(msg)
        # print(f"[NAVIGATOR] Posting land message: {msg}")
        return "Drone landing."

    def read_messages(self):
        while True:
            messages = self.message_pool.get_all()
            for msg in messages:
                if msg["msg_type"] == "mission_steps" and not msg["content"].get("executed"):
                    if msg["content"].get("vision_context") is not None:
                        vision_context = msg["content"]["vision_context"]
                        self.current_vision = vision_context
                        for step in msg["content"]["mission_plan"]:
                            self.current_step = step
                            print(f"[NAVIGATOR] Executing step: {step}")
                            content = f"Krok misji: {step} \nKontekst wizji: {vision_context}"
                            result = self.navigator.invoke({"messages": [HumanMessage(content=content)]}, 
                                                        #    {"recursion_limit": 100}
                                                           )
                            print(f"Result: {result}")
                            modified_msg = msg
                            modified_msg["executed"] = True

                            self.message_pool.post(modified_msg)
                            self.message_pool.remove_message(msg)
            # time.sleep(2)

    def run_task(self, task):
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
        print("[NAVIGATOR] Navigator agent started and listening for tasks...")
