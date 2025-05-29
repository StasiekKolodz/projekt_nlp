from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage
import threading
import time
import json
from textwrap import indent, wrap
import copy

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
                    "Argument (list of floats, in meters):\n"
                    "  • north (float): positive moves north, negative moves south\n"
                    "  • east (float):  positive moves east,  negative moves west\n"
                    "  • down (float):  positive moves downward, negative moves upward"
                )
            ),
            Tool(
                name="Land",
                func=self.land,
                description=(
                    "Command the drone to descend and land safely at its current location. No arguments required."
                )
            )
        ]

        self.navigator = create_react_agent(
            ChatOpenAI(model="gpt-4"),
            self.tools,
        )
        self.current_step = None
        self.current_vision = None


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

    def fly_to(self, parameters):
        if isinstance(parameters, str):
            parts = parameters.replace(",", " ").split()
            if len(parts) != 3:
                return "Invalid parameters. Expected format: 'north east down'."
            north, east, down = parts
        elif isinstance(parameters, (list, tuple)) and len(parameters) == 3:
            north, east, down = parameters
        elif isinstance(parameters, dict):
            north = parameters.get("north")
            east = parameters.get("east")
            down = parameters.get("down")
            if north is None or east is None or down is None:
                return "Invalid parameters. Expected keys: 'north', 'east', 'down'."
        else:
            return "Invalid parameters. Expected a string, list, tuple, or dict with three values."

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
                            print(f"[NAVIGATOR] Executing step: {step['cel']}")
                            content = f"Krok misji: {step} \nKontekst wizji: {vision_context}"
                            result = self.navigator.invoke({"messages": [HumanMessage(content=content)]}, 
                                                           {"recursion_limit": 25}
                                                           )
                            # print(f"Result: {self.summarize_chat(result)}")
                        modified_msg = msg.copy()
                        modified_msg["content"]["executed"] = True

                        self.message_pool.post(modified_msg)
                        self.message_pool.remove_message(msg)
            time.sleep(1)

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

    def summarize_chat(self, source: str | dict, width: int = 88) -> str:

        if isinstance(source, str):
            data = json.loads(source)
        else:
            data = source

        def _wrap(text: str) -> str:
            text = " ".join(text.split())
            return "\n".join(wrap(text, width)) or "<empty>"

        def _as_dict(msg):
            if isinstance(msg, dict):
                return msg
            if hasattr(msg, "model_dump"):
                return msg.model_dump()
            if hasattr(msg, "dict"):
                return msg.dict()
            return msg.__dict__

        lines: list[str] = []

        for raw in data.get("messages", []):
            msg = _as_dict(raw)

            tool_calls = (
                msg.get("tool_calls")
                or msg.get("additional_kwargs", {}).get("tool_calls")
            )
            if tool_calls is not None:
                lines.append(f"AI ⮕  {_wrap(msg.get('content', ''))}")
                for call in tool_calls:
                    c = _as_dict(call)
                    name = c.get("name") or c.get("function", {}).get("name")
                    raw_args = (
                        c.get("args")
                        or c.get("function", {}).get("arguments", "")
                        or ""
                    )
                    try:
                        parsed = (
                            json.loads(raw_args)
                            if isinstance(raw_args, str) else raw_args
                        )
                        arg_str = json.dumps(parsed, ensure_ascii=False)
                    except Exception:
                        arg_str = str(raw_args)
                    lines.append(indent(f"└─ call {name}({arg_str})", "   "))

            elif msg.get("tool_call_id") or msg.get("name"):
                tool_name = msg.get("name", "<tool>")
                lines.append(f"{tool_name} ⇠  {_wrap(msg.get('content', ''))}")

        return "\n".join(lines)
