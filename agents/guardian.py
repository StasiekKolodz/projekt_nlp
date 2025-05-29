from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
import threading
import time
from tools import drone_tools
import copy

class GuardianAgent:
    def __init__(self, message_pool=None):
        self.message_pool = message_pool
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=300,
        )

    def validate(self, mission_step, planned_action, vision_context=None, parameters=None):
        prompt = f"""
            Jesteś agentem Guardian. Twoim zadaniem jest sprawdzić, czy planowana akcja nawigatora jest logiczna i poprawna względem kroku misji.

            Krok misji:
            {mission_step}

            Kontekst wizyjny (opis widoku z kamery):
            {vision_context}

            Planowana akcja:
            {planned_action}

            Parametry akcji (jeśli są):
            {parameters if parameters else 'Brak parametrów'}

            Odpowiedz tylko 'OK' jeśli akcja jest logiczna i poprawna w danym kontekście wizyjnym. Jeśli nie, napisz krótko dlaczego odrzucasz akcję.
            """
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])
        return response.content.strip()

    def execute_action(self, action, parameters=None):
        if action == "takeoff":
            drone_tools.takeoff(parameters)
            print(f"[GUARDIAN] Drone action '{action}' executed with parameters: {parameters}")
        elif action == "fly_to":
            drone_tools.fly_to(parameters)
            print(f"[GUARDIAN] Drone action '{action}' executed with parameters: {parameters}")
        elif action == "land":
            drone_tools.land()
            print(f"[GUARDIAN] Drone action '{action}' executed with parameters: {parameters}")
        else:
            print(f"[GUARDIAN] Unknown action: {action}. No execution performed.")
        time.sleep(2)
    def read_messages(self):
        while True:
            messages = self.message_pool.get_all()
            for msg in messages:
                if msg["msg_type"] == "drone_action" and not msg["content"].get("executed"):
                    step = msg["content"].get("step")
                    action = msg["content"].get("action")
                    parameters = msg["content"].get("parameters", None)
                    vision_context = msg["content"].get("vision_context")
                    validation = self.validate(step, action, vision_context, parameters)
                    if validation != "OK":
                        print(f"[GUARDIAN] Guardian validation failed for step '{step['cel']}': {validation}")
                        result_msg = self.message_pool.build_message(
                            "guardian_validation",
                            {"step": step,
                            "validation": validation,
                            "logged": False}
                        )
                    else:
                        print(f"[GUARDIAN] Guardian validation passed for step '{step['cel']}'")
                        self.execute_action(action, parameters)
                        result_msg = self.message_pool.build_message(
                            "guardian_validation",
                            {"step": step,
                            "validation": "OK",
                            "logged": False}
                        )
                 
                    modified_msg = msg.copy()
                    modified_msg["content"]["executed"] = True

                    self.message_pool.post(modified_msg)
                    self.message_pool.post(result_msg)
                    self.message_pool.remove_message(msg)
            time.sleep(1)

    def start(self):
        guardian_thread = threading.Thread(target=self.read_messages, daemon=True)
        guardian_thread.start()
        print("[GUARDIAN] Guardian agent started and listening for planned actions...")
