from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
import threading
import time
from tools import drone_tools
class GuardianAgent:
    def __init__(self, message_pool=None):
        self.message_pool = message_pool
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.2,
            max_tokens=300,
        )

    def validate(self, mission_step, planned_action, parameters=None):
        prompt = f"""
            Jesteś agentem Guardian. Twoim zadaniem jest sprawdzić, czy planowana akcja nawigatora jest logiczna i poprawna względem kroku misji.

            Krok misji:
            {mission_step}

            Planowana akcja:
            {planned_action}

            Parametry akcji (jeśli są):
            {parameters if parameters else 'Brak parametrów'}

            Odpowiedz tylko 'OK' jeśli akcja jest logiczna i poprawna. Jeśli nie, napisz krótko dlaczego odrzucasz akcję.
            """
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])
        return response.content.strip()

    def execute_action(self, action, parameters=None):
        if action == "takeoff":
            drone_tools.takeoff(parameters)
        elif action == "fly_to":
            drone_tools.fly_to(parameters)
        elif action == "land":
            drone_tools.land()
        else:
            print(f"❗️ Unknown action: {action}. No execution performed.")

    def read_messages(self):
        while True:
            messages = self.message_pool.get_all()
            for msg in messages:
                if msg["msg_type"] == "drone_action" and not msg["content"].get("executed"):
                    step = msg["content"].get("step")
                    action = msg["content"].get("action")
                    parameters = msg["content"].get("parameters", None)

                    validation = self.validate(step, action, parameters)
                    if validation != "OK":
                        print(f"❌ Guardian validation failed for step '{step}': {validation}")
                        result_msg = self.message_pool.build_message(
                            "guardian_validation",
                            {"step": step,
                            "validation": validation,
                            "logged": False}
                        )
                    else:
                        print(f"✅ Guardian validation passed for step '{step}'")
                        self.execute_action(action, parameters)
                        result_msg = self.message_pool.build_message(
                            "guardian_validation",
                            {"step": step,
                            "validation": "OK",
                            "logged": False}
                        )
                 
                    modified_msg = msg
                    modified_msg["executed"] = True

                    self.message_pool.post(modified_msg)
                    self.message_pool.post(result_msg)
                    self.message_pool.remove_message(msg)
            time.sleep(2)

    def start(self):
        guardian_thread = threading.Thread(target=self.read_messages, daemon=True)
        guardian_thread.start()
        print("🛡️ Guardian agent started and listening for planned actions...")
