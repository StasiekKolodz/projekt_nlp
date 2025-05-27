from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

class GuardianAgent:
    def __init__(self, message_pool=None):
        self.message_pool = message_pool
        self.llm = ChatOpenAI(
            model="gpt-4.1-nano",
            temperature=0.2,
            max_tokens=300,
        )

    def validate(self, mission_step, planned_action):
        prompt = f"""
            Jesteś agentem Guardian. Twoim zadaniem jest sprawdzić, czy planowana akcja nawigatora jest logiczna i poprawna względem kroku misji.

            Krok misji:
            {mission_step}

            Planowana akcja:
            {planned_action}

            Odpowiedz tylko 'OK' jeśli akcja jest logiczna i poprawna. Jeśli nie, napisz krótko dlaczego odrzucasz akcję.
            """
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])
        return response.content.strip()
    
    def run(self):
        pass