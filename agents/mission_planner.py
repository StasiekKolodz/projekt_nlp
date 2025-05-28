from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
import json
import threading
import time
from langgraph.checkpoint.memory import MemorySaver

class MissionPlannerAgent:
    def __init__(self, message_pool):
        self.message_pool = message_pool
        self.llm = ChatOpenAI(model="gpt-4", max_tokens=500)
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        # memory = MemorySaver()

        # Example tool: you can add more tools as needed
        self.tools = [
            Tool(
                name="PlanMission",
                func=self.request_mission,
                description="Plan drone mission steps from operator command."
            )
        ]
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="conversational-react-description",
            # checkpointer=memory,
            memory=self.memory,
            verbose=True
        )
        self.current_input = None

    def chat(self, user_input: str):
        self.current_input = user_input
        response = self.agent.invoke({"input": user_input})
        return f"🤖 Mission Planner: {response.get('output')}\n"

    def request_mission(self, arg=None):
        msg = self.message_pool.build_message(
            "plan_mission",
            self.current_input
        )
        self.message_pool.post(msg)
        return "Planowanie misji zostało zlecone. Proszę czekać na wynik."

    def plan_mission(self, operator_command: str):

        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            max_tokens=500,
        )
        prompt_text = f"""
            Jesteś agentem Mission-Planner dla drona. Twoim zadaniem jest przekształcenie polecenia operatora
            w listę jasnych, małych kroków opisujących działania drona. Nie opisuj, jak dron ma to zrobić – tylko co ma wykonać.

            Polecenie operatora:
            "{operator_command}"

            Zwróć wynik w postaci listy JSON. Każdy krok powinien zawierać numer i cel misji.

            Przykład formatu:
            [
            {{ "id": 1, "cel": "Wystartuj" }},
            {{ "id": 2, "cel": "Leć 10m na zachód" }},
            {{ "id": 3, "cel": "Zrób zdjęcie" }},
            ...
            ]
            """
        message = HumanMessage(content=prompt_text)
        response = llm.invoke([message])
        try:
            # Try to extract JSON from the response
            plan = json.loads(response.content)
        except Exception:
            plan = response.content
        return plan

    def read_messages(self):
        while True:
            messages = self.message_pool.get_all()
            for msg in messages:
                if msg["msg_type"] == "plan_mission":
                    mission_plan = self.plan_mission(msg["content"])
                    result_msg = self.message_pool.build_message(
                            "mission_steps",
                            {"mission_plan": mission_plan,
                            "vision_context": None}
                        )
                    self.message_pool.post(result_msg)
                    self.message_pool.remove_type("plan_mission")
                    print(f"📜 Mission Plan: {mission_plan}")
        
                if msg["msg_type"] == "print_user":
                    print(f"Message: {msg['content']}")
            time.sleep(2)
        
    def run(self):

        # self._timer = threading.Timer(1.0, self.read_messages)
        # self._timer.daemon = True
        # self._timer.start()
        poller = threading.Thread(target=self.read_messages, daemon=True)
        poller.start()

        while True:
            user_input = input("Enter your command: ")
            if user_input.lower() == "exit":
                print("Exiting mission planner.")
                if hasattr(self, '_timer'):
                    self._timer.cancel()
                break
            print(self.chat(user_input))
