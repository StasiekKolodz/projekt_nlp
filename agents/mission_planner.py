from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
import json
import threading
import time
from langchain.chains import RetrievalQA
import os
import sys
import copy

class MissionPlannerAgent:
    def __init__(self, message_pool, retriever=None):
        self.message_pool = message_pool
        self.retriever = retriever
        self.llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=500)
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

        self.tools = [
            Tool(
            name="PlanMission",
            func=self.request_mission,
            description=(
                "Use this tool to convert the operator's command into a step-by-step mission plan for the drone. "
                "It will break down the user's instruction into clear, numbered actions for the drone to execute. "
                "Call this when the user asks to plan or create a mission, provides commands for the drone, "
                "or types commands such as 'land' or 'takeoff'."
            ),
            ),
            Tool(
            name="VectorStoreSearch",
            func=self.vector_search_tool,
            description=(
                "Search the vector store for relevant information about previous missions or related knowledge. "
                "Use this tool to retrieve context or details that may help in planning or answering questions."
            )
            ),
        ]
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="conversational-react-description",
            # checkpointer=memory,
            memory=self.memory,
            verbose=False
        )
        self.current_input = None
        self.validation_ok = 0
        self.validation_fail = 0
    # https://hub.athina.ai/blogs/agentic-rag-using-langchain-and-gemini-2-0/
    def vector_search(self, query: str):
        if self.retriever:
            qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever)
            return qa_chain.run(query)
        else:
            return "Vector store is inactive"

    def vector_search_tool(self, query: str) -> str:
        """Tool for searching the vector store."""
        return self.vector_search(query)

    def chat(self, user_input: str):
        self.current_input = user_input
        response = self.agent.invoke({"input": user_input})
        return f"🤖 Mission Planner: {response.get('output')}\n"

    def request_mission(self, arg=None):
        msg = self.message_pool.build_message(
            "plan_mission",
            {
            "user_input": self.current_input,
            "chat_history": self.memory.load_memory_variables({}).get("chat_history", []),
            "executed": False,
            "logged": False
            }
        )
        self.message_pool.post(msg)
        return "\n[MISSION PLANNER] Planowanie misji zostało zlecone. Zaraz misja ostanie wykonana. Przebieg misji wyświetli się na ekranie."

    def plan_mission(self, msg):
        operator_command = msg.get("user_input", "")
        chat_history = msg.get("chat_history", [])
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            max_tokens=500,
        )
        prompt_text = f"""
            Jesteś agentem Mission-Planner dla drona. Twoim zadaniem jest przekształcenie polecenia operatora drona
            w listę jasnych, małych kroków opisujących działania drona. Nie opisuj, jak dron ma to zrobić – tylko co ma wykonać. 
            Jeśli misja wymaga tylko jednego kroku, zwróć listę z jednym krokiem.
            Jeśli misja wymaga wielu kroków, zwróć listę z wieloma krokami, gdzie każdy krok ma swój numer i cel.
            Kroki misji powinny być niezależne od siebie i nie powinny zawierać przejść ani pętli.
            Jeśli krok wymaga warunków, to warunek powinien byc w jednym kroku razem z wymaganą akcją.
            Akcje powinny być ograniczone do startowania, lądowania, lotu w określonym kierunku i wysokości. Mogą mieć one warunki i parametry, ale nie powinny zawierać skomplikowanych instrukcji.
            
            Oto szczegóły:

            Polecenie operatora:
            "{operator_command}"

            Historia chatu z operatorem:
            {chat_history}

            Kot
            Zwróć wynik w postaci listy JSON. Każdy krok powinien zawierać numer i cel misji.

            Przykład formatu:
            [
            {{ "id": 1, "cel": "Wystartuj" }},
            {{ "id": 2, "cel": "Leć 10m na zachód" }},
            {{ "id": 3, "cel": "Jeśli widzisz dom to obniż lot o 1m" }},
            {{ "id": 4, "cel": "Wyląduj" }},
            ...
            ]
            """
        message = HumanMessage(content=prompt_text)
        response = llm.invoke([message])
        try:
            plan = json.loads(response.content)
        except Exception:
            plan = response.content
        return plan

    def clean_messages(self):
        messages = self.message_pool.get_all()
        for msg in messages:
            if msg["content"].get("executed", False) and msg["content"].get("logged", False):
                self.message_pool.remove(msg)

    def read_messages(self):
        while True:
            messages = self.message_pool.get_all()
            # print(f"messages: {messages}")
            for msg in messages:
                if msg["msg_type"] == "plan_mission" and not msg["content"].get("executed"):
                    mission_plan = self.plan_mission(msg["content"])
                    result_msg = self.message_pool.build_message(
                            "mission_steps",
                            {"mission_plan": mission_plan,
                            "vision_context": None,
                            "executed": False,
                            "logged": False}
                        )

                    modified_msg = msg.copy()
                    modified_msg["content"]["executed"] = True

                    self.message_pool.post(modified_msg)
                    self.message_pool.post(result_msg)
                    self.message_pool.remove_message(msg)
                    print("\n[MISSION PLANNER] Mission plan:")
                    for plan in mission_plan:
                        print(f"{plan['id']} - {plan['cel']}")
                
                if msg["msg_type"] == "guardian_validation":
                    # Add guardian validation info to chat history
                    validation_info = msg["content"]
                    step = validation_info.get("step")
                    action = validation_info.get("action")
                    parameters = validation_info.get("parameters")
                    validation = validation_info.get("validation")
                    if validation == "OK":
                        chat_entry = f"EXECUTED MISSION STEP: Step: {step['cel']} | Action: {action} | Parameters: {parameters} | Validation: {validation}"
                        self.validation_ok += 1
                    else:
                        chat_entry = f"REJECTED (failed) MISSION STEP: Step: {step['cel']} | Action: {action} | Parameters: {parameters} | Validation: {validation}"
                        self.validation_fail += 1
                    
                    # Add to memory
                    self.memory.save_context({"input": chat_entry}, {"output": ""})
                    # print(chat_entry)
                    self.message_pool.remove_message(msg)

                if msg["msg_type"] == "print_user":
                    print(f"Message: {msg['content']}")
            time.sleep(1)
        
    def run(self):
        poller = threading.Thread(target=self.read_messages, daemon=True)
        poller.start()

        messages = []

        while True:
            os.system('clear')
            print("=== Mission Planner ===")
            print("\n".join(messages))
            print("\n[MISSION PLANNER] Enter your command: ", end="", flush=True)

            sys.stdout.flush() 

            user_input = input()

            if user_input.lower() == "exit":
                print("Exiting mission planner.")
                if hasattr(self, '_timer'):
                    self._timer.cancel()
                print("TOTALGUARDIAN VALIDATIONS:", self.validation_ok + self.validation_fail)
                print("VALIDATIONS OK:", self.validation_ok)
                print("VALIDATIONS FAILED:", self.validation_fail)
                break

            response = self.chat(user_input)
            messages.append(response)

            if len(messages) > 20:
                messages.pop(0)
