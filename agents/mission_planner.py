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

        # Example tool: you can add more tools as needed
        self.tools = [
            Tool(
                name="PlanMission",
                func=self.request_mission,
                description="Plan drone mission steps from operator command."
            ),
            Tool(
                name="VectorStoreSearch",
                func=self.vector_search_tool,
                description="Use this to search the vector store for information about previous missions."
            ),

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
                "executed": False,
                "logged": False
            }
        )
        self.message_pool.post(msg)
        return "\n[MISSION PLANNER] Planowanie misji zostało zlecone. Proszę czekać na wynik."

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
            print(f"messages: {messages}")
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
                    modified_msg["executed"] = True

                    self.message_pool.post(modified_msg)
                    self.message_pool.post(result_msg)
                    self.message_pool.remove_message(msg)
                    print(f"\n[MISSION PLANNER] Mission Plan: {mission_plan}")
        
                if msg["msg_type"] == "print_user":
                    print(f"Message: {msg['content']}")
            time.sleep(1)
        
    def run(self):
        poller = threading.Thread(target=self.read_messages, daemon=True)
        poller.start()

        messages = []  # Store messages to display above the input prompt

        while True:
            os.system('clear')  # Clear the terminal
            print("=== Mission Planner ===")
            print("\n".join(messages))  # Display all messages above the input prompt
            print("\n[MISSION PLANNER] Enter your command: ", end="", flush=True)

            sys.stdout.flush()  # Ensure everything is printed before input
            user_input = input()  # Get user input

            if user_input.lower() == "exit":
                print("Exiting mission planner.")
                if hasattr(self, '_timer'):
                    self._timer.cancel()
                break

            # Process the input and add the response to the messages list
            response = self.chat(user_input)
            messages.append(response)

            # Limit the number of messages displayed to avoid clutter
            if len(messages) > 20:
                messages.pop(0)
