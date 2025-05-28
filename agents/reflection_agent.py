from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import threading
import time
import os
from tools import drone_tools

class ReflectionAgent:
    def __init__(self, message_pool=None, vector_store_path="reflection_store"):
        self.message_pool = message_pool
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=300,
        )
        self.vector_store_path = vector_store_path

        if not os.path.exists(vector_store_path) or not os.path.exists(f"{vector_store_path}/index.faiss"):
            os.makedirs(vector_store_path, exist_ok=True)
            self.vector_store = FAISS.from_texts(["Placeholder text for initialization"], OpenAIEmbeddings())
            self.vector_store.save_local(vector_store_path)
        else:
            self.vector_store = FAISS.load_local(
                vector_store_path, 
                OpenAIEmbeddings(), 
                allow_dangerous_deserialization=True
            )

    def reflection(self, mission_step, planned_action, vision_context=None, parameters=None):
        prompt = f"""
            Jesteś agentem Reflection. Twoim zadaniem jest przygotować dane do zapisania w bazie wektorowej na potrzeby następnych misji dronem dostaniesz informacje o powodzeniu misji oraz dane z message pool z całej misji.

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

    def save_to_vector_store(self, mission_success, message_pool_data):
        search_result = self.vector_store.search("Placeholder text for initialization", 1)
        if search_result and "Placeholder text for initialization" in getattr(search_result, 'texts', []):
            self.vector_store.delete(["Placeholder text for initialization"])
        document = f"""
        Mission Success: {mission_success}
        MessagePool Data: {message_pool_data}
        """
        self.vector_store.add_texts([document])
        self.vector_store.save_local(self.vector_store_path)
        print(f"Mission data saved to vector store: {self.vector_store_path}")

    def ask_user_for_mission_success(self):
        print("Czy misja się powiodła? (tak/nie): ", end="", flush=True)
        while True:
            user_input = input().strip().lower()
            if user_input in ["tak", "nie"]:
                return user_input == "tak"
            print("Proszę odpowiedzieć 'tak' lub 'nie': ", end="", flush=True)

    def read_messages(self):
        while True:
            messages = self.message_pool.get_all()
            mission_completed = any(
                msg["msg_type"] == "drone_action" and msg["content"].get("executed", False)
                for msg in messages
            )
            
            if mission_completed:
                print("Mission completed. Asking user for mission success...")
                mission_success = self.ask_user_for_mission_success()

                self.save_to_vector_store(mission_success, messages)

                with self.message_pool.lock:
                    self.message_pool.messages.clear()

                print("System reset. Ready for a new mission.")
                continue

            time.sleep(2)

    def start(self):
        reflection_thread = threading.Thread(target=self.read_messages, daemon=True)
        reflection_thread.start()
        print("Reflection agent started and waiting for mission completion...")
