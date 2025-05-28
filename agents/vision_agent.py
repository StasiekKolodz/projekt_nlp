from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
import requests
import time
import threading

class VisionAgent:
    def __init__(self, message_pool=None):
        self.message_pool = message_pool
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            max_tokens=500,
        )

        self.api_url = "http://localhost:5002/camera_image"

    def describe_image(self, image_path: str):
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{image_base64}"
        llm = ChatOpenAI(
            model="gpt-4-turbo",
            max_tokens=500,
        )
        message = HumanMessage(
            content=[
                {"type": "text", "text": """
    Jesteś agentem wizji komputerowej, którego zadaniem jest pomoc agentowi nawigacyjnemu drona. Otrzymujesz obraz z kamery zamontowanej na dronie.
    Twoim zadaniem jest wygenerować bardzo zwięzły opis przestrzenny tego, co znajduje się na obrazie — tylko informacje kluczowe dla nawigacji drona.
    Nie opisuj rodzaju ani koloru obiektów. Skup się tylko na ich:
    - Położeniu względem kamery (np. 'na wprost', 'po lewej', 'w prawym dolnym rogu'),
    - Szacunkowej odległości od kamery (np. 'blisko', 'daleko', 'średni dystans'),
    - Rozmiarze w kadrze (np. 'duży', 'mały', 'zajmuje większość kadru').
    Odpowiedź powinna mieć maksymalnie 3 zdania i może opcjonalnie przyjąć format listy.
    Analizuj obraz:
                """},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        )
        response = llm.invoke([message])
        return response.content

    def describe_image_from_api(self):
    
        resp = requests.get(self.api_url)
        if resp.status_code != 200:
            return f"API error: {resp.text}"
        image_bytes = resp.content
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{image_base64}"
        prompt = (
            "Jesteś agentem wizji komputerowej, którego zadaniem jest pomoc agentowi nawigacyjnemu drona. "
            "Otrzymujesz obraz z kamery zamontowanej na dronie.\n"
            "Twoim zadaniem jest wygenerować bardzo zwięzły opis przestrzenny tego, co znajduje się na obrazie — tylko informacje kluczowe dla nawigacji drona.\n"
            "Nie opisuj rodzaju ani koloru obiektów. Skup się tylko na ich:\n"
            "- Położeniu względem kamery (np. 'na wprost', 'po lewej', 'w prawym dolnym rogu'),\n"
            "- Szacunkowej odległości od kamery (np. 'blisko', 'daleko', 'średni dystans'),\n"
            "- Rozmiarze w kadrze (np. 'duży', 'mały', 'zajmuje większość kadru').\n"
            "Odpowiedź powinna mieć maksymalnie 3 zdania i może opcjonalnie przyjąć format listy.\n"
            "Analizuj obraz:"
        )
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        )
        response = self.llm.invoke([message])
        return response.content

    def read_messages(self):
        while True:
            messages = self.message_pool.get_all()
            for msg in messages:
                if msg["msg_type"] == "mission_steps":
                    if msg["content"].get("vision_context") is None:
                        # vision_context = self.describe_image_from_api()
                        # print("👁️ Processing plan_mission message without vision context...")
                        vision_context = self.describe_image("/home/stas/studia/nlp/projekt_nlp/person_img.jpeg")
                        print(f"👁️ Vision context generated: {vision_context}")

                        new_content = msg["content"]

                        self.message_pool.remove_message(msg)

                        new_content["vision_context"] = vision_context
                        result_msg = self.message_pool.build_message(
                            msg["msg_type"],
                            new_content
                        )
                        self.message_pool.post(result_msg)
            time.sleep(2)

    def start(self):
        vision_thread = threading.Thread(target=self.read_messages, daemon=True)
        vision_thread.start()
        print("👁️ Vision agent started and listening for plan_mission messages...")