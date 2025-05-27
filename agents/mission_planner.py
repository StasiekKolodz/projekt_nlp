from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import json

def plan_mission(operator_command: str):
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
