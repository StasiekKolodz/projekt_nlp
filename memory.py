from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4", max_tokens=500)
memory = ConversationBufferMemory(return_messages=True)

chat = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Pętla konwersacyjna
print("=== Rozmowa z agentem sterującym dronem (wpisz 'exit' aby zakończyć) ===\n")

while True:
    user_input = input("Operator: ")
    
    if user_input.lower() in ["exit", "quit"]:
        print("Zakończono konwersację.")
        break

    response = chat.predict(input=user_input)
    print(f"🤖 Agent: {response}\n")