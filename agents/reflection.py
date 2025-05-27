from langchain.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import
from langchain.schema import Document
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.docstore.in_memory import InMemoryDocstore  # Corrected import
import os
from dotenv import load_dotenv
import faiss  # Import FAISS to manually initialize an empty index

load_dotenv()

class ReflectionAgent:
    def __init__(self, db_path="reflection_db"):
        self.embeddings = OpenAIEmbeddings()
        self.db_path = db_path
        if os.path.exists(db_path):
            self.vector_db = FAISS.load_local(db_path, self.embeddings)
        else:
            # Manually initialize an empty FAISS index
            dimension = len(self.embeddings.embed_query("test"))  # Get embedding dimension
            index = faiss.IndexFlatL2(dimension)
            docstore = InMemoryDocstore({})  # Use InMemoryDocstore
            self.vector_db = FAISS(index=index, embedding_function=self.embeddings.embed_query, docstore=docstore, index_to_docstore_id={})

    def add_mission_feedback(self, mission_description: str, feedback: str, success: bool):
        """
        Add feedback about a mission to the vector database.
        """
        document = Document(
            page_content=f"Mission: {mission_description}\nFeedback: {feedback}\nSuccess: {success}"
        )
        self.vector_db.add_documents([document])
        self.vector_db.save_local(self.db_path)

    def retrieve_similar_missions(self, query: str, top_k: int = 3):
        """
        Retrieve similar missions from the vector database based on a query.
        """
        results = self.vector_db.similarity_search(query, k=top_k)
        return [result.page_content for result in results]

    def suggest_improvements(self, mission_description: str):
        """
        Suggest improvements for a mission based on past feedback.
        """
        similar_missions = self.retrieve_similar_missions(mission_description)
        if not similar_missions:
            return "No similar missions found in the database."

        llm = ChatOpenAI(model="gpt-4", temperature=0.3, max_tokens=500)
        prompt = f"""
Jesteś agentem Reflection. Twoim zadaniem jest zasugerowanie ulepszeń dla misji na podstawie podobnych misji i ich wyników.

Opis misji:
{mission_description}

Podobne misje i ich wyniki:
{chr(10).join(similar_missions)}

Zaproponuj ulepszenia dla tej misji:
"""
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        return response.content.strip()
