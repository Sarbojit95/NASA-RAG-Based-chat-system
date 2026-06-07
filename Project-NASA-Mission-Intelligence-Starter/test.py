from rag_client import initialize_rag_system, retrieve_documents
import os
from dotenv import load_dotenv

load_dotenv()

collection = initialize_rag_system(
    "./chroma_db_openai",
    "nasa_space_missions_text"
)

results = retrieve_documents(
    collection,
    "Who were the crew members of Apollo 13?"
)

print(len(results["documents"][0]))