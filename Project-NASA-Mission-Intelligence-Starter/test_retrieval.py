import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import os
print(os.getenv("OPENAI_API_KEY"))

client = chromadb.PersistentClient(path="./chroma_db_openai")

collection = client.get_collection("nasa_space_missions_text")

oai = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

response = oai.embeddings.create(
    model="text-embedding-3-small",
    input="Who were the crew members of Apollo 13?"
)

embedding = response.data[0].embedding

print("Embedding length:", len(embedding))

print("Before query")

results = collection.query(
    query_embeddings=[embedding],
    n_results=3
)

print("After query")
print(results.keys())
print(len(results["documents"][0]))

print(results.keys())
print(len(results["documents"][0]))