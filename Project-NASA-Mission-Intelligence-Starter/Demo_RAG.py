from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import os
import re

import rag_client
import llm_client
import ragas_evaluator

load_dotenv()

print("=" * 80)
print("NASA RAG EVALUATION DEMO")
print("=" * 80)

# --------------------------------------------------
# LOAD COLLECTION
# --------------------------------------------------

collection = rag_client.initialize_rag_system(
    "./chroma_db_openai",
    "nasa_space_missions_text"
)

print("\nCollection loaded")
print("Documents:", collection.count())

# --------------------------------------------------
# OPENAI CLIENT
# --------------------------------------------------

client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# --------------------------------------------------
# LOAD QUESTIONS
# --------------------------------------------------

questions = [
    "What was the primary objective of Apollo 11?",
    "Who were the crew members of Apollo 11?",
    "What emergency occurred during Apollo 13?",
    "What role did Mission Control play during Apollo 13?",
    "What factors contributed to the Challenger disaster?",
    "What communication systems were used during Apollo missions?"
]

all_scores = []

# --------------------------------------------------
# PROCESS EACH QUESTION
# --------------------------------------------------

for idx, question in enumerate(questions, start=1):

    print("\n" + "=" * 80)
    print(f"QUESTION {idx}")
    print("=" * 80)
    print(question)

    # ----------------------------------------------
    # EMBEDDING
    # ----------------------------------------------

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    query_embedding = list(
        map(float, response.data[0].embedding)
    )

    # ----------------------------------------------
    # RETRIEVAL
    # ----------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    print("\nRetrieved Documents:", len(documents))

    # ----------------------------------------------
    # CONTEXT
    # ----------------------------------------------

    context = rag_client.format_context(
        documents,
        metadatas
    )

    # ----------------------------------------------
    # GENERATION
    # ----------------------------------------------

    answer = llm_client.generate_response(
        os.getenv("OPENAI_API_KEY"),
        question,
        context,
        [],
        "gpt-4o-mini"
    )

    print("\nANSWER:")
    print(answer)

    # ----------------------------------------------
    # RAGAS
    # ----------------------------------------------

    try:

        scores = ragas_evaluator.evaluate_response_quality(
            question,
            answer,
            documents
        )

        print("\nRAGAS SCORES:")

        for metric, score in scores.items():
            print(f"{metric}: {score}")

        all_scores.append(scores)

    except Exception as e:

        print("\nRAGAS ERROR:")
        print(e)

# --------------------------------------------------
# AVERAGE SCORES
# --------------------------------------------------

print("\n" + "=" * 80)
print("AVERAGE PERFORMANCE")
print("=" * 80)

faithfulness = [
    s["faithfulness"]
    for s in all_scores
    if "faithfulness" in s
]

relevancy = [
    s["response_relevancy"]
    for s in all_scores
    if "response_relevancy" in s
]

if faithfulness:
    print(
        "Average Faithfulness:",
        round(sum(faithfulness) / len(faithfulness), 4)
    )

if relevancy:
    print(
        "Average Response Relevancy:",
        round(sum(relevancy) / len(relevancy), 4)
    )

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)