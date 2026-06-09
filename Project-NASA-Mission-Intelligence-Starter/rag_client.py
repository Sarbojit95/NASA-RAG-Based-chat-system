from random import sample
from xmlrpc import client
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path
from openai import OpenAI
import os

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    
    chroma_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and "chroma" in d.name.lower()
    ]

    for chroma_dir in chroma_dirs:

        try:

            client = chromadb.PersistentClient(
                path=str(chroma_dir)
            )

            collections = client.list_collections()

            for collection in collections:

                collection_name = collection.name

                key = f"{chroma_dir.name}_{collection_name}"

                try:
                    doc_count = client.get_collection(
                        collection_name
                    ).count()
                except Exception:
                    doc_count = "Unknown"

                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection_name,
                    "display_name":
                        f"{chroma_dir.name} | {collection_name}",
                    "document_count": doc_count
                }

        except Exception as e:

            backends[chroma_dir.name] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name":
                    f"{chroma_dir.name} (Error: {str(e)[:50]})",
                "document_count": 0
            }

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    client = chromadb.PersistentClient(
    path=chroma_dir
    )

    collection = client.get_collection(
    collection_name
    )

    return collection

def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None
):

    client = OpenAI(
        base_url="https://openai.vocareum.com/v1",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    query_embedding = list(
        map(float, response.data[0].embedding)
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results    

    

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    context_parts = ["NASA Mission Reference Context\n"]

    for idx, (doc, metadata) in enumerate(
    zip(documents, metadatas),
    start=1
    ):

        mission = metadata.get(
            "mission",
            "Unknown"
        ).replace("_", " ").title()

        source = metadata.get(
            "source",
            "Unknown"
        )

        category = metadata.get(
            "document_category",
            "Unknown"
        ).replace("_", " ").title()

        header = (
            f"\nSource {idx}\n"
            f"Mission: {mission}\n"
            f"Document: {source}\n"
            f"Category: {category}\n"
            f"{'-'*40}"
        )

        context_parts.append(header)

        if len(doc) > 1500:

            context_parts.append(
                doc[:1500] + "..."
            )

        else:

            context_parts.append(doc)

    return "\n".join(context_parts)