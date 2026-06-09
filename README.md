# NASA Mission Intelligence Chat System

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system capable of answering questions about NASA space missions using mission transcripts, technical reports, and archival documentation.

The solution combines semantic search, vector databases, large language models, and automated evaluation techniques to provide context-grounded responses.

The system was developed as part of the Udacity Generative AI Nanodegree program.

---

# Features

* NASA mission document ingestion and processing
* Document chunking and metadata extraction
* OpenAI embedding generation using text-embedding-3-small
* ChromaDB vector database integration
* Semantic document retrieval
* GPT-4o-mini response generation
* RAGAS-based response evaluation
* Streamlit user interface
* Automated evaluation workflow

---

# Dataset

The project uses NASA mission documentation covering:

* Apollo 11
* Apollo 13
* Challenger

Document sources include:

* Mission transcripts
* Technical reports
* Mission commentary
* NASA archival documents

### Indexed Dataset Statistics

| Metric       | Value  |
| ------------ | ------ |
| Total Chunks | 15,563 |
| Apollo 11    | 8,387  |
| Apollo 13    | 6,224  |
| Challenger   | 952    |

---

# System Architecture

NASA Documents

↓

Embedding Pipeline

↓

OpenAI text-embedding-3-small

↓

ChromaDB Vector Store

↓

Similarity Search

↓

GPT-4o-mini

↓

Generated Answer

↓

RAGAS Evaluation

---

# Project Structure

```text
.
├── embedding_pipeline.py
├── rag_client.py
├── llm_client.py
├── ragas_evaluator.py
├── chat.py
├── Demo_RAG.py
├── evaluation_dataset.txt
├── requirements.txt
├── README.md
└── chroma_db_openai/
```

---

# Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Building the Vector Database

Run:

```bash
python embedding_pipeline.py
```

This will:

1. Process NASA documents
2. Generate embeddings
3. Store vectors in ChromaDB
4. Generate collection statistics

---

# Running the Application

Launch the Streamlit interface:

```bash
streamlit run chat.py
```

---

# Running Backend Evaluation

A standalone evaluation script is included:

```bash
python Demo_RAG.py
```

This script performs:

* Retrieval
* Context generation
* Answer generation
* RAGAS evaluation

without requiring the Streamlit interface.

---

# Evaluation Dataset

The repository includes:

```text
evaluation_dataset.txt
```

containing NASA mission evaluation questions and expected answers used for testing the system.

---

# Evaluation Results

The backend RAG pipeline was evaluated using the provided evaluation dataset.

Average Scores:

| Metric             | Score  |
| ------------------ | ------ |
| Faithfulness       | 0.7892 |
| Response Relevancy | 0.7912 |

Example Question:

**Who were the crew members of Apollo 13?**

Generated Answer:

* Jim Lovell – Commander
* Fred Haise – Lunar Module Pilot
* Jack Swigert – Command Module Pilot

---

# Known Issue

During final integration testing, a runtime issue was encountered involving Streamlit and ChromaDB similarity search execution.

Extensive debugging was performed across:

* Embedding generation
* ChromaDB collection creation
* Retrieval pipeline
* Context formatting
* Response generation
* RAGAS evaluation

The issue was isolated to the Streamlit integration layer.

To independently validate the backend implementation, a standalone evaluation workflow (Demo_RAG.py) was developed and successfully executed.

The backend pipeline successfully demonstrates:

* Embedding generation
* Semantic retrieval
* Context construction
* LLM response generation
* RAGAS evaluation

Reviewers can reproduce the complete backend workflow by running:

```bash
python Demo_RAG.py
```

# Author

Sarbojit Chakroborty

