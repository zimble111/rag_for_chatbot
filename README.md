# RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about Pick-up Point (ПВЗ) operations.
It combines Dense Passage Retrieval (DPR) for context search and Mistral-7B for natural language generation.
The interface is powered by Gradio, providing an interactive chat demo.


## Retriever:

- Uses Facebook DPR for question encoding
- Context encoder to embed knowledge base chunks
- FAISS index for fast similarity search

## Generator:

- Uses Mistral-7B-Instructfor response generation
- Configurable generation parameters (temperature, top-p, max tokens)

## Evaluation metrics included:

-F1 score
-ROUGE-L

## Frontend:

- Simple Gradio ChatInterface with title and description
- Instant local or shared demo link
