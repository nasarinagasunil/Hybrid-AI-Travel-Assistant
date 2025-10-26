
# AI-Hybrid Chat: A Retrieval-Augmented Travel Assistant

[]()

This project is an advanced, retrieval-augmented generation (RAG) system designed to function as an intelligent travel assistant. It answers natural language queries by combining the power of **semantic vector search** with the contextual richness of a **knowledge graph**. The final, enriched context is then fed to a large language model to synthesize a coherent, data-driven response.

## Core Features

  * **Hybrid Retrieval**: Utilizes **Pinecone** for fast, semantic vector search to understand user intent and **Neo4j** to enrich the search results with explicit, factual relationships from a knowledge graph.
  * **Adaptable AI Backend**: Initially designed for OpenAI, the system was successfully migrated to use **Google's Generative AI**, employing the `text-embedding-004` model for embeddings and the powerful `gemini-2.5-pro` model for final response generation.
  * **Chain-of-Thought (CoT) Prompting**: The prompt is engineered to guide the AI to first "reason" about the best travel plan based on the context before generating the final, structured itinerary, significantly improving the quality of the output.
  * **Performance Optimization**: The system includes several performance enhancements:
      * **Asynchronous Operations**: Uses Python's `asyncio` library and an async Neo4j driver to handle network requests efficiently.
      * **Embedding Caching**: Implements an in-memory cache to store query embeddings, reducing API costs and latency on repeated queries.
  * **Dynamic Summarization**: A summary function analyzes the initial search results to provide the language model with a concise overview of the key themes, helping it generate a more focused response.

## Technology Stack

  * **Language**: Python 3.11+
  * **Vector Database**: Pinecone (Serverless Index)
  * **Graph Database**: Neo4j (Local Desktop Instance)
  * **AI Models**: Google Generative AI (`text-embedding-004`, `gemini-2.5-pro`)
  * **Core Libraries**:
      * `pinecone-client`
      * `neo4j` (async driver)
      * `google-generativeai`
      * `asyncio`
      * `pyvis` & `networkx` (for visualization)
      * `tqdm`

## Project Structure

```
Travelling-AI-Hybrid-Chat/
├── data/
│   └── vietnam_travel_dataset.json
├── docs/
│   └── improvements.md
├── venv/
├── config.py
├── hybrid_chat.py
├── load_to_neo4j.py
├── pinecone_upload.py
├── visualize_graph.py
├── requirements.txt
├── app.py
└── README.md
```

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1\. Prerequisites

  * **Python 3.11+** installed.
  * **Neo4j Desktop** installed and running.

### 2\. Initial Setup

1.  **Clone the repository**:

    ```bash
    git clone <repository-url>
    cd Travelling-AI-Hybrid-Chat
    ```

2.  **Create and activate a virtual environment**:

    ```bash
    python -m venv venv
    # On Windows (PowerShell)
    .\venv\Scripts\activate
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

### 3\. Configuration

1.  **Set up Neo4j**:

      * Open Neo4j Desktop and create a new local database instance.
      * Set the password for the default `neo4j` user.
      * Ensure the database is running.

2.  **Create Pinecone Index**:

      * Log in to your [Pinecone account](https://app.pinecone.io/).
      * Create a new **Serverless** index with the following settings:
          * **Index Name**: `vietnam-travel-google`
          * **Dimensions**: `768`
          * **Metric**: `cosine`

3.  **Update `config.py`**:

      * Rename `config.py.sample` to `config.py`.
      * Fill in your credentials:
          * `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD`.
          * `GOOGLE_API_KEY` (from Google AI Studio).
          * `PINECONE_API_KEY` and `PINECONE_ENV` (e.g., `aws-us-east-1`).
          * Ensure `PINECONE_INDEX_NAME` is set to `vietnam-travel-google`.
          * Ensure `PINECONE_VECTOR_DIM` is set to `768`.

4.  **Update `hybrid_chat.py`**:

      * Find the `INDEX_HOST` variable and replace the placeholder with the actual host URL for your `vietnam-travel-google` index from the Pinecone dashboard.

## Usage

Run the scripts from your activated virtual environment in the following order.

1.  **Load Data into Neo4j**:
    This script populates the graph database. It only needs to be run once.

    ```bash
    python load_to_neo4j.py
    ```

2.  **Upload Embeddings to Pinecone**:
    This script generates vector embeddings for the dataset and uploads them to your Pinecone index.

    ```bash
    python pinecone_upload.py
    ```

3.  **Launch the Streamlit Web Application**:
    This command starts the web server and opens the AI Travel Assistant in your browser.

    ```bash
    streamlit run app.py
    ```

    You can now interact with the assistant through the web interface. Ask it questions like create a romantic 4 day itinerary for Vietnam or where can I go trekking?

4.  **(Optional) Visualize the Graph**:
    This generates an `neo4j_viz.html` file that you can open in your browser to see the graph.

    ```bash
    python visualize_graph.py
    ```
