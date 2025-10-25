import json
import time
from tqdm import tqdm
from openai import OpenAI
import pinecone
import config

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32
INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = config.PINECONE_VECTOR_DIM  # 1536 for text-embedding-3-small

# -----------------------------
# Initialize clients
# -----------------------------
client = OpenAI(api_key=config.OPENAI_API_KEY)

pinecone.init(
    api_key=config.PINECONE_API_KEY,
    environment=config.PINECONE_ENV
)

# -----------------------------
# Create index if it doesn't exist
# -----------------------------
# Bypassing index creation. Index 'vietnam-travel' assumed to be created manually 
# as Serverless (1536/cosine) to avoid V2.x API errors.
print(f"Connecting to existing index: {INDEX_NAME}")

# Connect to index 
index = pinecone.Index(INDEX_NAME)

# Connect to index
index = pinecone.Index(INDEX_NAME)

# -----------------------------
# Helper functions
# -----------------------------
def get_embeddings(texts, model="text-embedding-3-small"):
    """Generate embeddings using OpenAI v1.0+ API."""
    resp = client.embeddings.create(model=model, input=texts)
    return [data.embedding for data in resp.data]

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text, meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone...")

    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings(texts, model="text-embedding-3-small")

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        index.upsert(vectors)
        time.sleep(0.2)

    print("All items uploaded successfully.")

# -----------------------------
if __name__ == "__main__":
    main()
