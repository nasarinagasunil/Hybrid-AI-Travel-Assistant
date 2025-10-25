import json
import time
from tqdm import tqdm
import google.generativeai as genai
# --- Use the modern Pinecone client ---
from pinecone import Pinecone
import config

# --- Config is the same ---
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32
INDEX_NAME = config.PINECONE_INDEX_NAME

# --- Initialize clients (Modern V3+ Syntax) ---
genai.configure(api_key=config.GOOGLE_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# --- Connect to index using the required host parameter ---
print(f"Connecting to existing index: {INDEX_NAME}")
index = pc.Index(
    host="https://vietnam-travel-google-5308776.svc.aped-4627-b74a.pinecone.io"
)

# --- Helper functions are the same ---
def get_embeddings(texts, model="models/text-embedding-004"):
    try:
        response = genai.embed_content(model=model, content=texts)
        return response['embedding']
    except Exception as e:
        print(f"An error occurred during embedding: {e}")
        return [[] for _ in texts]

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

# --- Main upload function is mostly the same ---
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
        embeddings = get_embeddings(texts)

        if any(not emb for emb in embeddings):
            print(f"Skipping a batch due to an embedding error.")
            continue

        # --- V3+ upsert uses a list of dictionaries ---
        vectors_to_upsert = [
            {'id': _id, 'values': emb, 'metadata': meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]
        index.upsert(vectors=vectors_to_upsert)
        time.sleep(0.2)

    print("All items uploaded successfully.")

if __name__ == "__main__":
    main()