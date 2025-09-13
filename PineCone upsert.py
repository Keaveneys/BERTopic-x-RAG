from pinecone import Pinecone, ServerlessSpec
import time
from tqdm import tqdm
import numpy as np
from langchain_pinecone import PineconeVectorStore


def initialize_pinecone(pinecone_api_key):
    pc = Pinecone(api_key = api_key)
    return pc 

# Initialize Pinecone
def init_index(pc, index_name, dimension, metric='cosine'):
    if index_name not in pc.list_indexes():
        spec = ServerlessSpec(
        cloud="aws",  # or "gcp"
        region="us-east-1"  # or your preferred region
    )
    pc.create_index(
        name=index_name,
        dimension=1536,  # number of dimensions for our embedding model
        metric="cosine",  # metric for search
        spec=spec
    )
    # if index not ready, wait for two seconds
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(2)
    
    index = pc.Index(index_name)
    return index 


def batch_upsert(index, embedding_model, docs, id_column, batch_size=100):
    batch_size = batch_size  # Adjust based on your needs and Pinecone limits
    # Your docs are already ready - just embed the page_content and upsert
    for i in tqdm(range(0, len(docs), batch_size)):
        batch = docs[i:i+batch_size]
        texts = [doc.page_content for doc in batch]  # This includes topic context!
        embeds = embedding_model.embed_documents(texts)
        vectors = [(
            str(doc.metadata[id_column]),  # Simple ID
            embed,
            doc.metadata  # Includes id, topic, chunk
            ) for doc, embed in zip(batch, embeds)]
    
    index.upsert(vectors=vectors)
    
    # Rate limiting to avoid hitting Pinecone free tier limits
    time.sleep(1)  # 1 second delay between batches
    # Get basic index statistics
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats['total_vector_count']}")
    print(f"Index dimension: {stats['dimension']}")
    print(f"Namespaces: {stats.get('namespaces', {})}")
    # Get some random vectors to see what's there
    random_vector = np.random.random(1536).tolist()  # Match your embedding dimension
    results = index.query(vector=random_vector, top_k=10, include_metadata=True)
    print("Sample of uploaded documents:")
    for match in results['matches']:
        topic = match['metadata'].get('topic', 'Unknown')
        print(f"ID: {match['id']}, Topic: {topic}")
    return None 

def query_index(text_field, embedding_model, index):
    vectorstore = PineconeVectorStore(
    index,
    embedding = embedding_model, 
    text_key = text_field)
    return vectorstore 
