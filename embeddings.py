from langchain_openai import OpenAIEmbeddings


def instantiate_embedding_model(model="text-embedding-3-small", api_key=api_key):
    embedding_model = OpenAIEmbeddings(
    model = "text-embedding-3-small"
    , api_key=api_key
    )
    return embedding_model 

def embed_documents(embedding_model, documents):
    embeddings = embedding_model.embed_documents(documents)
    len(embeddings), len(embeddings[0])
    print(f"This model has {len(embeddings[0])} embedding dimensions.")
    return embeddings 

