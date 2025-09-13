import embeddings
import bertopic
import Pinecone-upsert
import page_conent
import LLM-integration


client = OpenAI(api_key=api_key)


from datasets import load_dataset
#example usage using the arXiv dataset
datasets = load_dataset("maartengr/arxiv_nlp")["train"]
documents = datasets["Abstracts"]

embedding_model = embeddings.instantiate_embedding_model(api_key=api_key)
embeddings = embeddings.embed_documents(embedding_model, documents)

umap_model = bertopic.dimensionality_reduction()
hdbscan_model = bertopic.clustering_model()
topic_model = bertopic.create_topic_model(embedding_model,
                                          umap_model,
                                          hdbscan_model)

mapped_df = bertopic.label_topics(topic_model, documents)

chunked_df = page_content.split_text(
    mapped_df,
    text_column = 'COLUMN TO BE CHUNKED',
    id_column = 'DOCUMENT ID COLUMN',
    topic_name_column = 'Name',
    topic_number_column = 'Topic',
    chunk_size = 1200,
    chunk_overlap = 100
)

docs = page_content.document_loader(chunked_df,
                                    page_content_column = 'page_content')

pc = Pinecone-upsert.pinecone_initialise_pinecone(pinecone_api_key)

index = Pinecone-upsert.init_index(
    pc,
    index_name = "INSERT INDEX NAME",
    dimension = 1536, #change dimension based on embedding models
    metric = "cosine"
)

"""
Be mindful of upsert limits. Pinecone allows a maximum of 100 vectors per upsert request.
"""

Pinecone-upsert.batch_upsert(
    index,
    embedding_model,
    docs,
    id_column = "ID column"
)

vectorstore = Pinecone-upsert.query_index(
    text_field = 'chunk',
    embedding_model = embedding_model,
    index = index
)

messages = LLM-integration.create_messages()

chat = LLM-integration.instantiate_chat_model(
    api_key,
)

augmented_prompt = LLM-integration.augment_prompt(vectorstore)

conversation = LLM-integration.invoke_chat(
        chat,
        messages)