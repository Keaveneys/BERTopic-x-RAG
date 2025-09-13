from umap import UMAP
from bertopic import BERTopic
import numpy as np
from hdbscan import HDBSCAN
from openai import OpenAI
import os
import pandas as pd

def dimensionality_reduction(n_components = 5, min_dist = 0.1, metric = 'cosine', random_state = 42):
    umap_model = UMAP(n_components = n_components
                      , min_dist = min_dist
                      , metric = metric
                      , random_state = randome_state)
    return umap_model 


def clustering_model(min_cluster_size = 15, metric = 'euclidean', cluster_selection_method = 'eom'):
    hdbscan_model = HDBSCAN(min_cluster_size = min_cluster_size
                            , metric = metric
                            , cluster_selection_method = cluster_selection_method)
    return hdbscan_model

def topic_model(embedding_model, umap_model, hdbscan_model, abstracts, embeddings):
    topic_model = BERTopic(
    embedding_model = embedding_model,
    umap_model = umap_model,
    hdbscan_model = hdbscan_model,
    verbose = True 
).fit(abstracts, np.array(embeddings))
    return topic_model 

client = OpenAI(api_key=api_key)

# Custom function to get topic labels
def get_topic_label(keywords, documents_sample):
    prompt = f"""
    I have a topic described by these keywords: {', '.join(keywords)}
    
    Sample documents from this topic:
    {documents_sample[:500]}...
    
    Create a short, descriptive topic label (2-4 words):
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

def label_topics(topic_model, documents):
    # CREATE A DICTIONARY to store the labels
    openai_labels = {}
    # Apply labels manually after BERTopic fitting
    topic_info = topic_model.get_topic_info()
    for topic_id in topic_info['Topic']:
        if topic_id != -1:
            topic_keywords = [word for word, _ in topic_model.get_topic(topic_id)[:5]]
        # Get sample documents for this topic
        topic_docs = [doc for doc, topic in zip(documents, topic_model.topics_) if topic == topic_id][:3]
        label = get_topic_label(topic_keywords, ' '.join(topic_docs))
        
        # STORE the label in the dictionary
        openai_labels[topic_id] = label
        print(f"Topic {topic_id}: {label}")
    return openai_labels

def create_topic_df(topic_model, documents, openai_labels):
    mapped_df = pd.DataFrame(topic_model.get_document_info(docs = documents))
    # Map the OpenAI labels instead of the default BERTopic names
    mapped_df['Name'] = mapped_df['Topic'].map(openai_labels)
    return mapped_df


