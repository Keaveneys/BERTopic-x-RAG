from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from langchain.document_loaders import DataFrameLoader


def construct_page_content(dataframe,topic_name_column, topic_number_column,id_column, chunk_column = 'chunk'):
    dataframe['page_content'] = "Topic: " + dataframe[topic_number_column].astype(str) + ": " + dataframe[topic_name_column].astype(str) + "\n\n" + \
    "ID: " + dataframe[id_column].astype(str) + "\n\n" + "Chunk: " + dataframe[chunk_column]

    return dataframe

def split_text(dataframe, text_column, id_column, topic_name_column, topic_number_column, chunk_size=1200, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,  # Maximum size of each chunk
    chunk_overlap=chunk_overlap  # Overlap between chunks
)
    split_documents = []
    for index, row in dataframe.iterrows():
        text = row[text_column]
        id = row[id_column]
        topic = row[topic_name_column]
        topic_id = row[topic_number_column]
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            split_documents.append({"id": id, "topic": (f"{topic_id}: {topic}"), "chunk": chunk})
        
    chunked_df = pd.DataFrame(split_documents)
    chunked_df['topic'] = chunked_df['topic'].replace('-1: nan', 'Unallocated')
    chunked_df = construct_page_content(chunked_df, topic_name_column, topic_number_column, id_column, chunk_column="chunk")
    return chunked_df

def document_loader(dataframe, page_content_column):
    loader = DataFrameLoader(
    dataframe,
    page_content_column = page_content_column
)
    docs = loader.load()
    return docs 



