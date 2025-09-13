from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain_openai import ChatOpenAI


def augment_prompt(vectorstore, query: str):
    results = vectorstore.similarity_search(query, k=5)
    
    context_with_ids = []
    for i, result in enumerate(results, 1):
        doc_id = result.metadata.get('id', 'Unknown')
        context_with_ids.append(f"[Source {i} - Document ID: {doc_id}]\n{result.page_content}")
    
    source_knowledge = "\n\n".join(context_with_ids)
    
    augmented_prompt = f"""Using the contexts below, answer the query. When citing information, reference the specific Document ID shown in brackets.

Contexts:
{source_knowledge}

Query: {query}"""
    return augmented_prompt

def create_messages():
    messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?")
]
    return messages

def instantiate_chat_model(model_name="gpt-4o-mini", temperature = 0, api_key = api_key):
    chat = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=api_key)
    return chat

def invoke_chat(chat, messages):
    query = input("Enter your question: ")
    prompt = HumanMessage(content=augment_prompt(augment_prompt(query)))
    messages.append(prompt)
    res = chat(messages)
    messages.append(res)
    print(res.content)
    return None