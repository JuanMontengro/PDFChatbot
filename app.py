from flask import Flask
from flask import request
import pandas as pd
import pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import openai
from PyPDF2 import PdfReader
import itertools
import uuid

app = Flask(__name__)

pinecone_index_name = "INDEX NAME "
openai.api_key = "sk-..."

def initialize_pinecone():
    PINECONE_API_KEY = 'PINECONE APIKEY'
    pinecone.init(api_key=PINECONE_API_KEY,environment='ENVIRONMENT')
    
def delete_existing_pinecone_index():
    if pinecone_index_name in pinecone.list_indexes():
        pinecone.delete_index(pinecone_index_name)

def create_pinecone_index(index_name):
    pinecone.create_index(name=index_name, dimension=1536, metric="cosine", shards=1)
    pinecone_index = pinecone.Index(name=index_name)
    return pinecone_index

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 800,
        chunk_overlap=200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # create embeddings before loading into a vector store/knowledge base
    embeddings = OpenAIEmbeddings(openai_api_key="sk-...")
    # we'll use FAISS as our vector store.
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def split_text_into_chunks(text, chunk_size=100):
    words = text.split()
    text_chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        text_chunks.append(chunk)
    return text_chunks


def upsert_pdf_contents(user_id, files):
    # get raw pdf text
    raw_text = get_pdf_text(files)
    # segment raw pdf text into chunks
    # text_chunks = get_text_chunks(raw_text)
    text_chunks = split_text_into_chunks(raw_text)
    embeddings = []
    for chunk in text_chunks:
        id = uuid.uuid4().hex
        embedding = get_embedding(chunk)
        embeddings.append([(id, embedding, {"text": chunk})])
    sp_name = 'namespace_'+user_id
  
    for embedding in embeddings:
        pinecone_index.upsert(vectors=embedding, namespace=sp_name)
    return pinecone_index.describe_index_stats()

def get_embedding(chunk):
    """Get embedding using OpenAI"""
    response = openai.Embedding.create(
        input=chunk,
        model="text-embedding-ada-002",
    )
    embedding = response['data'][0]['embedding']
    return embedding

def get_response_from_openai(query, documents):
    """Get ChatGPT api response"""
    prompt = get_prompt_for_query(query, documents)
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=0,
        max_tokens=800,
        top_p=1,
    )
    return response["choices"][0]["message"]["content"]

def get_prompt_for_query(query, documents):
    """Build prompt for question answering"""
    template = """
    You are given a paragraph and a query. You need to answer the query on the basis of paragraph. If the answer is not contained within the text below, say \"Sorry, I don't know. Please try again.\"\n\nP:{documents}\nQ: {query}\nA:
    """
    final_prompt = template.format(
        documents=documents,
        query=query
    )
    return final_prompt

def search_for_query(user_id, query):
    """Main function to search answer for query"""
    query_embedding = get_embedding(query)
    query_response = pinecone_index.query(
        namespace="namespace_"+user_id,
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    
    documents = [
        match['metadata']['text'] for match in query_response['matches']
    ]
    documents_as_str = "\n".join(documents)
    response = get_response_from_openai(query, documents_as_str)
    return response

def remove_pdfs(user_id):
    # Get the Pinecone Index object for the target namespace
    sp_name = 'namespace_'+user_id
    # Delete all vectors from the target namespace
    delete_response = pinecone_index.delete(delete_all=True, namespace=sp_name)
    return delete_response

initialize_pinecone()
pinecone_index = pinecone.Index("125")

@app.route("/upload", methods=["POST"])
def upload_train():
    user_id = request.form['user_id']
    files = request.files.getlist('files')
    upsert_pdf_contents(user_id, files)
    return "training success"

@app.route("/query", methods=["POST"])
def query():
    response=search_for_query(request.form['user_id'], request.form['question'])
    return response

@app.route("/remove", methods=["POST"])
def remove():
    response=remove_pdfs(request.form['user_id'])
    return "removing success"

if __name__ == '__main__':
  app.run(debug=True)