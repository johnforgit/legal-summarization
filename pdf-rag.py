"""
1. Ingest the pdf files
2. Extract text from pdf files and split into small chunks
3. Send the chunks to the embedding model
4. Save the embeddings to a vector database
5. Perform similarity search on the vector database to find similar documents
6. Retrieve the similar documents and present them to the user
7. run pip install -r requirements.txt to install the required packages
"""

import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

doc_path="./data/BOI.pdf"
model="llama3.2"

# Local PDF file uploads
if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("Done loading")
else:
    print("Upload a pdf file")

# preview first page

content=data[0].page_content
print(content[:100])

# Extract text from PDF files and split into small chunks

# split and chunk

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("done splitting..")

print(f"Number of chunks: {len(chunks)}")
print(f"Example chunk : {chunks[0]}")