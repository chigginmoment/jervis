import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings  # Change if using another model
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Note: All this has to be the same during retrieval
chroma_db_path = "chroma_db"
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_function)

directory = "pages/"
documents = []

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filepath = os.path.join(directory, filename)
        loader = TextLoader(filepath, encoding="utf-8")  # Load document
        documents.extend(loader.load())  # Append to list

# we dont need text splitter as we should store whole documents
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = text_splitter.split_documents(documents)

# Add to ChromaDB
vectorstore.add_documents(documents)
vectorstore.persist()

print("Documents successfully added to ChromaDB!")