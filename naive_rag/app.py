import chromadb
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
import os

load_dotenv()

VERTEX_API_KEY = os.getenv("VERTEX_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

#load documents from directory

directory_path = 'C:\Users\LapMaster\advance-rag-tutorials\news_articles'
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents from {directory_path}")

# Split documents into chunks
# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

print(f"Split documents into {len(chunked_documents)} chunks")

def download_huggingface_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",)
    return embeddings

embeddings = download_huggingface_model()


# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"

collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=embeddings
)

# generate embeddings hugging face model
from tqdm import tqdm  # for progress bars

# Function to manually generate embeddings and store in ChromaDB
def generate_and_store_embeddings(documents, embedding_model, collection):
    print("==== Generating embeddings and storing in ChromaDB ====")

    ids = []
    texts = []
    embeddings = []

    for doc in tqdm(documents):
        ids.append(doc["id"])
        texts.append(doc["text"])
        embedding = embedding_model.embed_query(doc["text"])
        embeddings.append(embedding)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings
    )
    print(f"✅ Stored {len(texts)} chunks in ChromaDB with HuggingFace embeddings.")

    print(f"Stored {len(texts)} documents with embeddings in ChromaDB.")



#initializing the llm
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                            google_api_key=GOOGLE_API_KEY , 
                            temperature=0.7)







