#load pdf
#split into chunks
#create the embeddings
#store into chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

#loading the pdf file
data = PyPDFLoader("documents loaders/cr7biography.pdf")
docs = data.load()

#creating chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(data)

#creating the embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model,
    persist_directory = "Chroma_db"
)


