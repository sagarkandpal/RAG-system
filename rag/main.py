from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

#creating the embeddings 
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#creatitng the vector store
vector_store = Chroma(
    embedding_function = embedding_model,
    persist_directory = "Chroma_db"
)

#creating the retriever 
retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k":3,
        "fetch_k":10,
        "lambda_mult": 0.5
    }
)

llm = ChatMistralAI(model = "mistral-small-2506")

#promt template
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are an intelligent AI assistant.

Your job is to answer the user's question using the provided context from the document.

Guidelines:
- Carefully analyze the context, do not rely only on exact keyword matches.
- You are allowed to infer, summarize, and connect information from multiple parts of the context.
- If the answer is partially available, provide the best possible answer using available information.
- Keep answers clear, concise, and relevant.

STRICT RULE:
- Do NOT use outside knowledge.
- If the answer is completely not present in the context, then say:
  "sorry bhai iska answer iss document me nahi mila."

Always prioritize understanding over keyword matching

     """),
    (
        "human", 
        """Context: {context}

        Question: {question}
        """)
])

print("Rag system is created")

print("press 0 to exit")

while True:
    query = input("\nSagar: ")
    if query == "over":
        break

    docs = retriever.invoke(query)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print(f"\nRonaldo: {response.content}")