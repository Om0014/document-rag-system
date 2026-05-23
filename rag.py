from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama

# Load vector DB
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

def ask_question(query):

    # Retrieve docs
    docs = vectorstore.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer ONLY using the provided context.

    If answer is not in context, say:
    "The documents do not contain enough information."

    Context:
    {context}

    Question:
    {query}
    """

    response = ollama.chat(
        model='phi3',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    answer = response['message']['content']

    citations = []

    for doc in docs:
        citations.append({
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "snippet": doc.page_content[:200]
        })

    return {
        "answer": answer,
        "citations": citations
    }