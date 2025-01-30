from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

llm = ChatOllama(model="llama3.2", temperature=0.9)
embeddings = OllamaEmbeddings(model="llama3.2")

documents = [
    Document(
        page_content="John F. Kennedy served as the 35th president of the United States from 1961 until his assassination in 1963.",
        metadata={"source": "us-presidents-doc"},
    ),
    Document(
        page_content="Robert F. Kennedy was a key political figure and served as the U.S. Attorney General; he was also assassinated in 1968.",
        metadata={"source": "us-politics-doc"},
    ),
    Document(
        page_content="The Kennedy family is known for their significant influence in American politics and their extensive philanthropic efforts.",
        metadata={"source": "kennedy-family-doc"},
    ),
    Document(
        page_content="Edward M. Kennedy, often known as Ted Kennedy, was a U.S. Senator who played a major role in American legislation over several decades.",
        metadata={"source": "us-senators-doc"},
    ),
    Document(
        page_content="Jacqueline Kennedy Onassis, wife of John F. Kennedy, was an iconic First Lady known for her style, poise, and dedication to cultural and historical preservation.",
        metadata={"source": "first-lady-doc"},
    ),
]

vector_db = Chroma.from_documents(documents, embeddings)

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

# print(retriever.batch(["John", "Robert"]))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

chain = {
    "context": retriever, 
    "question": RunnablePassthrough()
    } | prompt | llm

response = chain.invoke("tell me about Jackie")

print("\n----------\n")

print("tell me about Jackie (simple retriever):")

print("\n----------\n")
print(response.content)

print("\n----------\n")