from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3.2", temperature=0.9)

file_path = "./data/Be_Good.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()

embeddings = OllamaEmbeddings(model="llama3.2")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
# create_stuff_documents_chain - (Takes a List of Documents:,Formats into a Prompt:,Passing to an LLM:,Fit within Context Window:)
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# create_retrieval_chain - (Receiving a User Inquiry:,Use a Retriever to Fetch Documents:,Pass Information to an LLM:,Generating a Response:)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

results = rag_chain.invoke({"input": "What is this article about?"})

print(results["answer"])
