from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3.2", temperature=0.9)
embeddings = OllamaEmbeddings(model="deepseek-r1")

#load document
loader = TextLoader("./data/be-good.txt",encoding='utf-8')
loaded_doc = loader.load()

# create text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks_of_text = text_splitter.split_documents(loaded_doc)

# create vector database
vector_db = Chroma.from_documents(
    chunks_of_text, embeddings
)

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = {
    "context": retriever | format_docs, 
    "question": RunnablePassthrough()
    } | prompt | llm | StrOutputParser()

response = chain.invoke("What is this article about?")
print(response)