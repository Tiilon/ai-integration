from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


#load chat model
llm = ChatOllama(model="llama3.2")

# load document
loader = TextLoader("./data/state_of_the_union.txt",encoding='utf-8')
loaded_doc = loader.load()

# create text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# # split document
chunks_of_text = text_splitter.split_documents(loaded_doc)

embeddings = OllamaEmbeddings(
    model="llama3.1:8b",
)

vector_db = Chroma.from_documents(chunks_of_text,embeddings)

retriever = vector_db.as_retriever()

question = "what did he say about ketanji brown jackson?"

response = retriever.invoke(question)

print(response[0].page_content)