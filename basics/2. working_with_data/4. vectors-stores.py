from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


#load chat model
llm = ChatOllama(model="llama3.2")

# load document
loader = TextLoader("./data/state_of_the_union.txt")
loaded_doc = loader.load()

# create text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# # split document
chunks_of_text = text_splitter.split_documents(loaded_doc)

embeddings = OllamaEmbeddings(
    model="llama3.1:8b",
)
print(chunks_of_text[0])
# embeddings_doc = embeddings.embed_documents(chunks_of_text)