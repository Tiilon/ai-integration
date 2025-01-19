from langchain_ollama import ChatOllama
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

llm = ChatOllama(model="llama3.2")

loader = TextLoader("./data/be-good.txt")
loaded_data = loader.load()

# text_splitter = CharacterTextSplitter(
#     separator="\n\n",
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
#     is_separator_regex=False,
# )

# texts = text_splitter.create_documents(
#     [loaded_data[0].page_content]
# )

# metadatas = [{"chunk": 0}, {"chunk": 1}]

# documents = text_splitter.create_documents(
#     [loaded_data[0].page_content, loaded_data[0].page_content], 
#     metadatas=metadatas
# )


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=4
)

texts = text_splitter.create_documents(
    [loaded_data[0].page_content]
)

print(texts[0].page_content)