from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

llm = ChatOllama(model="llama3.2")

embeddings = OllamaEmbeddings(
    model="llama3.1:8b",
)

chunks_of_text =     [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]

embeddings_doc = embeddings.embed_documents(chunks_of_text)

embedded_query = embeddings.embed_query("What was the name mentioned in the conversation?")
print(len(embedded_query))