# this is for temporary memory
from langchain_ollama import ChatOllama
from langchain.memory import ChatMessageHistory

llm = ChatOllama(model="llama3.2", temperature=0.9)

history = ChatMessageHistory()

history.add_user_message("hi!")

history.add_ai_message("whats up?")

my_chat_memory = history.messages

print("\n----------\n")

print("Chat Memory:")

print("\n----------\n")
print(my_chat_memory)

print("\n----------\n")