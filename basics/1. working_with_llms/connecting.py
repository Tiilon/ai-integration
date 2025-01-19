from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0.9)

messages = [
    ('system', 'You are a tech expert that can answer questions about technology.'),
    ('human', 'What can you tell me about docker?'),
]

response = llm.invoke(messages)

print(response)