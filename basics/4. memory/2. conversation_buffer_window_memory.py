# this is for temporary memory

from langchain_ollama import ChatOllama
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain


llm = ChatOllama(model="llama3.2", temperature=0.9)

# the chat bot will remember the last 3 messages
window_memory = ConversationBufferWindowMemory(k=3)

conversation_window = ConversationChain(
    llm=llm, 
    memory = window_memory,
    verbose=True
)

conversation_window({"input": "Hi, my name is Julio"})
conversation_window({"input": "My favorite color is blue"})
conversation_window({"input": "My favorite animals are dogs"})
conversation_window({"input": "I like to drive a vespa scooter in the city"})
conversation_window({"input": "My favorite city is San Francisco"})
conversation_window({"input": "My favorite season is summer"})
conversation_window({"input": "What is my favorite color?"})
conversation_window({"input": "My favorite city is San Francisco"})

