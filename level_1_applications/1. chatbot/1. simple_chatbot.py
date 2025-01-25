from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain.memory import FileChatMessageHistory
from langchain._api import LangChainDeprecationWarning
from langchain.chains import LLMChain
import warnings
from langchain._api import LangChainDeprecationWarning

warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

chatbot = ChatOllama(model="llama3.2", temperature=0.9)

memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chatbot,
    prompt=prompt,
    memory=memory
)

# response = chain.invoke("hello!")
# response = chain.invoke("my name is Julio")
response = chain.invoke("what is my name?")

print(response)