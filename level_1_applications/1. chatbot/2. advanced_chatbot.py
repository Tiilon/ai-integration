from langchain_ollama import ChatOllama
import warnings
from langchain._api import LangChainDeprecationWarning
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

chatbot = ChatOllama(model="llama3.2", temperature=0.9)

chatbotMemory = {}

# input: session_id, output: chatbotMemory[session_id]
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chatbotMemory:
        chatbotMemory[session_id] = ChatMessageHistory()
    return chatbotMemory[session_id]


chatbot_with_message_history = RunnableWithMessageHistory(
    chatbot, 
    get_session_history,
)

session1 = {"configurable": {"session_id": "001"}}

# responseFromChatbot = chatbot_with_message_history.invoke(
#     [HumanMessage(content="My favorite color is red.")],
#     config=session1,
# )
# print(responseFromChatbot.content)

# responseFromChatbot = chatbot_with_message_history.invoke(
#     [HumanMessage(content="What's my favorite color?")],
#     config=session1,
# )

# print(responseFromChatbot.content)


# Limiting the number of messages to be sent to the LLM to manage context
def limited_memory_of_messages(messages, number_of_messages_to_keep=2):
    return messages[-number_of_messages_to_keep:]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

limitedMemoryChain = (
    RunnablePassthrough.assign(messages=lambda x: limited_memory_of_messages(x["messages"]))
    | prompt 
    | chatbot
)

chatbot_with_limited_message_history = RunnableWithMessageHistory(
    limitedMemoryChain, # type: ignore
    get_session_history,
    input_messages_key="messages",
)

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite vehicles are Vespa scooters.")],
    config=session1, # type: ignore
)

responseFromChatbot.content

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite city is San Francisco.")],
    config=session1, # type: ignore
)

responseFromChatbot.content