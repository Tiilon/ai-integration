from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate

llm = ChatOllama(model="llama3.2", temperature=0.9)


# using PromptTemplate
# prompt = PromptTemplate.from_template(
#     "You are a tech expert that can answer questions about technology. {input}"
# )

# print(prompt.format(input = "What can you tell me about docker?"))
# print(prompt.invoke({"input": "What can you tell me about docker?"}))\


# using ChatPromptTemplate
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an {profession} expert on {topic}."),
        ("human", "Hello, Mr. {profession}, can you please answer a question?"),
        ("ai", "Sure!"),
        ("human", "{user_input}"),
    ]
)

prompt = chat_template.format_messages(
    profession="tech", topic="docker", user_input="What can you tell me about docker?"
)

print(llm.invoke(prompt))