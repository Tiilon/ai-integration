from operator import itemgetter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

llm = ChatOllama(model="llama3.2:3b", temperature=0.9)


# Runnable Passthrough does not modify the input and returns the same input as output
# chain = RunnablePassthrough()
# response = chain.invoke("What is the capital of the US?")
# print(response)


def capitalise(text):
    return text.upper()

# Runnable Lambda
# # chain = RunnableLambda(capitalise)
# chain = RunnablePassthrough() | RunnableLambda(capitalise)
# response = chain.invoke("sparrow")
# print(response)


# Runnable Parallel - runs multiple runnables in parallel(at the same time)
## Example 1
# chain = RunnableParallel({
#     "operation_a": RunnablePassthrough(),
#     "operation_b": RunnableLambda(capitalise)
# })
# response =chain.invoke("sparrow")
# print(response)

## Example 2
# chain = RunnableParallel({
#     "operation_a": RunnablePassthrough(),
#     "operation_b": lambda x: x['name'].upper(),
#     "operation_c": lambda x: x['age'] + x['incremental']
# })
# response = chain.invoke({"name":"sparrow","age":10,'incremental': 2})
# print(response)

# Example 3
# template = "tell me a curious fact about {soccer_player}"

# def full_name(person):
#     return f'Cristiano {person['name']}'

# prompt = ChatPromptTemplate.from_template(template)
# output_parser = StrOutputParser()

# runnable = RunnableParallel({
#     "operation_a": RunnablePassthrough(),
#     "soccer_player": RunnableLambda(full_name),
#     "operation_c": RunnablePassthrough(),
# }) | prompt | llm | output_parser

# response = runnable.invoke({
#     "name": "Ronaldo",
#     "age": 39
# })
# print(response)


#Example 4
embeddings = OllamaEmbeddings(model="llama3.2:3b")
vectorstore = Chroma.from_texts(
    ["AI Accelera has trained more than 10,000 Alumni from all continents and top companies"], embedding=embeddings
)
retriever = vectorstore.as_retriever()

template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
    "context": retriever,
    "question": RunnablePassthrough(),
    } 
    | prompt 
    | llm 
    | StrOutputParser()
)

response = chain.invoke("who are the Alumni of AI Accelera?")
print(response)