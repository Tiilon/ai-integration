# There are 3 ways to execute chains: 
# 1. invoke
# 2. stream
# 3. batch

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3.2", temperature=0.9)

# Invoke - this is the default. llm response is returned immediately
prompt = ChatPromptTemplate.from_template("Tell me one sentence about {politician}.")
chain = prompt | llm

# response = chain.invoke({"politician": "Churchill"})

# print("\n----------\n")

# print("Response with invoke:")

# print("\n----------\n")
# print(response.content)

# print("\n----------\n")
    
# print("\n----------\n")


# Stream - llm response is returned one by one
# print("Response with stream:")

# print("\n----------\n")

# for s in chain.stream({"politician": "F.D. Roosevelt"}):
#     print(s.content, end="", flush=True)
    
# print("\n----------\n")


# Batch - llm response is returned in batches
response = chain.batch([{"politician": "Lenin"}, {"politician": "Stalin"}])

print("\n----------\n")

print("Response with batch:")

print("\n----------\n")
print(response)

print("\n----------\n")