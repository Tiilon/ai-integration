from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

prompt1 = ChatPromptTemplate.from_template("what is the country {politician} is from?")
prompt2 = ChatPromptTemplate.from_template(
    "what continent is the country {country} in? respond in {language}"
)

model = ChatOllama(model="llama3.2")

chain1 = prompt1 | model | StrOutputParser()

chain2 = (
    {"country": chain1, "language": itemgetter("language")}
    | prompt2
    | model
    | StrOutputParser()
)

response = chain2.invoke({"politician": "Miterrand", "language": "French"})
print(response)