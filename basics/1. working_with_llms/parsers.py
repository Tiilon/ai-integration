from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser

from pydantic import BaseModel, Field

llm = ChatOllama(model="llama3.2", temperature=0.9)


# parsing outputs with SimpleJsonOutputParser
# json_prompt = PromptTemplate.from_template(
#     "Return a JSON object with a single key called 'answer' containing the answer to the question: {question}"
# )

# output_parser = SimpleJsonOutputParser()


# json_chain = json_prompt | llm | output_parser

# print(json_chain.invoke({"question": "What can you tell me about docker?"}))


# output_parser = PydanticOutputParser(pydantic_object=Answer)
class Answer(BaseModel):
    answer: str = Field(description="The answer to the question")
    
py_prompt = PromptTemplate.from_template(
    "You are a tech expert that can answer questions about technology. Return a JSON object with a single key called 'answer' containing the answer to the question {question}"
)

output_parser = PydanticOutputParser(pydantic_object=Answer)

pydantic_chain = py_prompt | llm | output_parser

print(pydantic_chain.invoke({"question": "What can you tell me about docker?"}))


