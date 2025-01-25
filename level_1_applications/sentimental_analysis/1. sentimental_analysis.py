from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

## With less control
# class Classification(BaseModel):
#     sentiment: str = Field(description="The sentiment of the text")
#     political_tendency: str = Field(
#         description="The political tendency of the user"
#     )
#     language: str = Field(description="The language the text is written in")

## With more control
class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    political_tendency: str = Field(
        ...,
        description="The political tendency of the user",
        enum=["conservative", "liberal", "independent"],
    )
    language: str = Field(
        ..., enum=["spanish", "english"]
    )
    
llm = ChatOllama(temperature=0, model="llama3.1:8b").with_structured_output(
    Classification
)

tagging_chain = tagging_prompt | llm

trump_follower = "I'm confident that President Trump's leadership and track record will once again resonate with Americans. His strong stance on economic growth and national security is exactly what our country needs at this pivotal moment. We need to bring back the proven leadership that can make America great again!"

biden_follower = "I believe President Biden's compassionate and steady approach is vital for our nation right now. His commitment to healthcare reform, climate change, and restoring our international alliances is crucial. It's time to continue the progress and ensure a future that benefits all Americans."

response = tagging_chain.invoke({"input": trump_follower})
response2 = tagging_chain.invoke({"input": biden_follower})

print(response)
print(response2)
