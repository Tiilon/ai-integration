from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(model="llama3.2", temperature=0.9)
sqlite_db_path = "./data/street_tree_db.sqlite"

db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, 
    corresponding SQL query, and SQL result, 
    answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

write_query = create_sql_query_chain(llm, db)

execute_query = QuerySQLDataBaseTool(db=db)

#chain = write_query | execute_query

#chain.invoke({"question": "List the species of trees that are present in San Francisco"})

# chain = (
#     RunnablePassthrough.assign(query=write_query).assign(
#         result=itemgetter("query") | execute_query
#     )
#     | answer_prompt
#     | llm
#     | StrOutputParser()
# )

chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "List the species of trees that are present in San Francisco"})

print(response)