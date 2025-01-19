from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader, CSVLoader,BSHTMLLoader,PyPDFLoader,WikipediaLoader
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

llm = ChatOllama(model="llama3.2")

#Using TextLoader for loading text files
# loader = TextLoader("./data/be-good.txt")
# loaded_data = loader.load()


#Using CSVLoader for loading CSV files
# loader = CSVLoader("./data/Street_Tree_List.csv")
# loaded_data = loader.load()

#Using BSHTMLLoader for loading PDF files
# loader = BSHTMLLoader("./data/100-startups.html")
# loaded_data = loader.load()

#Using PyPDFLoader for loading PDF files
# loader = PyPDFLoader("./data/5pages.pdf")
# loaded_data = loader.load()


# print(loaded_data[0].page_content)
# print("##################################################")
# print(loaded_data)


#Using WikipediaLoader
# loader = WikipediaLoader(query="JFK", load_max_docs=3)
# loaded_data = loader.load()[0].page_content
# chat_template = ChatPromptTemplate.from_messages(
#     [
#         ("human", "Answer this {question}, here is some extra {context}"),
#     ]
# )

# messages = chat_template.format_messages(
#     question="Where was JFK born?",
#     context=loaded_data
# )
# response = llm.invoke(messages)

# print(response)