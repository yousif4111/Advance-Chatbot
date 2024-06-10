# initing the VectorDB
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate


import os
import io
from dotenv import load_dotenv
load_dotenv()

persistent_client = chromadb.PersistentClient(path="./chromadb_dir")


embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"),
                              model="text-embedding-ada-002")
db3 = Chroma(client=persistent_client,
             collection_name = 'test_4.pdf',
             embedding_function=embeddings)


model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")




# Prompt -- for System message
template ="""You are an AI engaged in a conversation with a human.
Strictly follow the instructions here:

1: Use conversation history and the context below to answer the question:
##
{context}
##

2: If you are unable to provide an answer based on the context or our previous conversation, respond with the following message:
##
I can't answer based on the context or our previous conversation.
##

3: Format the answer well in Markdown.

Answer:"""

# List of Messages -- * This List act as the memory for the conversation as well as input for prompt*
chat_list = [
    ("system", template),
    # ("human", "Hello, how are you doing?"),
    # ("ai", "I'm doing well, thanks!"),

]

def chat_f(question):
    # Adding context and question to messages list
    chat_list.append(("human",question))
    # Adding context and question to messages list
    relevant_docs = db3.similarity_search(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    chat_template = ChatPromptTemplate.from_messages(chat_list)
    messages = chat_template.format_messages(
        context=context,
    )

    buffer = io.StringIO()
    print("\n")
    for chunk in model.stream(messages):
        print(chunk.content, end="", flush=True)
        buffer.write(chunk.content)
    
    # Adding the Response to the message list:
    chat_list.append(("ai",buffer.getvalue()))
    print("\n\n")

while True:
    user_input = input("\nYou: ")
    
    
    if "quit" in user_input:
        break

    chat_f(user_input)