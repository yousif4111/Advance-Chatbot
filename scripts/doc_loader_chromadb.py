# Script contain function that uses langchain and Unstructured.io to upload document to collection in chromadb
# Author: Yousif

#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
#±±±±± Instructured to install unstructured io ±±±±±±±±
#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
# # Install package
# pip install "unstructured[all-docs]"
# # Install other dependencies
# # https://github.com/Unstructured-IO/unstructured/blob/main/docs/source/installing.rst
# !brew install libmagic
# !brew install poppler
# !brew install tesseract
# # If parsing xml / html documents:
# !brew install libxml2
# !brew install libxslt



import pandas as pd
from langchain.document_loaders import UnstructuredFileLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
import uuid
from dotenv import load_dotenv
from tqdm import tqdm
import os
load_dotenv()
# Preparing embedding models for embedding the chroma 

# openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#                 api_key=os.getenv("OPENAI_API_KEY"),
#                 model_name="text-embedding-ada-002",
                
#             )

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1200,
    chunk_overlap  = 100,
    length_function = len,
    is_separator_regex = False,
)

# client = chromadb.PersistentClient(path="./chromadb_dir")
# client.heartbeat()



# ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
# ±±±±±±± Creating new collection ±±±±±±±
# ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±

def create_collection(client: chromadb.Client, name: str,emb_fn, **kwargs):
    """Creates a new collection in ChromaDB.

    Args:
        client: A ChromaDB client.
        name: The name of the new collection.
        emb_fn: the Embeding function will be used to extract the embeddings
        **kwargs: Additional keyword arguments for metadata in the collection.

    Returns:
        A ChromaDB collection object.
    """
    try:
        client.create_collection(name=name, embedding_function=emb_fn, metadata=kwargs)
        message = f"The {name} collection has been successfully created!"
        # print(message)
        return message
    except:
        message = f"Unfortunately, something went wrong, and the collection was not created.\nCheck if all input are correct"
        # print(message)
        return message



    



#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
#±±±±± Function to upload a File into particular Collection in VecDB ±±±±±±±
#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±

def upload_to_collection_chromadb(client,coll_n,doc_lo,emb_fn,strategy="auto"):

    # ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
    # ±±± This Sectoin to check file type and extension ±±±
    # ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±

    file_path, file_extension = os.path.splitext(doc_lo)
    file_extension = file_extension.lower()

    file_name = file_path.split("/")[-1]
    # Support file types
    sup_type = [".docx",".pdf",".txt"]

    if file_extension not in sup_type:
        return "file type not supported yet"


    # ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
    # ±±±±±±± Assigning and checking the collection ±±±±±±±
    # ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
    try:
        collection = client.get_collection(
                    name=coll_n,
                    embedding_function=emb_fn)
    except ValueError:
        coll_l = [i.name for i in client.list_collections()]
        message = "Oops, it seems the collection doesn't exist. Please choose a collection from the list below or create a new one."
        # print(f"{message}\n\n{coll_l}")
        return f"{message}\n\n{coll_l}"
        # return message, coll_l


    # ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
    # ±±±±±±± Loading the file using Unstructured and text_splitter ±±±±±±±
    # ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
    loader = UnstructuredFileLoader(file_path=doc_lo, mode="paged",post_processors=[clean_extra_whitespace],strategy=strategy)
    docs_1 = loader.load()

    # text splitter section
    docs=[]
    for i in docs_1:
        docs.extend(text_splitter.create_documents([i.page_content],[i.metadata]))


    # ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
    # ±±±±±±± Extracting the embeddings and Indexing into the VecDB ±±±±±±±
    # ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±

    if file_extension == ".pdf":
        # for i in docs:
        for i in tqdm(docs, desc="Uploading and Indexing the Document",ncols=80, ascii=True):
            # if i.metadata["links"]:
            if "links" in i.metadata:
                # print("all found")
                collection.add(
                    documents=[i.page_content],
                    metadatas=[{"filename": i.metadata["filename"],
                                "filetype": i.metadata["filetype"],
                                "page_number":i.metadata["page_number"],
                                "links":i.metadata["links"][0]["url"]}],
                    ids=[str(uuid.uuid4().int)]
                    )
            else:
                # print("not found")
                collection.add(
                    documents=[i.page_content],
                    metadatas=[{"filename": i.metadata["filename"],
                                "filetype": i.metadata["filetype"],
                                "page_number":i.metadata["page_number"]}],
                    ids=[str(uuid.uuid4().int)]
                    )
    else:
        # for i in docs:
        for i in tqdm(docs, desc="Uploading and Indexing the Document",ncols=80, ascii=True):
            # print("not found")
            collection.add(
                documents=[i.page_content],
                metadatas=[{"filename": i.metadata["filename"],
                            "filetype": i.metadata["filetype"],
                            # "page_number":i.metadata["page_number"]
                            }],
                ids=[str(uuid.uuid4().int)]
                )

    message = f"The file {file_name} has been successfully uploaded to the database!"
    # print(f"\n{message}")
    return f"\n{message}"





    


    # ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
    # ±±±±±±± Delete existing collection ±±±±±±±
    # ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
def delete_collection(client,coll_na):
    try:
        collection = client.get_collection(name=coll_na)
    except ValueError:
        message = "The collection doesn't exist."
        return message
    try:
        client.delete_collection(name=coll_na)
        message = "The collection has been successfully deleted."
        return message
    except:
        message = "Oops, something went wrong; the collection might not have been deleted."
        return message






## Addition Functions Section ##

#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
#±±±±± Function to upload to db and Creating Unique collection for each File ±±±±±±±
#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±

# note function can only read pdf
def upload_unique_collection_chromadb(doc_lo,strategy="auto"):

    #±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
    #±±±±± Ingesting the file section ±±±±±±±
    #±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
    # Unstructured io section
    loader = UnstructuredFileLoader(file_path=doc_lo, mode="paged",post_processors=[clean_extra_whitespace],strategy=strategy)
    docs_1 = loader.load()

    # text splitter section
    docs=[]
    for i in docs_1:
        docs.extend(text_splitter.create_documents([i.page_content],[i.metadata]))
    
    #±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
    #±±±±± Iinitating chroma collection ±±±±±±±
    #±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±

    # Creating the collection
    collection = client.get_or_create_collection(
                    name=docs[0].metadata["filename"],
                    embedding_function=openai_ef,
                    metadata={
                        "filetype":docs[0].metadata["filetype"],
                        "page_numbers":docs[-1].metadata["page_number"]})
    
    # Filling the collection with documents
    # for i in docs:
    for i in tqdm(docs, desc="Uploading and Indexing the Document",ncols=80, ascii=True):
        if i.metadata["links"]:
            # print("all found")
            collection.add(
                documents=[i.page_content],
                metadatas=[{"filename": i.metadata["filename"],
                            "filetype": i.metadata["filetype"],
                            "page_number":i.metadata["page_number"],
                            "links":i.metadata["links"][0]["url"]}],
                ids=[str(uuid.uuid4().int)]
                )
        else:
            # print("not found")
            collection.add(
                documents=[i.page_content],
                metadatas=[{"filename": i.metadata["filename"],
                            "filetype": i.metadata["filetype"],
                            "page_number":i.metadata["page_number"]}],
                ids=[str(uuid.uuid4().int)]
                )
    print("File has been added successfully to the db!!")




## Additional function ##

#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
#±±±±± Function to upload to db and Creating Unique collection for each File ±±±±±±±
#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±

# note function can only read pdf
def upload_unique_collection_chromadb(client,doc_lo,strategy="auto"):

    #±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
    #±±±±± Ingesting the file section ±±±±±±±
    #±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
    # Unstructured io section
    loader = UnstructuredFileLoader(file_path=doc_lo, mode="paged",post_processors=[clean_extra_whitespace],strategy=strategy)
    docs_1 = loader.load()

    # text splitter section
    docs=[]
    for i in docs_1:
        docs.extend(text_splitter.create_documents([i.page_content],[i.metadata]))
    
    #±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
    #±±±±± Iinitating chroma collection ±±±±±±±
    #±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±

    # Creating the collection
    collection = client.get_or_create_collection(
                    name=docs[0].metadata["filename"],
                    embedding_function=openai_ef,
                    metadata={
                        "filetype":docs[0].metadata["filetype"],
                        "page_numbers":docs[-1].metadata["page_number"]})
    
    # Filling the collection with documents
    # for i in docs:
    for i in tqdm(docs, desc="Uploading and Indexing the Document",ncols=80, ascii=True):
        if i.metadata["links"]:
            # print("all found")
            collection.add(
                documents=[i.page_content],
                metadatas=[{"filename": i.metadata["filename"],
                            "filetype": i.metadata["filetype"],
                            "page_number":i.metadata["page_number"],
                            "links":i.metadata["links"][0]["url"]}],
                ids=[str(uuid.uuid4().int)]
                )
        else:
            # print("not found")
            collection.add(
                documents=[i.page_content],
                metadatas=[{"filename": i.metadata["filename"],
                            "filetype": i.metadata["filetype"],
                            "page_number":i.metadata["page_number"]}],
                ids=[str(uuid.uuid4().int)]
                )
    print("File has been added successfully to the db!!")