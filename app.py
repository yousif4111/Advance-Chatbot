import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os
load_dotenv()

import sys
# Append the directory to sys.path
sys.path.append("./scripts")
from doc_loader_chromadb import create_collection, upload_to_collection_chromadb, delete_collection
from Chat_openai_yousif import chat_f,memory_t
sys.path.remove("./scripts")

import time

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=os.getenv("EMBEDDINGS_MODEL")
                # model_name="text-embedding-ada-002",
                
            )
chroma_client = chromadb.HttpClient(host='localhost', port=8000)




from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename
from flask_cors import CORS 




app = Flask(__name__)
CORS(app)

@app.route('/create_collection', methods=['POST'])
def createCollection():

    data = request.get_json()

        # Check if the 'name' field is present in the JSON data
    if 'name' not in data:
        return jsonify({'error': 'Name field is missing'}), 400
    
    collection_name = data['name']
    metadata = data["metadata"]

    # create_collection(client = chroma_client,name=collection_name,emb_fn=openai_ef,**metadata)
    return create_collection(client = chroma_client,name=collection_name,emb_fn=openai_ef,**metadata)




@app.route('/upload_to_collection', methods=['POST'])
def uploadToCollection():

    # Get the 'name' field from the request data
    name = request.form.get('name')

    # Check if the 'name' field is present
    if name is None:
        return jsonify({'error': 'Name field is missing'}), 400
        
    file = request.files["file"]
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

   
    filename = secure_filename(file.filename)
    path_file = f"./uploaded_raw/{filename}"
    file.save(path_file)# Save the uploaded file to a temporary location


    # Check if JSON data is included in the request
    if 'json_data' in request.form:
        json_data = request.form['name']

    return upload_to_collection_chromadb(client=chroma_client,coll_n=name,doc_lo=path_file,emb_fn=openai_ef)



@app.route('/delete_collection', methods=['POST'])
def deleteCollection():
    data = request.get_json()

        # Check if the 'name' field is present in the JSON data
    if 'name' not in data:
        return jsonify({'error': 'Name field is missing'}), 400
    
    collection_name = data['name']


    return delete_collection(client=chroma_client,coll_na=collection_name)







#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
#±±±±±±± Chating Section ±±±±±±±
#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±




@app.route('/chat', methods=['POST'])
def chatAI():
    data = request.get_json()

    if 'question' in data and 'collection' in data:
        ai_response, docs = chat_f(client=chroma_client, collection=data['collection'], question=data['question'])

        # Create a dictionary to hold the response data
        response_data = {
            "ai_response": ai_response,
            "docs": [doc.to_json() for doc in docs]
        }

        # Return the response data as JSON
        return jsonify(response_data)
    else:
        return jsonify(error="Invalid request data"), 400




@app.route('/chat_rest', methods=['GET'])
def restChat():
    try:
        memory_t()
        return jsonify({"message":"Chat Has been succfully reset !"})
    except:
        return jsonify({"message":"Something went wrong !"})



@app.route('/get_collections', methods=['GET'])
def get_collection():
    try:
        collection_list = [i.name for i in chroma_client.list_collections()]
        return jsonify({"collection_list":collection_list})
    except:
        return jsonify({"message":"Something went wrong !"})









# def stream_response(string):
#     for char in string:
#         yield char
#         time.sleep(0.3)

# @app.route('/stream', methods=['POST'])
# def stream():
#     data_to_stream = request.data.decode('utf-8')
#     return Response(stream_response(data_to_stream), content_type='text/plain')






















if __name__ == '__main__':
    app.run(debug=True)

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0')