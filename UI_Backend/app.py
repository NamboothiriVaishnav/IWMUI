from ast import Try
import json
from flask import Flask ,request
from blob_sdk import listing_blobs_directories,ConnectToStorageAcc,upload_file_to_directory
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/listing_blobs_directories',methods = ['GET'])
def f22():
    if (request.method=='GET'):
        file_list = listing_blobs_directories('finaldataiwm','')
        return json.dumps(file_list), 200

@app.route('/upload_file_to_directory',methods=['POST'])
def f23():
    if(request.method=='POST'):
        ConnectToStorageAcc('iwmstgacc','871wPYug1CXqXhOomD9hJGou/5ulJhgr9uW7MjS8BEhZLrXUfz4FXr26Ve6G8sQsnDQVTNBWRYIT+ASt3kzLcA==')
        upload_file_to_directory("C:\\Users\\Dataset_Bill1.csv")
        return "Uploaded Successfully"

if __name__ == '__main__':
    app.run(debug = True)   