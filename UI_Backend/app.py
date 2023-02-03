from ast import Try
import json
from flask import Flask ,request
from blob_sdk import listing_blobs_directories,upload_file_to_directory
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
        path_param=request.args.get('path')
        upload_file_to_directory(path_param)
        return "Uploaded Successfully"

if __name__ == '__main__':
    app.run(debug = True)