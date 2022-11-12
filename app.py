from flask import Flask, jsonify, request
import os
NGROK_TOKEN = os.environ["NGROK_TOKEN"]

from urllib.request import urlretrieve

bucket_name = "kds-e62ac6a7fd414c93dfaf8bcca6c566e6fb8523ab591f1ca05906296d"

if(not os.path.isdir("batch_images")):
    os.mkdir("batch_images")
if(not os.path.isdir("batch_images/batch_features")):
    os.mkdir("batch_images/batch_features")

downloads = [
    "batch_images/compiled_data.json",
    "batch_images/batch_features/batch-0",
    "batch_images/batch_features/batch-1",
    "batch_images/batch_features/batch-2",
    "batch_images/batch_features/batch-3",
    "batch_images/batch_features/batch-4",
    "batch_images/batch_features/batch-5",
    "batch_images/batch_features/batch-6",
    "batch_images/batch_features/batch-7",
    "batch_images/batch_features/batch-8",
    "batch_images/batch_features/batch-9",
    "batch_images/batch_features/batch-10",
    "batch_images/batch_features/batch-11"
]
#http://storage.googleapis.com/BUCKET_NAME/OBJECT_NAME
def getUrl(object_name):
    url = f'http://storage.googleapis.com/{bucket_name}/{object_name}'
    print(url)
    urlretrieve(url,object_name)

for download in downloads:
    if(os.path.isfile(download)):
        print(f"Already Downloaded {download}")
        continue
    getUrl(download)
from model import allFunctions 
# s
from flask_cors import CORS, cross_origin
from flask_ngrok2 import run_with_ngrok



app = Flask(__name__)
run_with_ngrok(app,NGROK_TOKEN)  # Start ngrok when app is run

cors = CORS(app)

@app.route("/encode",methods=["POST"])
@cross_origin()
def encode_text():
    # text = request.form['text']
    print(request.headers.get('Content-Type'))
    req = request.json
    out = allFunctions(req["text"])
    print(out)
    return jsonify({"vals":out})

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
if __name__ == '__main__':
    app.run()
