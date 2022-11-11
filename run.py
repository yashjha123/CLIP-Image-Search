from flask import Flask, jsonify, request
from model import allFunctions 

from flask_cors import CORS, cross_origin

app = Flask(__name__)
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