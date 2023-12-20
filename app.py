from flask import Flask, jsonify, request
import pandas as pd
from predict import predict_chatbot, predict_twitter

app = Flask(__name__)

@app.route('/')

def index():
    return jsonify({"status": {
        "code": 200,
        "message": "Success"
    }})

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # predict ML model output
        data = request.get_json() 
        text_input = data['text']
        model_type = data['model_type']
        result = None

        # tokenize text input
        if model_type == 'chatbot':
            result = predict_chatbot(text_input)
        elif model_type == 'twitter':
            result = predict_twitter(text_input)
        else:
            result = "Model type not found"

        # predict chatbot model

        return jsonify({"status": {
            "code": 200,
            "message": "Success"
        }, "data": result})
    else:
        return jsonify({"status": {
        "code": 405,
        "message": "Method now allowed"
    }, "data": None})

if __name__ == '__main__':
    app.run(port=8000, debug=True)