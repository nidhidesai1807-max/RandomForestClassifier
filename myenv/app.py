from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open("D:\\Study\\Sem-6\\Classifierdeploy\\myenv\\Classifier.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
