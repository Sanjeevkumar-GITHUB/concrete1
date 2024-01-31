import os
import numpy as np
import pickle
from flask import Flask, request, render_template

# Load ML model
model = ""
# if os.path.getsize("concrete_model.pkl") > 0:
#     model = pickle.load(open("concrete_model.pkl", "rb"))
# with open("concrete_model.pkl", "rb") as f:
#     model = pickle.load(f)
with open("concrete_model_1.pkl", "rb") as f:
    model = pickle.Unpickler(f)
# Create application
app = Flask(__name__)


# Bind home function to URL
@app.route("/")
def home():
    return render_template("index.html")


# Bind predict function to URL
@app.route("/predict", methods=["POST"])
def predict():
    # Put all form entries values in a list
    features = [i for i in request.form.values()]
    # Convert features to array
    array_features = np.array(features)
    # Predict features
    prediction = model.predict([array_features])

    output = prediction

    # Check the output values and retrive the result with html tag based on the value

    return render_template("index.html", result=output)


if __name__ == "__main__":
    app.run()
