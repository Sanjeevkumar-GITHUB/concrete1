import numpy as np
import pickle
from flask import Flask, request, render_template
# Load ML model

model=pickle.load(open('concrete_model.pkl', 'rb') )
# Create application
concrete = Flask(__name__)


# Bind home function to URL
@concrete.route('/')
def home():
    return render_template("sanjeev.html")


# Bind predict function to URL
@concrete.route('/predict', methods=['POST'])
def predict():
    # Put all form entries values in a list
    features = [i for i in request.form.values()]
    # Convert features to array
    array_features = np.array(features)
    # Predict features
    prediction = model.predict([array_features])

    output = prediction

    # Check the output values and retrive the result with html tag based on the value

    return render_template('sanjeev.html', result=output)


if __name__ == '__main__':
    concrete.run()
