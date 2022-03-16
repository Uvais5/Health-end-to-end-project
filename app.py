
from flask import Flask,render_template,request
import pickle
import numpy as np 
import pandas as pd

model = pickle.load(open("Health.pkl","rb"))
print(model.predict([[19,1,23.22,0,1]]))
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/Predict",methods=["POST"])
def predict():
    age = request.form["a"]
    sex = request.form["b"]
    bmi = request.form["c"]
    child = request.form["d"]
    smoker = request.form["e"]
    all = np.array([[age,sex,bmi,child,smoker]])
    output = model.predict(all)
    return render_template("index.html", prediction_text=" Insurance Price is $ {}".format(output))
if __name__ == "__main__":
    app.run(debug=True)
