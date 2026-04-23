from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("iris_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])

    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

    flowers = ["Setosa", "Versicolor", "Virginica"]
    result = flowers[prediction[0]]

    return render_template(
        "index.html",
        prediction_text="Prediction: " + result,
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width
    )

if __name__ == "__main__":
    app.run(debug=True)