import os
import case
import json
from flask import Flask, send_from_directory, request, render_template

app = Flask(__name__)


# Landing route
@app.route("/")
def landing():
    return send_from_directory('templates', "index.html")


# Route to data csv file
@app.route("/data/data.csv")
def send_data_csv():
    return send_from_directory('static', "data/data.csv")


# Route to LDA html file
@app.route("/lda")
def send_data_LDA():
    return send_from_directory('templates', "LDA.html")


# Route to LSI txt file
@app.route("/lsi")
def send_data_LSI():
    return send_from_directory('templates', "LSI.txt")


# Route to predict new document
@app.route("/predict", methods=['POST'])
def predict_input():
    data = request.form['text']
    prediction = case.predict_new_document(data)
    topic, probability = zip(*prediction)
    topic_name = ["Customer orders", "Personal information", "Train ticket", "Package delivery", "Payment",
                  "Information about issues", "Amazon", "Contacting support", "Flight feedback", "(Good) Internet"]
    return render_template('prediction.html', topic_name=topic_name, results=zip(topic, probability))


# Generic route if no custom operation is required
@app.route("/<path:path>")
def send_page(path):
    return send_from_directory('templates', path+".html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
