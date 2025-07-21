from flask import Flask, request, jsonify
from joblib import load

# Load the data
model = load("./MailByLR.joblib")
vectorizer = load("./tfidf_vectorizer.joblib")

app = Flask(__name__)

map = {
    0 : "Job",
    1 : "Internship",
    2 :"Hackathon",
    3 : "Other"

}

#home route
@app.route('/')
def Home():
    return ("Home server is Running")

@app.route('/LR-checker',methods=["POST"])
def LR():
    data = request.get_json()

    email_text = data["email"]

    vectorized = vectorizer.transform([email_text])

    prediction = model.predict(vectorized)[0]

    confidence = model.predict_proba(vectorized).max()

    return jsonify({
        "prediction" :map[int(prediction)],

        "confidence": float(round(confidence, 4))
    })



#flask server 
if __name__ == "__main__":
    app.run(port=5001,debug=True)