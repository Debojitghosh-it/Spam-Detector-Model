import joblib
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load trained artifacts at startup
model = joblib.load("logistic_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict-ui", methods=["POST"])
def predict_ui():
    email_text = request.form.get("email_text", "")
    text_vector = vectorizer.transform([email_text])
    pred = model.predict(text_vector)[0]
    probs = model.predict_proba(text_vector)[0]

    if pred == 1:
        res = "SPAM ⚠️"
        res_class = "spam"
        conf = round(probs[1] * 100, 2)
    else:
        res = "LEGITIMATE (Ham) ✅"
        res_class = "ham"
        conf = round(probs[0] * 100, 2)

    return render_template_string(
        HTML_TEMPLATE, result=res, result_class=res_class, confidence=conf
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text_vector = vectorizer.transform([data["text"]])
    pred = model.predict(text_vector)[0]
    probs = model.predict_proba(text_vector)[0]

    return jsonify(
        {
            "is_spam": bool(pred == 1),
            "label": "spam" if pred == 1 else "ham",
            "spam_probability": float(probs[1]),
            "confidence_percentage": round(float(probs[pred]) * 100, 2),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
