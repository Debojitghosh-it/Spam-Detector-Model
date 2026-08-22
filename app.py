import os
import joblib

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load trained artifacts
model = joblib.load("logistic_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# =========================
# Home page
# =========================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# =========================
# Web UI prediction
# =========================
@app.route("/predict-ui", methods=["POST"])
def predict_ui():
    email_text = request.form.get("email_text", "").strip()

    if not email_text:
        return render_template(
            "index.html",
            result="Please enter a message.",
            result_class="error",
            confidence=0
        )

    # Convert text into TF-IDF features
    text_vector = vectorizer.transform([email_text])

    # Make prediction
    pred = model.predict(text_vector)[0]
    probs = model.predict_proba(text_vector)[0]

    if pred == 1:
        result = "SPAM ⚠️"
        result_class = "spam"
        confidence = round(float(probs[1]) * 100, 2)
    else:
        result = "LEGITIMATE (Ham) ✅"
        result_class = "ham"
        confidence = round(float(probs[0]) * 100, 2)

    return render_template(
        "index.html",
        result=result,
        result_class=result_class,
        confidence=confidence
    )


# =========================
# API prediction
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():

    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({
            "error": "Missing 'text' field in request body"
        }), 400

    text = str(data["text"]).strip()

    if not text:
        return jsonify({
            "error": "Text cannot be empty"
        }), 400

    # Convert text into TF-IDF features
    text_vector = vectorizer.transform([text])

    # Make prediction
    pred = model.predict(text_vector)[0]
    probs = model.predict_proba(text_vector)[0]

    return jsonify({
        "is_spam": bool(pred == 1),
        "label": "spam" if pred == 1 else "ham",
        "spam_probability": float(probs[1]),
        "confidence_percentage": round(
            float(probs[pred]) * 100, 2
        )
    })


# =========================
# Health check
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    })


# =========================
# Start Flask server
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )