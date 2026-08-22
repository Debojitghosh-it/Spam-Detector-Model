import joblib
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# Load trained artifacts at startup
model = joblib.load("logistic_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Email Spam Detector</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 650px; margin: 40px auto; padding: 20px; line-height: 1.6; }
        textarea { width: 100%; height: 130px; padding: 10px; border-radius: 6px; border: 1px solid #ccc; font-size: 14px; box-sizing: border-box; }
        button { background-color: #0066cc; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 15px; margin-top: 10px; }
        button:hover { background-color: #0052a3; }
        .result-box { margin-top: 25px; padding: 15px; border-radius: 6px; font-size: 1.1em; }
        .spam { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .ham { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
    <h2>📩 Email / SMS Spam Detector</h2>
    <form method="POST" action="/predict-ui">
        <textarea name="email_text" placeholder="Paste email or message text here..." required></textarea><br>
        <button type="submit">Analyze Content</button>
    </form>
    {% if result %}
        <div class="result-box {{ result_class }}">
            <strong>Prediction:</strong> {{ result }}<br>
            <strong>Confidence:</strong> {{ confidence }}%
        </div>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)


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
