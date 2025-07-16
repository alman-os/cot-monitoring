# server.py
from flask import Flask, request, jsonify
from diagnostics import analyze_cot, to_json

app = Flask(__name__)

@app.route("/analyze_cot", methods=["POST"])
def analyze():
    data = request.json
    cot  = data.get("cot")  # expect list[str]
    if not cot: return jsonify({"error":"cot list missing"}), 400
    report = analyze_cot(cot)
    return jsonify(to_json(report))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)
