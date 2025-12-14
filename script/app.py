# app.py (di folder script)
import os
from flask import Flask, request, jsonify
from predict import predict as predict_text   # karena satu folder, bisa langsung import
from predict_full import predict_full

app = Flask(__name__)

@app.route("/predict_text", methods=["POST"])
def predict_text_endpoint():
    data = request.json
    keluhan = data.get("keluhan", "")
    if not keluhan:
        return jsonify({"error": "keluhan is required"}), 400
    return jsonify(predict_text(keluhan))


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    # Ambil keluhan dari form-data
    keluhan = request.form.get("keluhan")
    if not keluhan:
        return jsonify({"error": "keluhan is required"}), 400

    # Ambil file image
    files = request.files.getlist("image[]")
    if not files:
        return jsonify({"error": "image file is required"}), 400

    try:
        text_result = predict_text(keluhan)
        #image_result = predict_image_file(tmp_path)
        #final_result = predict_full(text_result, files)
        image_results = []
        for file in files:
            result = predict_full(text_result, file)
            image_results.append(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    #return jsonify(final_result)
    return jsonify({
        "text": text_result,
        "images": image_results
    })

if __name__ == "__main__":
    # Run Flask langsung dari script
    app.run(host="0.0.0.0", port=5000, debug=True)
