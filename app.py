from flask import Flask, render_template, request
import torch
from src.pipeline.prediction_pipeline import PredictionPipeline


prediction_pipeline=PredictionPipeline()
app = Flask(__name__)

# Burada eğittiğin modelin dosyasını yüklemelisin

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_input = request.form.get("sequence")
        try:
            sequence = list(map(int, user_input.split()))
            if len(sequence) > 8 or any(num < 1 or num > 10 for num in sequence):
                result = "Hatalı giriş! 1 ile 10 arasında en fazla 8 sayı giriniz."
            else:
                sequence = torch.tensor(sequence, dtype=torch.long)
                result = prediction_pipeline.run_prediction_pipeline(sequence)
                result = " ".join(map(str, result))
        
        except ValueError:
            result = "Hatalı format! Lütfen boşlukla ayrılmış sayılar giriniz."
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)
