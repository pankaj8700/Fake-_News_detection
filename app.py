
from flask import Flask, request, render_template
from markupsafe import escape
import pickle

app = Flask(__name__)

# Load your model and vectorizer
with open('finalized_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vector = pickle.load(vectorizer_file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        news = request.form.get('news')
        if news:
            # Debugging: Print the input news headline
            print(f"Input news headline: {news}")
            
            # Transform the input and make a prediction
            transformed_news = vector.transform([news])
            predict = model.predict(transformed_news)[0]
            
            # Debugging: Print the transformed input and prediction result
            print(f"Transformed input: {transformed_news}")
            print(f"Prediction result: {predict}")
            
            return render_template("prediction.html", prediction_text="News headline is {}".format(predict))
        else:
            return render_template("prediction.html", prediction_text="No news headline provided.")
    return render_template("prediction.html", prediction_text="")

if __name__ == "__main__":
    app.run()