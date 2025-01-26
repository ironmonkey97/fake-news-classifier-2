from flask import Flask, request, render_template
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
vectorizer = joblib.load("vectorizer.pkl")
nb_model = joblib.load("naive_bayes_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from form
        user_input = request.form['news_text']

        # Transform input using the loaded vectorizer
        X_new = vectorizer.transform([user_input])

        # Predict using the loaded model
        prediction = nb_model.predict(X_new)

        # Convert prediction to human-readable format
        prediction_label = "Fake News" if prediction[0] == 0 else "Real News"

        return render_template('index.html', prediction_text=f"Prediction: {prediction_label}")

if __name__ == "__main__":
    app.run(debug=True)
