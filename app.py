import os
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the saved model and vectorizer at startup
clf = joblib.load("NB_spam_model.pkl")
cv = joblib.load("count_vectorizer.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        print("DEBUG PREDICTION:", my_prediction) # <-- Yeh add karo
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)