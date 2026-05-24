from flask import Flask, render_template, request
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

ps = PorterStemmer()


def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    text = text.split()

    text = [ps.stem(word) for word in text
            if word not in stopwords.words('english')]

    return ' '.join(text)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']

    processed = preprocess(news_text)

    vector_input = vectorizer.transform([processed])

    prediction = model.predict(vector_input)[0]

    probability = model.predict_proba(vector_input)[0]

    authenticity_score = round(max(probability) * 100, 2)

    if prediction == 1:
        result = 'REAL NEWS'
    else:
        result = 'FAKE NEWS'

    return render_template(
        'index.html',
        prediction_text=result,
        score=authenticity_score,
        news=news_text
    )


if __name__ == '__main__':
    app.run(debug=True)