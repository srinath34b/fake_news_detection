import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')


news = pd.read_csv('fakenewsdataset.csv')


news = news.dropna()

ps = PorterStemmer()


def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    text = text.split()

    text = [ps.stem(word) for word in text
            if word not in stopwords.words('english')]

    return ' '.join(text)


news['processed_text'] = news['text'].apply(preprocess)

X = news['processed_text']
y = news['label']

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)


accuracy = accuracy_score(y_test, predictions)
print('Model Accuracy:', accuracy * 100)


joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print('Model and Vectorizer saved successfully!')

