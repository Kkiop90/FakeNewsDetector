import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load data
df = pd.read_csv("news.csv")

# Features and labels
X = df["text"]
y = df["label"]

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test with your own news
news = input("Enter a news headline: ")

news_vector = vectorizer.transform([news])

prediction = model.predict(news_vector)

print("Prediction:", prediction[0])