import pandas as pd
import numpy as np , time as t
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("\n Loading and Preparing Data",end='')
for i in range(1,4):
    print(".",end='')
    t.sleep(1)
print()

df = pd.read_csv("Reviews.csv", low_memory=False)

df = df[['Text', 'Score']]
df.dropna(inplace=True)

pd.options.display.max_rows = 30
print(df.head(30))


df = df[df['Score'] != 3]

df['Sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)

min_count = min(df['Sentiment'].value_counts())
df_balanced = pd.concat([
    df[df['Sentiment'] == 0].sample(min_count, random_state=42),
    df[df['Sentiment'] == 1].sample(min_count, random_state=42)
])

print(" Data loaded and balanced.")

X_train, X_test, y_train, y_test = train_test_split(
    df_balanced['Text'], df_balanced['Sentiment'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\n Training the Sentiment Classifier",end='')
for i in range(1,4):
    print(".",end='')
    t.sleep(1)
print()
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_vec, y_train)
print("Model training complete.")

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred) * 100
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n Sentiment Analysis Report")
print("="*40)
print(f" Accuracy: {accuracy:.2f}%")
print(f" True Positives (Correctly predicted positive): {tp}")
print(f" True Negatives (Correctly predicted negative): {tn}")
print(f" False Positives (Wrongly predicted positive): {fp}")
print(f" False Negatives (Wrongly predicted negative): {fn}")
print("="*40)
if accuracy > 85:
    print(" Excellent! Your model is performing very well.")
elif accuracy > 70:
    print(" Good job! Your model is fairly accurate.")
else:
    print(" The model needs some improvement. Try tuning it more.")

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

labels = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
values = [1523, 1476, 190, 211]
colors = ['green', 'blue', 'orange', 'red']
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, values, color=colors)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval/2, str(yval),
             ha='center', va='center', fontsize=12, color='white', fontweight='bold')

plt.title('Sentiment Prediction Breakdown', fontsize=16)
plt.ylabel('Number of Reviews')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title("Confusion Matrix")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()