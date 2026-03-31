import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


print("--- Loading Dataset ---")
df = pd.read_csv('resumes.csv')


def clean_function(text):
    text = re.sub('http\S+\s*', ' ', text) 
    text = re.sub('#\S+', '', text)        
    text = re.sub('@\S+', '  ', text)       
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text) 
    text = re.sub(r'[^\x00-\x7f]',r' ', text) 
    text = re.sub('\s+', ' ', text)         
    return text.lower()

print("--- Cleaning Resume Text ---")
df['Cleaned_Resume'] = df['Resume'].apply(lambda x: clean_function(x))


print("--- Vectorizing Text Data ---")
tfidf = TfidfVectorizer(stop_words='english', max_features=1500)
X = tfidf.fit_transform(df['Cleaned_Resume'])
y = df['Category']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("--- Training Logistic Regression Model ---")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n[FINAL RESULT] Accuracy: {accuracy * 100:.2f}%")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix: Resume Classification')
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.show() 


print("--- Saving Model for Deployment ---")
joblib.dump(model, 'resume_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("Files saved: resume_model.pkl and tfidf_vectorizer.pkl")