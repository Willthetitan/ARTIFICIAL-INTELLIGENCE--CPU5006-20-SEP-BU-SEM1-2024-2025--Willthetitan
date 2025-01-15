import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load Datasets
fake_news = pd.read_csv("c:/Users/willt/OneDrive/Documents/GitHub/ARTIFICIAL-INTELLIGENCE--CPU5006-20-SEP-BU-SEM1-2024-2025--Willthetitan/S2 Machine Learning Research Paper/Fake.csv")
true_news = pd.read_csv("c:/Users/willt/OneDrive/Documents/GitHub/ARTIFICIAL-INTELLIGENCE--CPU5006-20-SEP-BU-SEM1-2024-2025--Willthetitan/S2 Machine Learning Research Paper/True.csv")

# Add labels
true_news['label'] = 1
fake_news['label'] = 0

# Combine datasets
combined_news = pd.concat([true_news, fake_news], ignore_index=True)

# Feature extraction
X = combined_news['text']
y = combined_news['label']

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, Y_train)

# Predictions and evaluation
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'True'])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()
