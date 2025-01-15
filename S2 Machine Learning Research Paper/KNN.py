import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix, ConfusionMatrixDisplay

# Load data and add labels
fake_news = pd.read_csv("c:/Users/willt/OneDrive/Documents/GitHub/ARTIFICIAL-INTELLIGENCE--CPU5006-20-SEP-BU-SEM1-2024-2025--Willthetitan/S2 Machine Learning Research Paper/Fake.csv")
true_news = pd.read_csv("c:/Users/willt/OneDrive/Documents/GitHub/ARTIFICIAL-INTELLIGENCE--CPU5006-20-SEP-BU-SEM1-2024-2025--Willthetitan/S2 Machine Learning Research Paper/True.csv")

#add labels
true_news['label'] = 1
fake_news['label'] = 0

# Combine datasets
combined_news = pd.concat([fake_news, true_news], ignore_index=True)

# Feature extraction
X = TfidfVectorizer(max_features=5000, stop_words='english').fit_transform(combined_news['text'])
y = combined_news['label']

# Dimensionality reduction
X_reduced = PCA(n_components=2).fit_transform(X.toarray())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Train KNN and make predictions
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'True'])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()

