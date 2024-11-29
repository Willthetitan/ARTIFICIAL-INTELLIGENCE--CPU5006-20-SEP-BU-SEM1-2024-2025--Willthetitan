import pandas as pd

df = pd.read_csv('Dataset/HeartDiseaseTrain-Test.csv')

def rule_based_prediction(row):
    if row['chest_pain_type'] == 'Asymptomatic' and row['age'] < 45 and row['vessels_colored_by_flourosopy'] == 'Zero' and row['cholestoral'] < 200:
        return 0
    elif row['age'] > 45 and row['cholestoral'] < 200 and row['chest_pain_type'] == 'Asymptomatic':
        return 0
    elif row['chest_pain_type'] == 'Non-anginal pain'and row['oldpeak'] < 0.2 and row['rest_ecg'] == 'Normal':
        return 0
    else:
        return 1
   

# Apply the rules to the dataset
df['predicted_target'] = df.apply(rule_based_prediction, axis=1)

truePositive = 0
falsePositive = 0
trueNegative = 0
falseNegative = 0

for index,row in df.iterrows():
    if row['predicted_target'] == 1 and row['target'] == 1:
        truePositive += 1
    elif row['predicted_target'] == 0 and row['target'] == 0:
        trueNegative += 1
    elif row['predicted_target'] == 1 and row['target'] == 0:
        falsePositive += 1
    elif row['predicted_target'] == 0 and row['target'] == 1:
        falseNegative += 1

#  made via chat GPT - Section 
confusion_matrix = pd.DataFrame([[truePositive, falseNegative],
                                 [falsePositive, trueNegative]],
                                columns=['Predicted 1', 'Predicted 0'],
                                index=['Actual 1', 'Actual 0'])

print("Confusion Matrix:")
print(confusion_matrix)

# Made via chat GPT - Section


# Evaluate the accuracy
accuracy = ((truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative ))
precision = ((truePositive)/(truePositive + falsePositive))
recall = ((truePositive)/(truePositive + falseNegative))
F1Score = ((2* precision * recall)/(precision + recall))
specific = ((trueNegative)/(trueNegative + falsePositive))
typeOne = ((falsePositive)/(trueNegative + falsePositive))
typeTwo = ((falseNegative) / (truePositive + falseNegative))

print(f'Accuracy of the rule-based algorithm: {accuracy * 100:.2f}%')
print(F1Score * 100)
print(specific)
# Display a few predictions
print(df[['age', 'chest_pain_type', 'cholestoral', 'target', 'predicted_target']].head())