import pandas as pd

df = pd.read_csv('Dataset/HeartDiseaseTrain-Test.csv')

def rule_based_prediction(row):
        
        if row['chest_pain_type'] == 'Typical angina' and row['vessels_colored_by_flourosopy'] == 'Two' or 'Three' and row['oldpeak'] >= 2 and row['slope'] == 'Downsloping':
             return 1
        elif row['chest_pain_type'] == 'Asymptomatic' and row['fasting_blood_sugar'] == 'Greater than 120 mg/ml' and row['cholestoral'] > 240 and row['rest_ecg'] == 'ST-T wave abnormality':
             return 1
        elif row['chest_pain_type'] == 'Non-anginal pain' and row['vessels_colored_by_flourosopy'] == 'One' and row['slope'] == 'Flat' and row['cholestoral'] > 240:
             return 1
        elif row['age'] < 45 and row['thalassemia'] == 'Reversable Defect' and row['rest_ecg'] == 'ST-T wave abnormality':
             return 1
        else:
             return 0
   

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


# Evaluate the accuracy
accuracy = ((truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative ))
precision = ((truePositive)/(truePositive + falsePositive))
recall = ((truePositive)/(truePositive + falseNegative))
F1Score = ((2* precision * recall)/(precision + recall))
specific = ((trueNegative)/(trueNegative+falsePositive))
typeOne = ((falsePositive)/(trueNegative + falsePositive))
typeTwo = ((falseNegative) / (truePositive + falseNegative))

print(f'Accuracy of the rule-based algorithm: {accuracy * 100:.2f}%')

# Display a few predictions
print(df[['age', 'chest_pain_type', 'cholestoral', 'target', 'predicted_target']].head())