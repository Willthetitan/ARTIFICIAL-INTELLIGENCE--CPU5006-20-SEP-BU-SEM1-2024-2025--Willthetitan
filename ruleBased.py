import pandas as pd

df = pd.read_csv('Dataset/HeartDiseaseTrain-Test.csv')

def rule_based_prediction(row):
    # Rule 1: If chest pain type (cp) = 4 (asymptomatic), predict heart disease = 1
    if row['thalassemia'] == 'Fixed Defect':
        return 1
    # Rule 2: If cholesterol > 240 and age > 50, predict heart disease = 1
    # elif row['cholestoral'] > 130:
    #     return 1
    # elif row['age'] > 65:
    #     return 1
    # elif row['resting_blood_pressure'] > 140:
    #     return 1
    # Rule 3: If maximum heart rate achieved (thalach) > 150 and cholesterol < 200, predict heart disease = 0
    # Default case: Predict no heart disease (0)
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
print(f'Accuracy of the rule-based algorithm: {accuracy * 100:.2f}%')

# Display a few predictions
print(df[['age', 'chest_pain_type', 'cholestoral', 'target', 'predicted_target']].head())