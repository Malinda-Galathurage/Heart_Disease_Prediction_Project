
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv("hddata.csv")


print("Missing values before processing:\n", data.isnull().sum())


categorical_columns = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke',
                       'DiffWalking', 'Sex', 'Race', 'Diabetic', 'PhysicalActivity',
                       'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']


data[categorical_columns] = data[categorical_columns].apply(lambda x: x.map({'Yes': 1, 'No': 0, 'No, borderline diabetes': 0}))

for col in ['Sex', 'AgeCategory', 'Race', 'GenHealth']:
    if data[col].mode().empty:
        print(f"Column {col} has no mode, skipping fill operation.")
    else:
       
        data[col].fillna(data[col].mode()[0], inplace=True)


age_mapping = {
    '18-24': 1, '25-29': 2, '30-34': 3, '35-39': 4, '40-44': 5,
    '45-49': 6, '50-54': 7, '55-59': 8, '60-64': 9, '65-69': 10,
    '70-74': 11, '75-79': 12, '80 or older': 13
}
data['AgeCategory'] = data['AgeCategory'].map(age_mapping)


scaler = MinMaxScaler()
numerical_columns = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])


print("Missing values after processing:\n", data.isnull().sum())


print(data.head())

data.to_csv('preprocessed_data.csv', index=False)

print("Data preprocessing completed and saved as 'preprocessed_hddata.csv'.")
