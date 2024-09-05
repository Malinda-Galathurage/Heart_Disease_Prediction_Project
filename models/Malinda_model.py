import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('../data/preprocessed_data.csv')
data = data.drop(['Sex', 'Race', 'GenHealth'], axis=1)

numerical_features = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
categorical_features = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking',
                       'AgeCategory', 'Diabetic', 'PhysicalActivity', 'Asthma',
                       'KidneyDisease', 'SkinCancer']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the Logistic Regression model
model = LogisticRegression(random_state=0) 

# Create the pipeline (using the LogisticRegression model)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])  

X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy}")
