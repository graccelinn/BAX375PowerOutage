import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.model_selection import cross_val_score

df = pd.read_csv('austin_data.csv')

df.drop('time', axis=1, inplace=True)

X = df.drop('outage_occur', axis=1)
y = df['outage_occur']

imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_scaled, y)
print("Test 2")
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
print("Test 1")
model = RandomForestClassifier(random_state=42)
print("Test 3")
model.fit(X_train, y_train)
print("Test 4")
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
probabilities = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, probabilities)
print(f"ROC-AUC score: {roc_auc}")

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Accuracy scores for each fold: {scores}")
print(f"Average cross-validation score: {scores.mean()}")

joblib.dump(model, 'final_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(imputer, 'imputer.joblib')