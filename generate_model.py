#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('Loan_default.csv')


# In[4]:


df


# In[5]:


conn = sqlite3.connect('loan_default.db')
cur = conn.cursor()


# In[6]:


cur.execute("DROP TABLE IF EXISTS Clients;")
cur.execute("DROP TABLE IF EXISTS Loans;")
cur.execute("DROP TABLE IF EXISTS Employment;")
cur.execute("DROP TABLE IF EXISTS CreditDetails;")


# In[7]:


# Generate unique ClientID by grouping client-specific attributes
df['ClientID'] = df.groupby(['Age', 'Income', 'CreditScore', 'EmploymentType',
                             'Education', 'MaritalStatus', 'HasDependents', 'HasMortgage']).ngroup()


# In[8]:


create_clients_table = """
CREATE TABLE IF NOT EXISTS Clients (
    ClientID INTEGER PRIMARY KEY,
    Age INTEGER,
    Income REAL,
    CreditScore INTEGER,
    EmploymentType TEXT,
    Education TEXT,
    MaritalStatus TEXT,
    HasDependents TEXT,
    HasMortgage TEXT
);
"""


# In[9]:


create_loans_table = """
CREATE TABLE IF NOT EXISTS Loans (
    LoanID TEXT PRIMARY KEY,
    ClientID INTEGER,
    LoanAmount REAL,
    InterestRate REAL,
    LoanTerm INTEGER,
    DTIRatio REAL,
    LoanPurpose TEXT,
    HasCoSigner TEXT,
    [Default] INTEGER,
    FOREIGN KEY(ClientID) REFERENCES Clients(ClientID)
);
"""


# In[10]:


create_employment_table = """
CREATE TABLE IF NOT EXISTS Employment (
    EmploymentType TEXT PRIMARY KEY,
    MonthsEmployed INTEGER
);
"""


# In[11]:


create_credit_details_table = """
CREATE TABLE IF NOT EXISTS CreditDetails (
    LoanID TEXT PRIMARY KEY,
    NumCreditLines INTEGER,
    FOREIGN KEY(LoanID) REFERENCES Loans(LoanID)
);
"""


# In[12]:


for table_sql in [create_clients_table, create_loans_table, create_employment_table, create_credit_details_table]:
    cur.execute(table_sql)


# In[13]:


conn.commit()


# In[14]:


clients_data = df[['ClientID', 'Age', 'Income', 'CreditScore', 'EmploymentType',
                   'Education', 'MaritalStatus', 'HasDependents', 'HasMortgage']].drop_duplicates()
clients_data.to_sql('Clients', conn, if_exists='replace', index=False)


# In[15]:


loans_data = df[['LoanID', 'ClientID', 'LoanAmount', 'InterestRate', 'LoanTerm',
                 'DTIRatio', 'LoanPurpose', 'HasCoSigner', 'Default']]
loans_data.to_sql('Loans', conn, if_exists='replace', index=False)


# In[16]:


employment_data = df[['ClientID', 'EmploymentType', 'MonthsEmployed']].drop_duplicates()
employment_data.to_sql('Employment', conn, if_exists='replace', index=False)


# In[17]:


credit_details_data = df[['LoanID', 'NumCreditLines']].drop_duplicates()
credit_details_data.to_sql('CreditDetails', conn, if_exists='replace', index=False)


# In[18]:


print("Clients Table:")
print(pd.read_sql("SELECT * FROM Clients LIMIT 5", conn))


# In[19]:


print("\nLoans Table:")
print(pd.read_sql("SELECT * FROM Loans LIMIT 5", conn))


# In[20]:


print("\nEmployment Table:")
print(pd.read_sql("SELECT * FROM Employment LIMIT 5", conn))


# In[21]:


print("\nCreditDetails Table:")
print(pd.read_sql("SELECT * FROM CreditDetails LIMIT 5", conn))


# In[22]:


query = """
SELECT DISTINCT
    Loans.LoanID,
    Clients.Age,
    Clients.Income,
    Loans.LoanAmount,
    Clients.CreditScore,
    Employment.MonthsEmployed,
    CreditDetails.NumCreditLines,
    Loans.InterestRate,
    Loans.LoanTerm,
    Loans.DTIRatio,
    Clients.Education,
    Clients.EmploymentType,
    Clients.MaritalStatus,
    Clients.HasMortgage,
    Clients.HasDependents,
    Loans.LoanPurpose,
    Loans.HasCoSigner,
    Loans.[Default]
FROM Loans
JOIN Clients ON Loans.ClientID = Clients.ClientID
JOIN Employment ON Clients.ClientID = Employment.ClientID -- Match ClientID for accuracy
JOIN CreditDetails ON Loans.LoanID = CreditDetails.LoanID;
"""


# In[23]:


combined_data = pd.read_sql(query, conn)


# In[24]:


print(combined_data)

conn.close()


# In[25]:


df = combined_data


# In[26]:


default_counts = df['Default'].value_counts(normalize=True)  # Proportions
print("Class distribution for 'Default':\n", default_counts)


# In[27]:


plt.bar(default_counts.index, default_counts.values, tick_label=['No Default', 'Default'])
plt.title('Class Distribution of Default')
plt.xlabel('Default')
plt.ylabel('Proportion')
plt.show()


# In[28]:


categorical_columns = ['LoanPurpose', 'EmploymentType', 'MaritalStatus']
for col in categorical_columns:
    print(f"\nDistribution for {col}:\n", df[col].value_counts(normalize=True))
    df[col].value_counts(normalize=True).plot(kind='bar', title=f'{col} Distribution')
    plt.show()


# In[31]:


#only numerical columns
numerical_data = df.select_dtypes(include=['float64', 'int64'])

#correlation matrix
correlation = numerical_data.corr()['Default'].sort_values(ascending=False)
print("\nCorrelation with 'Default':\n", correlation)


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X = df.drop(columns=['Default', 'LoanID'])
y = df['Default']


# In[31]:


X


# In[32]:


y


# In[96]:


#stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[97]:


print("Train set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])


# In[98]:


print("\nTrain 'Default' distribution:\n", y_train.value_counts(normalize=True))
print("\nTest 'Default' distribution:\n", y_test.value_counts(normalize=True))


# In[99]:


train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)


# In[100]:


from ydata_profiling import ProfileReport


# In[101]:


profile = ProfileReport(train_data, title="Train Data Profiling Report", explorative=True)


# In[102]:


profile.to_file("train_data_profile_report.html")


# In[38]:


profile.to_notebook_iframe()


# In[43]:


get_ipython().system('pip install --upgrade --force-reinstall --user mlflow dagshub')


# In[ ]:


get_ipython().system('pip install --user --upgrade jinja2')


# In[33]:


import dagshub
import mlflow
from mlflow.models import infer_signature
import os


# In[34]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
import mlflow.sklearn
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)


# In[106]:


# Preprocessing for numerical and categorical data
numerical_features = ['Income', 'LoanAmount', 'CreditScore', 'InterestRate', 'DTIRatio']
categorical_features = ['Education', 'EmploymentType', 'LoanPurpose', 'MaritalStatus']

numerical_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Define models to evaluate
models = {
    "LogisticRegression": LogisticRegression(class_weight="balanced", random_state=42),
    "RidgeClassifier": RidgeClassifier(class_weight="balanced", random_state=42),
    "RandomForestClassifier": RandomForestClassifier(class_weight="balanced", random_state=42),
    "XGBClassifier": XGBClassifier(scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]), random_state=42)
}

# Initialize MLflow tracking. Credentials must be provided via environment
# variables before running this script (see .env.example).
mlflow.set_tracking_uri(os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/Shruti-1205/my-first-repo.mlflow",
))
mlflow.set_experiment("Classifier Comparison Experiment")

# Training and logging
f1_scores = {}  # To store F1-scores for comparison
for model_name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    val_f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Cross-validation for F1-score
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")

    # Store F1-score for plotting
    f1_scores[model_name] = val_f1

    # Logging to MLflow
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("f1_score", val_f1)
        mlflow.log_metric("mean_cv_f1", cv_scores.mean())
        mlflow.log_metric("std_cv_f1", cv_scores.std())
        mlflow.log_metric("tn", tn)
        mlflow.log_metric("fp", fp)
        mlflow.log_metric("fn", fn)
        mlflow.log_metric("tp", tp)

        # Save predictions
        predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        predictions_file = f"{model_name}_predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)
        mlflow.log_artifact(predictions_file)

        # Log the model
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="models",
            signature=signature,
            input_example=X_train.iloc[:5]
        )

# F1-score plot
import matplotlib.pyplot as plt
plt.bar(f1_scores.keys(), f1_scores.values())
plt.ylabel("F1-Score")
plt.title("Model F1-Score Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("f1_score_comparison.png")
#plt.show()
mlflow.log_artifact("f1_score_comparison.png")


# In[169]:


mlflow.end_run()


# In[109]:


get_ipython().system('pip install --user imbalanced-learn')


# In[35]:


from sklearn.preprocessing import FunctionTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin


# In[124]:


# Custom transformer for feature engineering
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Assuming 'LoanAmount', 'Income', and 'CreditScore' are the first 3 numerical features
        loan_amount_idx = numerical_features.index("LoanAmount")
        income_idx = numerical_features.index("Income")
        credit_score_idx = numerical_features.index("CreditScore")

        # Add new features
        loan_amount_to_income = X[:, loan_amount_idx] / (X[:, income_idx] + 1e-9)
        credit_score_to_income = X[:, credit_score_idx] / (X[:, income_idx] + 1e-9)

        # Append the new features to the array
        X = np.column_stack([X, loan_amount_to_income, credit_score_to_income])
        return X


# In[127]:


feature_engineering = FeatureEngineering()

models = {
    "LogisticRegression": LogisticRegression(class_weight="balanced", random_state=42),
    "RidgeClassifier": RidgeClassifier(class_weight="balanced", random_state=42),
    #"RandomForestClassifier": RandomForestClassifier(class_weight="balanced", random_state=42),
    #"XGBClassifier": XGBClassifier(scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]), random_state=42)
}

for model_name, model in models.items():
    pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("feature_engineering", feature_engineering),
        ("smote", SMOTE(random_state=42)),
        ("classifier", model)
    ])

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    val_f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Log results to MLflow
    with mlflow.start_run(run_name=f"Experiment #3 - Feature Engineering"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("f1_score", val_f1)
        mlflow.log_metric("tn", tn)
        mlflow.log_metric("fp", fp)
        mlflow.log_metric("fn", fn)
        mlflow.log_metric("tp", tp)

        # Save model to MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="models",
            input_example=X_train.iloc[:5]
        )


# In[148]:


# Select only numerical columns for Random Forest
X_train_numeric = X_train.select_dtypes(include=[np.number])

# Train Random Forest to get feature importances
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_numeric, y_train)

# Get feature importances
importances = rf.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X_train_numeric.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Select top features based on cumulative importance
cumulative_importance = importance_df["Importance"].cumsum()
selected_features = importance_df[cumulative_importance <= 0.95]["Feature"]

# Filter training and testing sets to include only selected features
X_train_imp = X_train_numeric[selected_features]
X_test_imp = X_test[selected_features]

print(f"Selected features based on importance: {selected_features.tolist()}")


# In[152]:


from sklearn.feature_selection import VarianceThreshold

# Apply variance threshold
vt = VarianceThreshold(threshold=0.05)
X_train_vt = vt.fit_transform(X_train_imp)
X_test_vt = vt.transform(X_test_imp)

# Get selected feature names
selected_variance_features = X_train_imp.columns[vt.get_support()]
print(f"Selected features after variance threshold: {selected_variance_features.tolist()}")


# In[154]:


# Updated pipeline for Experiment #4
models = {
    "LogisticRegression": LogisticRegression(class_weight="balanced", random_state=42),
    "RidgeClassifier": RidgeClassifier(class_weight="balanced", random_state=42),
    "RandomForestClassifier": RandomForestClassifier(class_weight="balanced", random_state=42),
    "XGBClassifier": XGBClassifier(scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]), random_state=42)
}

for model_name, model in models.items():
    pipeline = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=42)),  # Balance classes with SMOTE
        ("classifier", model)  # Use the model
    ])
    pipeline.fit(X_train_vt, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test_vt)
    val_f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Log results to MLflow
    with mlflow.start_run(run_name=f"Experiment #4 - {model_name}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("f1_score", val_f1)
        mlflow.log_metric("tn", tn)
        mlflow.log_metric("fp", fp)
        mlflow.log_metric("fn", fn)
        mlflow.log_metric("tp", tp)

        # Save model to MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="models",
            input_example=X_train_vt[:5]
        )

    print(f"{model_name} F1-Score: {val_f1}")


# In[156]:


from sklearn.decomposition import PCA
# Scale the data for PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

# Apply PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Scree plot
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title("Scree Plot")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid()
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.legend()
plt.show()

# Select the number of components to explain 95% variance
n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components to retain 95% variance: {n_components}")


# In[164]:


# Reduce to the optimal number of components
pca = PCA(n_components=n_components)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)
X_train_reduced = X_train_reduced.astype(np.float32)
X_test_reduced = X_test_reduced.astype(np.float32)


# In[167]:


from sklearn.model_selection import RandomizedSearchCV
# Hyperparameter grids
param_grids = {
    "XGBClassifier": {
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.1, 0.2],
        "classifier__n_estimators": [50, 100]
    },
    "RandomForestClassifier": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10, None],
        "classifier__class_weight": ["balanced"]
    },
    "RidgeClassifier": {
        "classifier__alpha": [0.1, 1, 10]
    },
    "LogisticRegression": {
        "classifier__C": [0.1, 1, 10]
    }
}

# Train models with hyperparameter tuning
best_f1_scores = {}
for model_name, model in models.items():
    param_grid = param_grids.get(model_name, {})
    pipeline = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=42)),  # Balance classes
        ("classifier", model)
    ])

    random_search = RandomizedSearchCV(pipeline, param_grid, n_iter=10, scoring="f1", cv=3, n_jobs=1, random_state=42)
    random_search.fit(X_train_reduced, y_train)

    # Best model
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test_reduced)

    # Evaluate performance
    val_f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    best_f1_scores[model_name] = val_f1

    # Log results to MLflow
    with mlflow.start_run(run_name=f"Experiment #5 - {model_name}"):
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("f1_score", val_f1)
        mlflow.log_metric("tn", tn)
        mlflow.log_metric("fp", fp)
        mlflow.log_metric("fn", fn)
        mlflow.log_metric("tp", tp)

        # Save model to MLflow
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="models",
            input_example=X_train_reduced[:5]
        )

    print(f"{model_name} - Best F1-Score: {val_f1}")
    print(f"False Positives: {fp}, False Negatives: {fn}")


# In[168]:


# Visualize F1-scores
plt.bar(best_f1_scores.keys(), best_f1_scores.values())
plt.ylabel("F1-Score")
plt.title("Experiment #5: F1-Score Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("experiment5_f1_scores.png")
plt.show()

# Log the plot to MLflow
mlflow.log_artifact("experiment5_f1_scores.png")


# In[193]:


mlflow.end_run()


# In[190]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_features = X.select_dtypes(include=[np.number]).columns
categorical_features = X.select_dtypes(include=['object']).columns

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Experiment 6: Decision Tree Classifier
mlflow.set_experiment("Experiment 6: Decision Tree Classifier")
dt_params = {
    'max_depth': 5,
    'min_samples_split': 10,
    'class_weight': 'balanced',  # Penalize false negatives
    'random_state': 42
}

dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(**dt_params))
])

with mlflow.start_run(run_name="Decision Tree Classifier"):
    # Train the model
    dt_pipeline.fit(X_train, y_train)
    y_pred = dt_pipeline.predict(X_test)

    # Evaluate the model
    val_f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Log parameters, metrics, and confusion matrix
    mlflow.log_param("max_depth", dt_params['max_depth'])
    mlflow.log_param("min_samples_split", dt_params['min_samples_split'])
    mlflow.log_metric("f1_score", val_f1)
    mlflow.log_metric("precision", tp / (tp + fp))
    mlflow.log_metric("recall", tp / (tp + fn))
    mlflow.log_metric("tp", tp)
    mlflow.log_metric("fp", fp)
    mlflow.log_metric("tn", tn)
    mlflow.log_metric("fn", fn)
    mlflow.sklearn.log_model(dt_pipeline, artifact_path="model")

    print(f"Decision Tree - F1-Score: {val_f1}")
    
rf_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'class_weight': 'balanced',  # Penalize false negatives
    'random_state': 42
}

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(**rf_params))
])

mlflow.set_experiment("Updated Random Forest")
with mlflow.start_run(run_name="Random Forest Classifier"):
    # Train the model
    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)

    # Evaluate the model
    val_f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Log parameters, metrics, and confusion matrix
    mlflow.log_param("n_estimators", rf_params['n_estimators'])
    mlflow.log_param("max_depth", rf_params['max_depth'])
    mlflow.log_metric("f1_score", val_f1)
    mlflow.log_metric("precision", tp / (tp + fp))
    mlflow.log_metric("recall", tp / (tp + fn))
    mlflow.log_metric("tp", tp)
    mlflow.log_metric("fp", fp)
    mlflow.log_metric("tn", tn)
    mlflow.log_metric("fn", fn)
    mlflow.sklearn.log_model(rf_pipeline, artifact_path="model")

    print(f"Random Forest - F1-Score: {val_f1}")
    
xgb_params = {
    'n_estimators': 150,
    'max_depth': 10,
    'learning_rate': 0.1,
    'scale_pos_weight': y_train.value_counts()[0] / y_train.value_counts()[1],  # Balance classes
    'random_state': 42
}

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(**xgb_params))
])

mlflow.set_experiment("Updated XGBoost")
with mlflow.start_run(run_name="XGBoost Classifier"):
    # Train the model
    xgb_pipeline.fit(X_train, y_train)
    y_pred = xgb_pipeline.predict(X_test)

    # Evaluate the model
    val_f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Log parameters, metrics, and confusion matrix
    mlflow.log_param("n_estimators", xgb_params['n_estimators'])
    mlflow.log_param("max_depth", xgb_params['max_depth'])
    mlflow.log_param("learning_rate", xgb_params['learning_rate'])
    mlflow.log_metric("f1_score", val_f1)
    mlflow.log_metric("precision", tp / (tp + fp))
    mlflow.log_metric("recall", tp / (tp + fn))
    mlflow.log_metric("tp", tp)
    mlflow.log_metric("fp", fp)
    mlflow.log_metric("tn", tn)
    mlflow.log_metric("fn", fn)
    mlflow.sklearn.log_model(xgb_pipeline, artifact_path="model")

    print(f"XGBoost - F1-Score: {val_f1}")


# In[192]:


from sklearn.ensemble import VotingClassifier

dt_model = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
xgb_model = XGBClassifier(n_estimators=150, max_depth=10, learning_rate=0.1, random_state=42)

# Ensemble model with soft voting
voting_model = VotingClassifier(
    estimators=[
        ('Decision Tree', dt_model),
        ('Random Forest', rf_model),
        ('XGBoost', xgb_model)
    ],
    voting='soft'  # Soft voting for better precision-recall balance
)

# Define the full pipeline
ensemble_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('voting', voting_model)
])

# Train and evaluate the ensemble model
mlflow.set_experiment("Ensemble Model Evaluation")
with mlflow.start_run(run_name="Ensemble Voting Classifier"):
    # Train the ensemble pipeline
    ensemble_pipeline.fit(X_train, y_train)
    y_pred = ensemble_pipeline.predict(X_test)

    # Evaluate performance
    val_f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Log metrics and confusion matrix
    mlflow.log_param("voting", "soft")
    mlflow.log_metric("f1_score", val_f1)
    mlflow.log_metric("precision", tp / (tp + fp))
    mlflow.log_metric("recall", tp / (tp + fn))
    mlflow.log_metric("tp", tp)
    mlflow.log_metric("fp", fp)
    mlflow.log_metric("tn", tn)
    mlflow.log_metric("fn", fn)

    # Log the model
    mlflow.sklearn.log_model(ensemble_pipeline, artifact_path="model")
    print(f"Ensemble Voting Classifier - F1-Score: {val_f1}")


# In[197]:


import mlflow
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("https://dagshub.com/Shruti-1205/my-first-repo.mlflow")

experiment_names = [
    "Classifier Comparison Experiment",
    "Experiment #3 - Feature Engineering",
    "Experiment #4 - Feature Selection",
    "Experiment #5 - PCA and Hyperparameter Tuning",
    "Experiment 6: Decision Tree Classifier",
    "Updated Random Forest",
    "Updated XGBoost",
    "Ensemble Model Evaluation"
]

f1_scores = {}
missing_data = [] 

for experiment_name in experiment_names:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="attributes.status = 'FINISHED'"
        )
        if not runs.empty:
            f1_score = runs['metrics.f1_score'].max() if 'metrics.f1_score' in runs else None
            if f1_score:
                f1_scores[experiment_name] = float(f1_score)
            else:
                missing_data.append(experiment_name)
        else:
            missing_data.append(experiment_name)
    else:
        missing_data.append(experiment_name)

# Sort experiments by F1-score
sorted_f1_scores = {k: v for k, v in sorted(f1_scores.items(), key=lambda item: item[1], reverse=True)}

# Plot F1-scores
plt.figure(figsize=(12, 6))
plt.bar(sorted_f1_scores.keys(), sorted_f1_scores.values(), color='skyblue')
plt.ylabel("F1-Score")
plt.title("F1-Score Comparison Across Experiments")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("f1_score_comparison_final.png")
plt.show()

if sorted_f1_scores:
    best_model = max(sorted_f1_scores, key=sorted_f1_scores.get)
    print(f"The best-performing model is '{best_model}' with an F1-Score of {sorted_f1_scores[best_model]:.4f}")
else:
    print("No valid F1-scores found. Please check the experiments.")


# In[36]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_features = X.select_dtypes(include=[np.number]).columns
categorical_features = X.select_dtypes(include=['object']).columns

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

rf_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'class_weight': 'balanced',  # Penalize false negatives
    'random_state': 42
}

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(**rf_params))
])

mlflow.set_experiment("Updated Random Forest")
with mlflow.start_run(run_name="Random Forest Classifier"):
    # Train the model
    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)

    # Evaluate the model
    val_f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Log parameters, metrics, and confusion matrix
    mlflow.log_param("n_estimators", rf_params['n_estimators'])
    mlflow.log_param("max_depth", rf_params['max_depth'])
    mlflow.log_metric("f1_score", val_f1)
    mlflow.log_metric("precision", tp / (tp + fp))
    mlflow.log_metric("recall", tp / (tp + fn))
    mlflow.log_metric("tp", tp)
    mlflow.log_metric("fp", fp)
    mlflow.log_metric("tn", tn)
    mlflow.log_metric("fn", fn)
    mlflow.sklearn.log_model(rf_pipeline, artifact_path="model")

    print(f"Random Forest - F1-Score: {val_f1}")


# In[37]:


import joblib
import sklearn
joblib.dump((rf_pipeline, {"scikit_learn_version": sklearn.__version__}), "final_rf_pipeline_with_metadata.joblib")


# In[38]:


import joblib
import sklearn

# Load the model and metadata
loaded_model, metadata = joblib.load("final_rf_pipeline_with_metadata.joblib")

# Verify version compatibility
if sklearn.__version__ != metadata["scikit_learn_version"]:
    print(f"Warning: Model was saved with scikit-learn {metadata['scikit_learn_version']} "
          f"but you are using {sklearn.__version__}. Consider using the same version.")


# In[39]:


from sklearn.metrics import classification_report, confusion_matrix, f1_score

loaded_predictions = loaded_model.predict(X_test)

# Evaluate the predictions
val_f1 = f1_score(y_test, loaded_predictions)
tn, fp, fn, tp = confusion_matrix(y_test, loaded_predictions).ravel()

# Display evaluation metrics
print(f"F1-Score: {val_f1}")
print(f"Confusion Matrix:\n TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
print("Classification Report:")
print(classification_report(y_test, loaded_predictions))


# In[40]:


test_sample = X_test.iloc[0:1]
prediction = loaded_model.predict(test_sample)
actual_label = y_test.iloc[0]

print(f"Prediction: {prediction[0]}, Ground Truth: {actual_label}")


# In[ ]:


# requests
#data = json.dumps(data, indent =2)
#r= requests.post
#r.json

