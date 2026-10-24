"""Data Preparation: clean the tourism dataset and create train/test splits."""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Drop unnamed index column if present
if df.columns[0].startswith("Unnamed"):
    df = df.drop(columns=df.columns[0])

# --- Data Cleaning ---
# Drop the unique identifier (no predictive value)
df = df.drop(columns=["CustomerID"])

# Fix typo in Gender column
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Impute missing values
# Numerical columns: fill with median
num_cols = df.select_dtypes(include="number").columns.tolist()
num_cols.remove("ProdTaken")  # exclude target
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Categorical columns: fill with mode
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print(f"Cleaned data shape: {df.shape}")
print(f"Remaining missing values: {df.isnull().sum().sum()}")

# --- Train-Test Split ---
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the splits
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)
train_df.to_csv("tourism_project/data/train.csv", index=False)
test_df.to_csv("tourism_project/data/test.csv", index=False)

print(f"Train set: {train_df.shape[0]} rows, Test set: {test_df.shape[0]} rows")
print(f"Train purchase rate: {y_train.mean():.2%}")
print(f"Test purchase rate:  {y_test.mean():.2%}")
