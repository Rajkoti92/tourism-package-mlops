"""Data Registration: validate the tourism dataset and print a summary."""
import pandas as pd
import sys

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

df = pd.read_csv(DATA_PATH)

# Drop unnamed index column if present
if df.columns[0].startswith("Unnamed"):
    df = df.drop(columns=df.columns[0])

# Validate columns
missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
if missing_cols:
    print(f"VALIDATION FAILED - missing columns: {missing_cols}")
    sys.exit(1)

print("VALIDATION PASSED")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nTarget distribution (ProdTaken):\n{df['ProdTaken'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nData types:\n{df.dtypes}")
