import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Your existing code (perfect) + aggregation + drop Dwelling Type
df = pd.read_csv('data.csv')
df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
df = df.dropna(subset=['VALUE'])
df = df.drop('STATISTIC Label', axis=1)

df['Dwelling_Category'] = df['Dwelling Type'].apply(
    lambda x: 'Apartment' if pd.isna(x) == False and 'apartment' in str(x).lower()
    else ('House' if pd.isna(x) == False and 'house' in str(x).lower()
          else 'Other')
)

print("Original Dwelling Type unique:", df['Dwelling Type'].nunique())
print("Dwelling_Category counts:", df['Dwelling_Category'].value_counts())

# ── CHECK & AGGREGATE DUPLICATES ───────────────────────────────────────────
dup_check = df.groupby(['Year', 'Period of Construction', 'BER Rating', 'County and Dublin Postal District',
                        'Dwelling_Category']).size().reset_index(name='dup_count')

print("\nDuplicate groups:")
print(dup_check[dup_check['dup_count'] > 1])
print(f"Total rows before agg: {len(df)}")

if len(dup_check[dup_check['dup_count'] > 1]):
    aggregated = df.groupby(['Year', 'Period of Construction', 'BER Rating',
                             'County and Dublin Postal District', 'Dwelling_Category'])['VALUE'].sum().reset_index()
    print(f"Aggregated {len(df) - len(aggregated)} duplicate rows")

    df = aggregated.copy()
else:
    print("No duplicates found")

print(f"Final rows: {len(df)}")

# ── DROP ORIGINAL COLUMN & SAVE ─────────────────────────────────────────────
df = df.drop('Dwelling Type', axis=1, errors='ignore')
df.to_csv('cleanedData.csv', index=False)
print("Saved")
print(df['Dwelling_Category'].value_counts())
