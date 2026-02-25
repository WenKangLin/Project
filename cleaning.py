import pandas as pd

# ----- Load and Drop -----
df = pd.read_csv('data.csv')
df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
df = df.dropna(subset=['VALUE'])
df = df.drop('STATISTIC Label', axis=1)

# ----- Cleaning Data -----
def clean_county(name):
    if pd.isna(name):
        return 'Unknown'

    name = str(name).strip()
    name = name.replace('Co. ', '').replace('Co.', '').replace(' City', '')

    return name

df['County'] = df['County and Dublin Postal District'].apply(clean_county)
print("\nCounty unique after clean:", df['County'].nunique())


df['Dwelling_Category'] = df['Dwelling Type'].apply(
    lambda x: 'Apartment' if 'apartment' in str(x).lower()
            else 'House' if 'house' in str(x).lower()
            else 'Other')

print(df['Dwelling_Category'].value_counts())

# ----- AGGREGATE DUPLICATES -----
aggregated = df.groupby(['Year', 'BER Rating', 'Period of Construction', 'Dwelling_Category', 'County'])['VALUE'].sum().reset_index()
print(f"Rows before: {len(df)} → after aggregation: {len(aggregated)}")
print(f"Combined {len(df) - len(aggregated)} duplicates")

df = aggregated.copy()

# ── DROP UNNEEDED COLUMNS & SAVE ────────────────────────────────────────────
df = df.drop(['County and Dublin Postal District', 'Dwelling Type'], axis=1, errors='ignore')
df.to_csv('cleanedData.csv', index=False)
print("Final shape:", df.shape)
print("Final columns:", df.columns.tolist())
print("\nDwelling category final:", df['Dwelling_Category'].value_counts())
