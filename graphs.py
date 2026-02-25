import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import requests


# -----Set up -----
df = pd.read_csv('cleanedData.csv')
BER = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C', 'D', 'E', 'F-G']
df['BER Rating'] = pd.Categorical(df['BER Rating'], categories=BER, ordered=True)
BER_MAP = {'A1':10, 'A2':9, 'A3':8, 'B1':7, 'B2':6, 'B3':5, 'C':4, 'D':3, 'E':2, 'F-G':1}
df['BER Score'] = df['BER Rating'].astype(str).map(BER_MAP)


# ----- Graph 1: Total buildings by BER Rating & Dwelling Type ────────────────────
graph1, ax = plt.subplots(figsize=(12, 6))
grp1 = df.groupby(['BER Rating','Dwelling_Category'])['VALUE'].sum().reset_index()
sns.barplot(data=grp1, x='BER Rating', y='VALUE', hue='Dwelling_Category',
            palette={'House':'#1976D2','Apartment':'#FF8F00'}, ax=ax)
ax.set_title('Total Buildings by BER Rating & Dwelling Type (2020-2025)', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Buildings')
ax.set_xlabel('BER Rating')
ax.legend(title='Dwelling Type')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('G1_BER_Dwelling.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()


# ----- Graph 2: Heatmap — BER x Period of Construction (House vs Apartment) -----
graph2, axes = plt.subplots(1, 2, figsize=(16, 6))
for i, cat in enumerate(['House', 'Apartment']):
    sub = df[df['Dwelling_Category']==cat].groupby(
        ['Period of Construction','BER Rating'])['VALUE'].sum().reset_index()
    pivot = sub.pivot_table(index='Period of Construction', columns='BER Rating',
                            values='VALUE', aggfunc='sum', fill_value=0)
    pivot = pivot.reindex(columns=BER)
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[i],
                linewidths=0.5, cbar_kws={'label': 'No. Buildings'}, annot_kws={'size': 9})
    axes[i].set_title(f'{cat}: BER x Construction Period', fontweight='bold', fontsize=12)
    axes[i].set_xlabel('BER Rating')
    axes[i].set_ylabel('Period of Construction')
    axes[i].tick_params(axis='x', rotation=0)
    axes[i].tick_params(axis='y', rotation=0)
graph2.suptitle('Building Energy Ratings by Construction Period', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('G2_BER_Construction.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()


# ----- Graph 3: % A-Rated by Construction Period ────────────────────────────
a_rated = df[df['BER Rating'].isin(['A1','A2','A3'])]
total = df.groupby(['Period of Construction','Dwelling_Category'])['VALUE'].sum().reset_index(name='Total')
a_period = a_rated.groupby(['Period of Construction','Dwelling_Category'])['VALUE'].sum().reset_index(name='A_rated')
merged = a_period.merge(total, on=['Period of Construction','Dwelling_Category'])
merged['A_percentage'] = (merged['A_rated'] / merged['Total']) * 100

graph3, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=merged, x='Period of Construction', y='A_percentage', hue='Dwelling_Category',
            palette={'House':'#1976D2','Apartment':'#FF8F00'}, ax=ax)
for bar in ax.patches:
    height = bar.get_height()
    bar_radius = bar.get_width() / 2
    if height > 0:
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar_radius, height / 2),
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
ax.set_title('% A-Rated Buildings (A1–A3) by Period of Construction & Dwelling Type',
             fontsize=13, fontweight='bold')
ax.set_ylabel('% A-Rated')
ax.set_xlabel('Construction Period')
ax.tick_params(axis='x')
ax.legend(title='Dwelling Type')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('G3_A-Rated_Construction.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()


# ----- Graph 4: GeoMap of Avg BER -----
URL = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/ireland-counties.geojson"
response = requests.get(URL)
with open('countyMap.geojson', 'wb') as f:
    f.write(response.content)

gdf = gpd.read_file('countyMap.geojson')

def clean_geojson(name):
    name = str(name).strip()
    name = name.replace(' County', '').replace(' City', '').strip()
    if 'Tipperary' in name:  return 'Tipperary'
    if 'Limerick'  in name:  return 'Limerick'
    if 'Waterford' in name:  return 'Waterford'
    if 'Galway'    in name:  return 'Galway'
    if 'Cork'      in name:  return 'Cork'
    if name in ['Dublin', 'South Dublin', 'Fingal', 'Dún Laoghaire-Rathdown']:
        return 'Dublin'
    return name

gdf['County'] = gdf['name'].apply(clean_geojson)
gdf = gdf.dissolve(by='County').reset_index()
print("GeoJSON counties:", sorted(gdf['County'].unique()))
print("Your data counties:", sorted(df['County'].unique()))

# ----- Plot -----
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

for i, cat in enumerate(['House', 'Apartment']):
    cat_df = df[df['Dwelling_Category'] == cat]
    county_ber = cat_df.groupby('County').apply(
        lambda g: (g['BER Score'] * g['VALUE']).sum() / g['VALUE'].sum(),
        include_groups=False
    ).reset_index(name='Avg_BER_Score')

    merged_gdf = gdf.merge(county_ber, on='County', how='left')

    merged_gdf.plot(
        column='Avg_BER_Score', ax=axes[i], cmap='RdYlGn',
        vmin=1, vmax=10, legend=True, edgecolor='white', linewidth=0.5,
        missing_kwds={'color': 'lightgrey', 'label': 'No data'},
        legend_kwds={'label': 'Avg BER Score (1=F-G → 10=A1)', 'shrink': 0.6}
    )

    for _, row in merged_gdf.iterrows():
        if row['geometry'] is not None:
            centroid = row['geometry'].centroid
            axes[i].annotate(row['County'],
                             xy=(centroid.x, centroid.y),
                             ha='center', fontsize=6.5,
                             fontweight='bold', color='black')

    axes[i].set_title(f'{cat}s', fontsize=14, fontweight='bold')
    axes[i].axis('off')

fig.suptitle('Ireland: Average BER Score by County\n(Green = Better, Red = Worse)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('G4_BER_Map.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved")
