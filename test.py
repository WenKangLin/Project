import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIG_W = 12
FIG_H = 5
PALETTE = {'House': '#8e90cf', 'Apartment': '#d2cd90'}

# ----- Set up -----
df = pd.read_csv('cleanedData.csv')
BER = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C', 'D', 'E', 'F-G']
df['BER Rating'] = pd.Categorical(df['BER Rating'], categories=BER, ordered=True)
BER_MAP = {'A1':10, 'A2':9, 'A3':8, 'B1':7, 'B2':6, 'B3':5, 'C':4, 'D':3, 'E':2, 'F-G':1}
df['BER Score'] = df['BER Rating'].astype(str).map(BER_MAP)

# ----- Graph 8: Buildings Rated C or Below -----
total = df['VALUE'].sum()

poor     = df[df['BER Rating'].isin(['C', 'D', 'E', 'F-G'])]
poor_grp = poor.groupby(['BER Rating', 'Dwelling_Category'])['VALUE'].sum().reset_index()
poor_grp['PCT'] = (poor_grp['VALUE'] / total) * 100

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
sns.barplot(data=poor_grp, x='BER Rating', y='PCT', hue='Dwelling_Category',
            palette=PALETTE, ax=ax, order=['C', 'D', 'E', 'F-G'])
for bar in ax.patches:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    textcoords='offset points', xytext=(0, 5),
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='black')
ax.set_title('Buildings Rated C or Below by Dwelling Type', fontsize=13, fontweight='bold')
ax.set_ylabel('% of All Buildings')
ax.set_xlabel('BER Rating')
ax.legend(title='Dwelling Type')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
