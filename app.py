import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import geopandas as gpd
import requests

# ----- CONSTS -----
# BER_CMAP = LinearSegmentedColormap.from_list(
#     'ber', ['#a5b289', '#b7905c', '#a94405', '#91232f', '#6e1449']
# )
GEO_CMAP = LinearSegmentedColormap.from_list(
    'geo', ['#e61d3e', '#fc5971', '#fc691a', '#f9ad08', '#b3d51e', '#42cc51']
)
BER_CMAP = GEO_CMAP.reversed()

BER = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C', 'D', 'E', 'F-G']
PALETTE = {'House': '#8e90cf', 'Apartment': '#d2cd90'}
FIG_W = 12
FIG_H = 5

# ----- Page Config -----
st.set_page_config(page_title="Ireland BER Dashboard", layout="wide")
st.title("Ireland Domestic Building Energy Ratings (2020 – 2025)")
st.markdown("Analysis of BER ratings across years, counties, dwelling types and construction periods.")

# ----- Data Setup -----
@st.cache_data
def load_data():
    df = pd.read_csv('cleanedData.csv')
    df['BER Rating'] = pd.Categorical(df['BER Rating'], categories=BER, ordered=True)
    BER_MAP = {'A1': 10, 'A2': 9, 'A3': 8, 'B1': 7, 'B2': 6,
               'B3': 5, 'C': 4, 'D': 3, 'E': 2, 'F-G': 1}
    df['BER Score'] = df['BER Rating'].astype(str).map(BER_MAP)
    return df

df = load_data()

# ----- GeoJSON Setup -----
@st.cache_data
def load_geo():
    URL = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/ireland-counties.geojson"
    response = requests.get(URL)
    with open('countyMap.geojson', 'wb') as f:
        f.write(response.content)
    gdf = gpd.read_file('countyMap.geojson')

    def clean(name):
        name = str(name).strip().replace(' County', '').replace(' City', '')
        if 'Tipperary' in name: return 'Tipperary'
        if 'Limerick'  in name: return 'Limerick'
        if 'Waterford' in name: return 'Waterford'
        if 'Galway'    in name: return 'Galway'
        if 'Cork'      in name: return 'Cork'
        if name in ['Dublin', 'South Dublin', 'Fingal', 'Dún Laoghaire-Rathdown']:
            return 'Dublin'
        return name

    gdf['County'] = gdf['name'].apply(clean)
    return gdf.dissolve(by='County').reset_index()

gdf = load_geo()

# ----- Sidebar Filters -----
st.sidebar.header("Filters")
st.divider()

selected_cat = st.sidebar.radio(
    "Dwelling Type",
    options=["House", "Apartment", "Both"],
    index=2,
    horizontal=True
)

min_year = int(df['Year'].min())
max_year = int(df['Year'].max())

year_range = st.sidebar.slider(
    "Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1
)

st.sidebar.markdown("Period of Construction")
all_periods = sorted(df['Period of Construction'].dropna().unique().tolist())
selected_periods = []
for period in all_periods:
    if st.sidebar.toggle(period, value=True, key=f"toggle_{period}"):
        selected_periods.append(period)
if len(selected_periods) == 0:
    selected_periods = all_periods

if selected_cat == "Both":
    cats = ["House", "Apartment"]
else:
    cats = [selected_cat]

df_filtered = df[
    df['Dwelling_Category'].isin(cats) &
    df['Year'].between(year_range[0], year_range[1]) &
    df['Period of Construction'].isin(selected_periods)
]

st.sidebar.divider()
st.sidebar.markdown(f"**Showing:** {year_range[0]} – {year_range[1]}")
st.sidebar.markdown(f"**Rows in view:** {len(df_filtered):,}")

# ========== SECTION 1 ==========
st.header("Section 1: National Overview")
st.markdown("A high level insight into BER performance for selected features.")

total_buildings  = int(df_filtered['VALUE'].sum())
a_rated_total    = int(df_filtered[df_filtered['BER Rating'].isin(['A1', 'A2', 'A3'])]['VALUE'].sum())
pct_a_rated      = (a_rated_total / total_buildings * 100) if total_buildings > 0 else 0
poor_rated_total = int(df_filtered[df_filtered['BER Rating'].isin(['C', 'D', 'E', 'F-G'])]['VALUE'].sum())
pct_poor         = (poor_rated_total / total_buildings * 100) if total_buildings > 0 else 0

county_avg = df_filtered.groupby('County').apply(
    lambda g: (g['BER Score'] * g['VALUE']).sum() / g['VALUE'].sum(),
    include_groups=False
)
best_county  = county_avg.idxmax()
worst_county = county_avg.idxmin()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Buildings",   f"{total_buildings:,}")
col2.metric("A-Rated (A1–A3)",   f"{a_rated_total:,}",    f"{pct_a_rated:.1f}%")
col3.metric("C-Rated or Below",  f"{poor_rated_total:,}", f"{pct_poor:.1f}%")
col4.metric("Best County",       best_county,             f"{county_avg[best_county]:.1f}/10")
col5.metric("Worst County",      worst_county,            f"{county_avg[worst_county]:.1f}/10")

st.divider()

# ----- GRAPH 1: BER Rating Distribution by Dwelling Type -----
st.subheader("BER Rating Distribution by Dwelling Type")
st.markdown("Shows the total number of buildings at each BER rating level, split by houses and apartments.")

def graph1(data):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    grp = data.groupby(['BER Rating', 'Dwelling_Category'], observed=True)['VALUE'].sum().reset_index()
    sns.barplot(data=grp, x='BER Rating', y='VALUE', hue='Dwelling_Category',
                palette=PALETTE, ax=ax, order=BER)
    for bar in ax.patches:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f"{height:,.0f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        textcoords='offset points', xytext=(0, 5),
                        ha='center', va='bottom', fontsize=8,
                        fontweight='bold', color='black')
    ax.set_title('Total Buildings by BER Rating & Dwelling Type', fontweight='bold')
    ax.set_ylabel('Number of Buildings')
    ax.set_xlabel('BER Rating')
    ax.legend(title='Dwelling Type')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

st.pyplot(graph1(df_filtered))
st.markdown("**Insight:** With no filters, we can see over past 5 years, there has been a high concentration of housing being built with a A2 Rating or higher. What gives us an interesting graph is when we include or exclude periods of construction from the 2005 regulations and also the most recent 2019 regulations")

st.divider()

# ----- GRAPH 2: BER Rating Distribution under C -----
st.subheader("Proportion of dwellings C rated and below")
st.markdown("Shows percentage of dwellings that are below or just meet the 2005 regulations")
def graph2(data):
    total = data['VALUE'].sum()
    poor = data[data['BER Rating'].isin(['C', 'D', 'E', 'F-G'])]
    poor_grp = poor.groupby(['BER Rating', 'Dwelling_Category'], observed=True)['VALUE'].sum().reset_index()
    poor_grp['PCT'] = (poor_grp['VALUE'] / total) * 100

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.barplot(data=poor_grp, x='BER Rating', y='PCT', hue='Dwelling_Category',
                palette=PALETTE, ax=ax, order=['C', 'D', 'E', 'F-G'])
    for bar in ax.patches:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        textcoords='offset points', xytext=(0, 5),
                        ha='center', va='bottom', fontsize=8,
                        fontweight='bold', color='black')
    ax.set_title('Buildings Rated C or Below by Dwelling Type', fontweight='bold')
    ax.set_ylabel('% of All Buildings')
    ax.set_xlabel('BER Rating')
    ax.legend(title='Dwelling Type')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

st.pyplot(graph2(df_filtered))
st.markdown("**Insight:** Houses that are just on the minimum requirements and below from the 2005 regulations make up for nearly 1/5 of the data. If we play around with the period of construction we see a clear decrease in the numbers over time.")

st.divider()

# ========== SECTION 2 ==========
st.header("📈 Section 2: Trends Over Time")
st.markdown("Examines how average BER performance has changed year on year and which counties have improved the most.")

# ----- GRAPH 3: Year-over-Year Avg BER Trend -----
st.subheader("BER Score Trend by Year")
st.markdown("Tracks the national average BER score from 2020 to 2025 for houses and apartments. An upward trend indicates the dwellings are becoming more energy efficient over time.")

def graph3(data):
    year_ber = data.groupby(['Year', 'Dwelling_Category']).apply(
        lambda g: (g['BER Score'] * g['VALUE']).sum() / g['VALUE'].sum(),
        include_groups=False
    ).reset_index(name='Avg_BER_Score')
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.lineplot(data=year_ber, x='Year', y='Avg_BER_Score',
                 hue='Dwelling_Category', marker='o', linewidth=2.5,
                 palette=PALETTE, ax=ax)
    for _, row in year_ber.iterrows():
        ax.annotate(f"{row['Avg_BER_Score']:.2f}",
                    xy=(row['Year'], row['Avg_BER_Score']),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Average BER Score by Year & Dwelling Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg BER Score (1=F-G → 10=A1)')
    ax.set_xlabel('Year')
    ax.set_xticks(sorted(data['Year'].unique()))
    ax.legend(title='Dwelling Type')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

st.pyplot(graph3(df_filtered))
st.markdown("**Insight:** A consistent upward trend indicates that new builds and refurbishments are pushing average scores higher each year. Dip for apartments in 2024 suggests that there were more refurbishments happening on apartments since this period is after the regulations where the minimum would have to be an A2 or above.")

st.divider()

# ----- GRAPH 4: Most Improved County -----
st.subheader("Most Improved County (2020 → 2025)")
st.markdown("Compares each county's average BER score in the first and last year of the dataset to identify which counties have made the greatest energy efficiency gains.")

def graph4(data):
    yr_min = data['Year'].min()
    yr_max = data['Year'].max()

    def county_score(d):
        total = d['VALUE'].sum()
        if total == 0:
            return None
        return (d['BER Score'] * d['VALUE']).sum() / total

    score_start = (
        data[data['Year'] == yr_min]
        .groupby('County')
        .apply(county_score, include_groups=False)
        .reset_index(name='Score_Start')
    )
    score_end = (
        data[data['Year'] == yr_max]
        .groupby('County')
        .apply(county_score, include_groups=False)
        .reset_index(name='Score_End')
    )

    merged = score_start.merge(score_end, on='County').dropna()
    merged['Improvement'] = merged['Score_End'] - merged['Score_Start']
    merged = merged.sort_values('Improvement', ascending=True)

    colors = ['#fc5971' if x < 0 else '#42cc51' for x in merged['Improvement']]

    fig, ax = plt.subplots(figsize=(FIG_W, max(6, len(merged) * 0.38)))
    bars = ax.barh(merged['County'], merged['Improvement'], color=colors, edgecolor='white')
    for bar, val in zip(bars, merged['Improvement']):
        ax.annotate(f"{val:+.2f}",
                    xy=(val, bar.get_y() + bar.get_height() / 2),
                    textcoords='offset points',
                    xytext=(5 if val >= 0 else -5, 0),
                    ha='left' if val >= 0 else 'right',
                    va='center', fontsize=8, fontweight='bold', color='black')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_title(f'Change in Avg BER Score by County ({yr_min} → {yr_max})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Change in Avg BER Score (points)')
    ax.set_ylabel('')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

st.pyplot(graph4(df_filtered))
st.markdown("**Insight:** Green bars show counties making meaningful progress, likely driven by higher rates of new A-rated builds or successful refurbishments. "
            "It was to my expectation that bigger cities, Dublin, Cork, Galway wouldn't be at the top of improvements but rather in the middle due to more prebuilt housing "
            "that would refurnish dwellings which don't require to meet the A2 requirements and also more so for Dublin and Cork, there's a lack of space for new developments.")

st.divider()

# ========== SECTION 3 ==========
st.header("Section 3: Period of Construction Analysis")
st.markdown("Breaking down BER performance by the age of the building to explain the 'why' behind the national picture. Building regulations tightened significantly from 2005 and again post-2011 and most recently 2019.")

# ----- GRAPH 5: % A-Rated by Construction Period -----
st.subheader("A-Rated Buildings by Construction Period")
st.markdown("Shows what percentage of buildings in each construction era achieve an A rating (A1–A3). This directly shows the impact of stricter building regulations introduced over time.")

def graph5(data):
    a_rated = data[data['BER Rating'].isin(['A1', 'A2', 'A3'])]
    total   = data.groupby(['Period of Construction', 'Dwelling_Category'])['VALUE'].sum().reset_index(name='Total')
    a_grp   = a_rated.groupby(['Period of Construction', 'Dwelling_Category'])['VALUE'].sum().reset_index(name='A_rated')
    merged  = a_grp.merge(total, on=['Period of Construction', 'Dwelling_Category'])
    merged['pct_A'] = (merged['A_rated'] / merged['Total']) * 100
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.barplot(data=merged, x='Period of Construction', y='pct_A',
                hue='Dwelling_Category', palette=PALETTE, ax=ax, width=0.5)
    for bar in ax.patches:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        textcoords='offset points', xytext=(0, 5),
                        ha='center', va='bottom', fontsize=9,
                        fontweight='bold', color='black')
    ax.set_title('% A-Rated Buildings by Construction Period', fontweight='bold')
    ax.set_ylabel('% A-Rated')
    ax.set_xlabel('Construction Period')
    ax.legend(title='Dwelling Type')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

st.pyplot(graph5(df_filtered))
st.markdown("**Insight:** Buildings constructed pre 2005 had regulations for new buildings to have C or D BER rating depending on certain criteria. "
            "Buildings constructed post-2011 show dramatically higher A-ratings, reflecting the EU Energy Performance of Buildings Directive requirements "
            "adopted into Irish building regulations. We see this is improved again with the 2019 regulations.")

st.divider()

# ----- GRAPH 6: Avg BER Score by Construction Period -----
st.subheader("Average BER Score by Construction Period")
st.markdown("Plots the mean BER score for each construction era, allowing direct comparison of how building standards evolved across decades. Reference lines mark the A3 and C thresholds.")

def graph6(data):
    period_ber = data.groupby(['Period of Construction', 'Dwelling_Category']).apply(
        lambda g: (g['BER Score'] * g['VALUE']).sum() / g['VALUE'].sum(),
        include_groups=False
    ).reset_index(name='Avg_BER_Score')
    period_ber['Period of Construction'] = pd.Categorical(
        period_ber['Period of Construction'],
        categories=selected_periods,
        ordered=True
    )
    period_ber = period_ber.dropna(subset=['Period of Construction'])
    period_ber = period_ber.sort_values('Period of Construction')
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.lineplot(data=period_ber, x='Period of Construction', y='Avg_BER_Score',
                 hue='Dwelling_Category', marker='o', linewidth=2.5,
                 palette=PALETTE, ax=ax)
    for _, row in period_ber.iterrows():
        ax.annotate(f"{row['Avg_BER_Score']:.2f}",
                    xy=(row['Period of Construction'], row['Avg_BER_Score']),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=9, fontweight='bold')
    ax.axhline(y=8, color='green',  linestyle='--', alpha=0.4, label='A3 threshold (8)')
    ax.axhline(y=4, color='orange', linestyle='--', alpha=0.4, label='C threshold (4)')
    ax.set_title('Average BER Score by Construction Period & Dwelling Type',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg BER Score (1=F-G → 10=A1)')
    ax.set_xlabel('Construction Period')
    ax.set_ylim(1, 11)
    ax.legend(title='Dwelling Type')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

st.pyplot(graph6(df_filtered))
st.markdown("**Insight:** We observe that with the increase of construction periods we also see a correlation in BER ratings due to regulations. "
            "2000-2004 had no building regulations. 2005-2009 introduced a threshold of C/D depending on certain criteria. 2010-2014 includes the EU regulation "
            "which required new buildings to achieve a A3 rating, we can clearly see there was an improvement in BER ratings during this period. "
            "2015-2019 introduced late 2019 for new buildings to be A2 rated or higher. We see that we have reached the A3 regulation ret by the EU. "
            "2020-2024 and 2025-2029 we are very close to these A2 requirements. The delta could be attributed to the allowance for lower BER ratings for "
            "buildings being refurbished having to get a B2 rating. ")

st.divider()

# ----- GRAPH 7: Heatmap BER x Construction Period -----
st.subheader("BER Rating by Construction Period (Heatmap)")
st.markdown("A heatmap showing exactly how buildings are distributed across every BER rating and period of construction combination. Darker cells indicate higher concentrations of buildings.")

if selected_cat == "Both":
    col1, col2 = st.columns(2)
    cols_cats = [(col1, 'House'), (col2, 'Apartment')]
else:
    cols_cats = [(st.container(), selected_cat)]

for col, cat in cols_cats:
    with col:
        def graph7(data, cat):
            sub = data[data['Dwelling_Category'] == cat].groupby(
                ['Period of Construction', 'BER Rating'], observed=True)['VALUE'].sum().reset_index()
            pivot = sub.pivot_table(index='Period of Construction', columns='BER Rating', observed=True,
                                    values='VALUE', aggfunc='sum', fill_value=0)
            pivot = pivot.reindex(columns=BER)
            pivot = pivot.reindex(index=selected_periods)
            cell_w, cell_h = 0.7, 0.55
            fig_w = max(8, len(BER) * cell_w + 2)
            fig_h = max(3, len(pivot) * cell_h + 1.5)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap=BER_CMAP, ax=ax,
                        linewidths=0.5, annot_kws={'size': 9})
            ax.set_title(f'{cat}: BER × Construction Period', fontweight='bold')
            plt.tight_layout()
            return fig
        st.pyplot(graph7(df_filtered, cat))

st.markdown("**Insight:** The heatmap makes it easy to spot period of construction combinations at a glance. Buildings built in the early 2000's show heavy concentration in the D–G columns, while post-2011 buildings cluster in the A1–A3 columns, confirming the regulations are making an effect.")

st.divider()

# ========== SECTION 4 ==========
st.header("Section 4: Geographic Analysis")
st.markdown("Maps and ranks BER performance across Ireland's 26 counties to reveal regional patterns and disparities.")

# ----- GRAPH 8: Ireland Map -----
st.subheader("Average BER Score by County (Map)")
st.markdown("A choropleth map colour-coded from red (low BER score) to green (high BER score), giving an immediate visual sense of geographic energy performance across Ireland.")

def graph8(data, cat):
    cat_df = data[data['Dwelling_Category'] == cat]
    county_ber = cat_df.groupby('County').apply(
        lambda g: (g['BER Score'] * g['VALUE']).sum() / g['VALUE'].sum(),
        include_groups=False
    ).reset_index(name='Avg_BER_Score')
    merged_gdf = gdf.merge(county_ber, on='County', how='left')
    fig, ax = plt.subplots(figsize=(6, 8))
    merged_gdf.plot(column='Avg_BER_Score', ax=ax, cmap=GEO_CMAP,
                    vmin=1, vmax=10, legend=True, edgecolor='white', linewidth=0.5,
                    missing_kwds={'color': 'lightgrey', 'label': 'No data'},
                    legend_kwds={'label': 'Avg BER Score (1=F-G → 10=A1)', 'shrink': 0.6})
    for _, row in merged_gdf.iterrows():
        if row['geometry'] is not None:
            c = row['geometry'].centroid
            ax.annotate(row['County'], xy=(c.x, c.y),
                        ha='center', fontsize=5.5, fontweight='bold', color='black')
    ax.set_title(f'{cat}s — Avg BER Score by County', fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig

if selected_cat == "Both":
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(graph8(df_filtered, 'House'))
    with col2:
        st.pyplot(graph8(df_filtered, 'Apartment'))
else:
    st.pyplot(graph8(df_filtered, selected_cat))

st.markdown("**Insight:** Dublin and surrounding commuter belt counties tend to score higher due to a larger proportion of newer apartments "
            "and house builds. Rural western counties with older housing stock typically appear in the amber range with more of these refurbishing derelict housing.")

st.divider()

# ----- GRAPH 9: County Rankings -----
st.subheader("County Rankings by Average BER Score")
st.markdown("Ranks all 26 counties from worst to best average BER score, with house and apartment scores shown side by side. Reference lines mark the A3 (8) and C (5) thresholds.")

def graph9(data):
    county_ber = data.groupby(['County', 'Dwelling_Category']).apply(
        lambda g: (g['BER Score'] * g['VALUE']).sum() / g['VALUE'].sum(),
        include_groups=False
    ).reset_index(name='Avg_BER_Score')
    order = data.groupby('County').apply(
        lambda g: (g['BER Score'] * g['VALUE']).sum() / g['VALUE'].sum(),
        include_groups=False
    ).sort_values(ascending=True).index
    fig, ax = plt.subplots(figsize=(11, 12))
    sns.barplot(data=county_ber, y='County', x='Avg_BER_Score',
                hue='Dwelling_Category', order=order,
                palette=PALETTE, ax=ax)
    for bar in ax.patches:
        width = bar.get_width()
        if width > 0:
            ax.annotate(f"{width:.2f}",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        textcoords='offset points', xytext=(5, 0),
                        ha='left', va='center', fontsize=8,
                        fontweight='bold', color='black')
    ax.axvline(x=4, color='orange', linestyle='--', alpha=0.5, label='C threshold (4)')
    ax.axvline(x=8, color='green',  linestyle='--', alpha=0.5, label='A3 threshold (8)')
    ax.set_title('County Rankings — Avg BER Score (Best → Worst)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Avg BER Score (1=F-G → 10=A1)')
    ax.set_ylabel('')
    ax.legend(title='Dwelling Type', bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

st.pyplot(graph9(df_filtered))
st.markdown("**Insight:** The gap between the best and worst counties highlights significant regional inequality in housing BER ratings.")

st.divider()

st.header("Section 5: County Comparison")
st.markdown("Select any two counties to compare their BER distribution, construction era breakdown, and year-on-year trend side by side.")

col_a, col_b = st.columns(2)
with col_a:
    county_1 = st.selectbox("County 1", options=sorted(df_filtered['County'].unique()), index=0)
with col_b:
    county_2 = st.selectbox("County 2", options=sorted(df_filtered['County'].unique()), index=1)

def graph_compare(data, county, col):
    county_df = data[data['County'] == county]
    with col:
        st.subheader(f" {county}")

        # Panel 1: BER Distribution
        fig1, ax1 = plt.subplots(figsize=(FIG_W / 2, FIG_H))
        grp = county_df.groupby(['BER Rating', 'Dwelling_Category'], observed=True)['VALUE'].sum().reset_index()
        sns.barplot(data=grp, x='BER Rating', y='VALUE', hue='Dwelling_Category',
                    palette=PALETTE, ax=ax1, order=BER)
        for bar in ax1.patches:
            height = bar.get_height()
            if height > 0:
                ax1.annotate(f"{height:.0f}",
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             textcoords='offset points', xytext=(0, 5),
                             ha='center', va='bottom', fontsize=8,
                             fontweight='bold', color='black')
        ax1.set_title('BER Rating Distribution', fontweight='bold')
        ax1.set_ylabel('Number of Buildings')
        ax1.set_xlabel('BER Rating')
        ax1.legend(title='Dwelling Type')
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)

        # Panel 2: Buildings by Construction Perioda
        fig2, ax2 = plt.subplots(figsize=(FIG_W / 2, FIG_H))
        period_grp = county_df.groupby(['Period of Construction', 'Dwelling_Category'])['VALUE'].sum().reset_index()
        period_grp['Period of Construction'] = pd.Categorical(
            period_grp['Period of Construction'], categories=selected_periods, ordered=True
        )
        sns.barplot(data=period_grp, x='Period of Construction', y='VALUE',
                    hue='Dwelling_Category', palette=PALETTE, ax=ax2, width=0.5)
        for bar in ax2.patches:
            height = bar.get_height()
            if height > 0:
                ax2.annotate(f"{height:.0f}",
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             textcoords='offset points', xytext=(0, 5),
                             ha='center', va='bottom', fontsize=8,
                             fontweight='bold', color='black')
        ax2.set_title('Buildings by Construction Period', fontweight='bold')
        ax2.set_ylabel('Number of Buildings')
        ax2.set_xlabel('Construction Period')
        ax2.legend(title='Dwelling Type')
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

        # Panel 3: Avg BER Score by Year
        fig3, ax3 = plt.subplots(figsize=(FIG_W / 2, FIG_H))
        year_ber = county_df.groupby(['Year', 'Dwelling_Category']).apply(
            lambda g: (g['BER Score'] * g['VALUE']).sum() / g['VALUE'].sum(),
            include_groups=False
        ).reset_index(name='Avg_BER_Score')
        sns.lineplot(data=year_ber, x='Year', y='Avg_BER_Score',
                     hue='Dwelling_Category', marker='o', linewidth=2.5,
                     palette=PALETTE, ax=ax3)
        for _, row in year_ber.iterrows():
            ax3.annotate(f"{row['Avg_BER_Score']:.2f}",
                         xy=(row['Year'], row['Avg_BER_Score']),
                         textcoords='offset points', xytext=(0, 8),
                         ha='center', fontsize=8, fontweight='bold')
        ax3.set_title('Avg BER Score by Year', fontweight='bold')
        ax3.set_ylabel('Avg BER Score (1=F-G → 10=A1)')
        ax3.set_xlabel('Year')
        ax3.set_xticks(sorted(data['Year'].unique()))
        ax3.legend(title='Dwelling Type')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(1, 11)
        plt.tight_layout()
        st.pyplot(fig3)

        # KPI Stats
        st.markdown("**Statistics**")
        total     = int(county_df['VALUE'].sum())
        a_count   = int(county_df[county_df['BER Rating'].isin(['A1', 'A2', 'A3'])]['VALUE'].sum())
        poor      = int(county_df[county_df['BER Rating'].isin(['C', 'D', 'E', 'F-G'])]['VALUE'].sum())
        avg_score = (county_df['BER Score'] * county_df['VALUE']).sum() / county_df['VALUE'].sum()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Buildings", f"{total:,}")
        m2.metric("A-Rated",         f"{a_count:,}",  f"{a_count/total*100:.1f}% of total")
        m3.metric("C or Below",      f"{poor:,}",     f"{poor/total*100:.1f}% of total")
        m4.metric("Avg BER Score",   f"{avg_score:.2f} / 10")

col1, col2 = st.columns(2)
graph_compare(df_filtered, county_1, col1)
graph_compare(df_filtered, county_2, col2)

# ========== OUTCOME AND SUMMARY ============
st.divider()
st.header("Outcomes and Summary")
st.markdown("""
The data conveys clear pictures of Ireland in many terms: BER Ratings of buildings built before and after regulations, the year on year progress of BER ratings and the comparison between counties and more so urban vs rural comparison.

Buildings constructed post 2005 had Irish regulations to be C/D rated or higher depending on criteria. 
Buildings constructed post 2011 had EU regulations to be A ratings and refurbished buildings had to be B rated.
Buildings constructed post 2019 had Irish regulations to be A2 rated and refurbished was B2 rated.
These 3 regulations have a profound effect on our graphs above. We see the data from 2020 onwards that these regulations have a major influence on the overall ratings of these buildings. 
Newly built apartments, we are able to see that they outperform newly built houses in terms of ratings in our graph, this could be caused by  the smaller floor area of apartments which make it easier to achieve better ratings.
We are able to see that there is a majority of new buildings being built are houses instead of apartments as well.
Older dwellings are being refurbished and we can see a decrease each year of C rated dwellings and lower.

On a geographical level, over the last 5 years every county has had an average increase in their BER ratings.
We aren't too surprised about the results from the Most Improved County Graph, more than half of the counties over the last 5 years have on average gone up one BER rating.
Urban and commuter counties show stronger average BER scores, driven by a higher proportion of new dwellings. 
Looking at the average BER of each county the results aren't surprising. There seems to show a correlation of county population and ease of commute to these high population counties and higher average BER ratings. 

This data has shown the immense progress that has been made with BER ratings and in general an overall push towards self sustainability. The number of derelict housing and opportunity for more housing is still high and with the help of grants and other financial aid, these housing could be refurbished and would contribute positively towards the general BER ratings
Bridging the gap between the newly built A-rated dwellings and the pre 2000 dwellings is the central energy policy challenge visible throughout this data.
""")

# ----- Footer -----
st.divider()
st.caption("Data source: CSO Ireland — Domestic Building Energy Ratings (2020–2025)")
