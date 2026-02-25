import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import requests

# ----- Set up -----
df = pd.read_csv('cleanedData.csv')
BER = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C', 'D', 'E', 'F-G']
df['BER Rating'] = pd.Categorical(df['BER Rating'], categories=BER, ordered=True)
BER_MAP = {'A1':10, 'A2':9, 'A3':8, 'B1':7, 'B2':6, 'B3':5, 'C':4, 'D':3, 'E':2, 'F-G':1}
df['BER Score'] = df['BER Rating'].astype(str).map(BER_MAP)

