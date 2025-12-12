import streamlit as st
import pandas as pd
import plotly.express as px
from urbanmart_analysis import compute_daily_sales

st.set_page_config(layout='wide', page_title='UrbanMart Dashboard')

st.title("UrbanMart – Daily Revenue Dashboard")
st.write("Business Owner View: Understand how revenue is generated each day.")

df_daily = compute_daily_sales('urbanmart_sales.csv')

min_date = df_daily.index.min().date()
max_date = df_daily.index.max().date()

start = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
end = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

df_filtered = df_daily.loc[pd.to_datetime(start):pd.to_datetime(end)]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"{df_filtered['daily_revenue'].sum():.2f}")
col2.metric("Total Days", f"{len(df_filtered)}")
col3.metric("Avg Revenue / Day", f"{df_filtered['daily_revenue'].mean():.2f}")
col4.metric("Total Transactions", f"{df_filtered['daily_transactions'].sum():.0f}")

fig = px.line(df_filtered.reset_index(), x='date', y='daily_revenue',
              title='Daily Revenue Trend', markers=True)
fig.add_scatter(x=df_filtered.index, y=df_filtered['moving_avg_7d'],
                mode='lines', name='7-day Moving Avg')
st.plotly_chart(fig, use_container_width=True)

st.subheader("Detailed Daily Revenue Table")
st.dataframe(df_filtered)

csv = df_filtered.to_csv().encode('utf-8')
st.download_button("Download Daily Sales CSV", csv, "daily_sales_filtered.csv")
