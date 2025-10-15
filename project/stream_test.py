import streamlit as st
from library import *

df_cobots = data_cleaning()


st.set_page_config(page_title="Robot Joint Monitoring", layout="wide")
st.title("Robot Performance Analysis and Failure Prediction")
st.title("By Ben Toaz")

st.header("Joint Feature Distributions")
fig = histogram_plots(df_cobots)
st.plotly_chart(fig, use_container_width=True)

st.header("Time Series Data")
fig_lst = time_series_plots(df_cobots)
for fig in fig_lst:
   st.plotly_chart(fig, use_container_width=True)

st.header("Failure Events")
fig = failure_events_heatmap(df_cobots)
st.pyplot(fig, clear_figure=True)

st.header("Correlation Analysis by Joint")
fig = joint_correlation_heatmaps(df_cobots)
st.plotly_chart(fig, use_container_width=True)

st.header("Cross-Feature Correlation Analysis")
fig = feature_correlation_heatmaps(df_cobots)
st.plotly_chart(fig, use_container_width=True)
