import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ====================================================================================================================

# Data Cleaning Steps

df_cobots = pd.read_excel("project/data/cobot_dataset.xlsx")

# redundant
df_cobots.drop(columns=['Num'], inplace=True)

# miss named
df_cobots.rename(columns={'Temperature_T0': 'Temperature_J0'}, inplace=True)

# extra space
df_cobots.rename(columns={'cycle ': 'cycle'}, inplace=True)

# Encoding timestamps

# Convert to datetime
df_cobots['Timestamp'] = df_cobots['Timestamp'].astype(str).str.strip('"').str.strip("'").str.strip()
df_cobots['Timestamp'] = pd.to_datetime(df_cobots['Timestamp'], format='mixed', utc=True)

# Extract components, only use hour or shorter since all done in a single day
df_cobots['hour'] = df_cobots['Timestamp'].dt.hour
df_cobots['minute'] = df_cobots['Timestamp'].dt.minute
df_cobots['second'] = df_cobots['Timestamp'].dt.second

df_cobots.drop(columns=['Timestamp'], inplace=True)

# Time feature cleaning
df_cobots['time'] = (
    df_cobots['hour'] * 3600 + df_cobots['minute'] * 60 + df_cobots['second']
)
df_cobots = df_cobots.sort_values('time')

# Imputation
cols_interpolate = [col for col in df_cobots.columns if col != 'Robot_ProtectiveStop']
df_cobots[cols_interpolate] = df_cobots[cols_interpolate].interpolate(method='linear')
df_cobots.fillna({'Robot_ProtectiveStop': 0}, inplace=True)

# ====================================================================================================================

# Visuals


st.title("Robot Performance Analysis and Failure Prediction")
st.title("By Ben Toaz")

st.set_page_config(page_title="Robot Joint Monitoring", layout="wide")

# https://plotly.com/python/time-series/
#  Claude Sonnet 4.5, 10-11-25

st.header("Time Series Data")

# Define your column groups
feature_type_lst = ["Current", "Speed", "Temperature"]
unit = ["A", "m/s", "Degrees C"]
colors = px.colors.qualitative.Dark24

for feature_type, unit in zip(feature_type_lst, unit):
   cols1 = [f"{feature_type}_J{i}" for i in range(0, 3)]
   cols2 = [f"{feature_type}_J{i}" for i in range(3, 6)]

   # Create subplots
   fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                     subplot_titles=(f'{feature_type} Joints 0-2', f'{feature_type} Joints 3-5'),
                     vertical_spacing=0.1)

   # Add current traces to first subplot
   for i, col in enumerate(cols1):
      fig.add_trace(go.Scatter(x=df_cobots['time'], y=df_cobots[col],
                              name=col, mode='lines', line=dict(color=colors[i])), row=1, col=1)

   # Add speed traces to second subplot
   for i, col in enumerate(cols2):
      fig.add_trace(go.Scatter(x=df_cobots['time'], y=df_cobots[col], 
                              name=col, mode='lines', line=dict(color=colors[i+3])), row=2, col=1)

   # Add rangeslider to bottom subplot only
   fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

   fig.update_xaxes(title_text="Time (s)", row=2, col=1, rangeslider_visible=True)
   fig.update_yaxes(title_text=f"{feature_type} ({unit})", row=1, col=1)
   fig.update_yaxes(title_text=f"{feature_type} ({unit})", row=2, col=1)

   # Update layout
   fig.update_layout(height=800)
   # fig.show()

   # Display in Streamlit
   st.plotly_chart(fig, use_container_width=True)


# Correlation Heatmap
# Scaled using Cloud Sonnet 4.5, 10-11-25

st.header("Correlation Analysis by Joint")

# Create 2x3 subplots
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=[f'Joint {i}' for i in range(6)],
    vertical_spacing=0.225,
    horizontal_spacing=0.125
)

# Loop through joints 0-5
for joint_idx in range(6):
    # Select columns for this joint
    cols = []
    for feature_type in feature_type_lst:
        cols.append(f"{feature_type}_J{joint_idx}")
    
    cols += ['Robot_ProtectiveStop', 'grip_lost', 'cycle', 'Tool_current']
    
    # Calculate correlation
    df_corr = df_cobots[cols].corr().round(2)
    
    # Mask upper triangle
    mask = np.zeros_like(df_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Apply mask and drop empty rows/cols
    df_corr_viz = df_corr.mask(mask).dropna(how='all').dropna(axis='columns', how='all')

    # Create text array with blanks instead of nan
    text_values = df_corr_viz.values.astype(str)
    text_values[text_values == 'nan'] = ''
    
    # Calculate position in grid
    row = (joint_idx // 3) + 1  # 1 or 2
    col = (joint_idx % 3) + 1   # 1, 2, or 3
    
    # Add heatmap to subplot
    fig.add_trace(
        go.Heatmap(
            z=df_corr_viz.values,
            x=df_corr_viz.columns,
            y=df_corr_viz.index,
            colorscale='Viridis',
            zmid=0,
            text=text_values,
            texttemplate='%{text}',
            textfont={"size": 8},
            showscale=(joint_idx == 5)  # Only show colorbar on last plot
        ),
        row=row, col=col
    )
    
    # Update axes for this subplot
    fig.update_xaxes(tickangle=-45, row=row, col=col)

fig.update_layout(
    height=800,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)
