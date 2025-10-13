import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

st.set_page_config(page_title="Robot Joint Monitoring", layout="wide")

st.title("🤖 Robot Joint Monitoring Dashboard")

# Configuration
feature_type_lst = ["Current", "Speed", "Temperature"]
unit = ["A", "m/s", "Degrees C"]
colors = px.colors.qualitative.Dark24

for feature_type, unit_label in zip(feature_type_lst, unit):
    cols1 = [f"{feature_type}_J{i}" for i in range(0, 3)]
    cols2 = [f"{feature_type}_J{i}" for i in range(3, 6)]

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        subplot_titles=(f'{feature_type} Joints 0-2', f'{feature_type} Joints 3-5'),
        vertical_spacing=0.1
    )

    # Add traces to first subplot
    for i, col in enumerate(cols1):
        fig.add_trace(
            go.Scatter(
                x=df_cobots['time'], 
                y=df_cobots[col],
                name=col, 
                mode='lines', 
                line=dict(color=colors[i])
            ), 
            row=1, col=1
        )

    # Add traces to second subplot
    for i, col in enumerate(cols2):
        fig.add_trace(
            go.Scatter(
                x=df_cobots['time'], 
                y=df_cobots[col], 
                name=col, 
                mode='lines', 
                line=dict(color=colors[i+3])
            ), 
            row=2, col=1
        )

    # Add rangeslider to bottom subplot only
    fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text=f"{feature_type} ({unit_label})", row=1, col=1)
    fig.update_yaxes(title_text=f"{feature_type} ({unit_label})", row=2, col=1)

    # Update layout
    fig.update_layout(
        height=800, 
        title=f'{feature_type} Time Series',
        showlegend=True
    )
    
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)