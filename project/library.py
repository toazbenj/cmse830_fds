import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# Data Cleaning Steps
def data_cleaning():
    df_cobots = pd.read_excel("project/data/cobot_dataset.xlsx")
    df_cobots_original = df_cobots.copy()

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
    df_cobots['time'] =  df_cobots['time'] - df_cobots['time'].iloc[0]

    # Imputation
    cols_interpolate = [col for col in df_cobots.columns if col != 'Robot_ProtectiveStop']
    df_cobots[cols_interpolate] = df_cobots[cols_interpolate].interpolate(method='linear')
    df_cobots.fillna({'Robot_ProtectiveStop': 0}, inplace=True)

    # Make it match Protective Stop type
    df_cobots['grip_lost'] = df_cobots['grip_lost'].astype(int)

    return df_cobots, df_cobots_original

# ====================================================================================================================
# Processing Steps
# def missingness_heatmap(df_cobots):
#     # plt.style.use('dark_background')
#     fig, ax = plt.subplots(figsize=(16, 8))
#     # fig.patch.set_facecolor('#0E1117')
#     # ax.set_facecolor('#0E1117')

#     cols = []
#     feature_type_lst = ["Current", "Speed", "Temperature"]
#     for feature_type in feature_type_lst:
#         cols += [f"{feature_type}_J{i}" for i in range(1, 6)]

#     df_subset = df_cobots[cols]


#     sns.heatmap(df_subset.isna(), cbar=False, cmap='viridis', yticklabels=False)
#     plt.xlabel('Features')
#     # plt.title('Missing Values Heatmap')
#     plt.tight_layout()

#     return fig


def missingness_heatmap(df_cobots):
    # plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    feature_type_lst = ["Current", "Speed", "Temperature"]
    cols = [f"{ft}_J{i}" for ft in feature_type_lst for i in range(1, 6)]
    df_subset = df_cobots[cols]

    sns.heatmap(
        df_subset.isna(),
        cbar=True,
        cmap='viridis',
        yticklabels=False, 
        ax=ax
    )

    # Rotate and fit labels nicely
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Row Index', fontsize=12)
    # ax.set_title('Missing Values Heatmap', fontsize=14, pad=12)

    # Improve label rotation for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    return fig

def interpolation_example(df_cobots, df_cobots_original, feature_type='Current'):
    colors = px.colors.qualitative.Dark24
    feature = f"{feature_type}_J2"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=(f'Original {feature}', f'Interpolated {feature}'),
        vertical_spacing=0.1
    )

    # Plot original and interpolated data
    fig.add_trace(
        go.Scatter(
            x=df_cobots['time'],
            y=df_cobots_original[feature],
            name=f"Original {feature}",
            mode='lines',
            line=dict(color=colors[1])
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_cobots['time'],
            y=df_cobots[feature],
            name=f"Interpolated {feature}",
            mode='lines',
            line=dict(color=colors[0])
        ),
        row=2, col=1
    )

    # Detect missing segments
    mask = df_cobots_original[feature].isna()
    time_col = df_cobots['time']

    if mask.any():
        missing_groups = np.split(np.where(mask)[0], np.where(np.diff(np.where(mask)[0]) > 1)[0] + 1)
        for group in missing_groups:
            if len(group) == 0:
                continue

            # Identify preceding and following indices (if valid)
            prev_idx = group[0] - 1 if group[0] > 0 else None
            next_idx = group[-1] + 1 if group[-1] < len(df_cobots) - 1 else None

            highlight_points = []
            if prev_idx is not None:
                highlight_points.append(prev_idx)
            if next_idx is not None:
                highlight_points.append(next_idx)

            # Add large dots at edge points
            for idx in highlight_points:
                # Original plot
                fig.add_trace(
                    go.Scatter(
                        x=[time_col.iloc[idx]],
                        y=[df_cobots_original[feature].iloc[idx]],
                        mode='markers',
                        marker=dict(color='yellow', size=6, symbol='circle'),
                        name='Gap Edge',
                        showlegend=False
                    ),
                    row=1, col=1
                )
                # Interpolated plot
                fig.add_trace(
                    go.Scatter(
                        x=[time_col.iloc[idx]],
                        y=[df_cobots[feature].iloc[idx]],
                        mode='markers',
                        marker=dict(color='yellow', size=6, symbol='circle'),
                        name='Gap Edge',
                        showlegend=False
                    ),
                    row=2, col=1
                )

    # Zoom range
    zoom_range = [25800, 26100]
    fig.update_xaxes(range=zoom_range, row=1, col=1)
    fig.update_xaxes(range=zoom_range, row=2, col=1)

    # Range slider
    fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

    # Labels and layout
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text=f'Original {feature}', row=1, col=1)
    fig.update_yaxes(title_text=f'Interpolated {feature}', row=2, col=1)
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"",
        title_x=0.5
    )

    return fig



# ====================================================================================================================
# EDA Plots

def histogram_plots(df_cobots):
    feature_type_lst = ["Current", "Speed", "Temperature"]
    unit = ["A", "m/s", "Degrees C"]
    colors = px.colors.qualitative.Dark24

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f'{feature} Distribution' for feature in feature_type_lst],
        horizontal_spacing=0.1
    )

    # Loop through each feature type
    for feat_idx, (feature_type, unit_label) in enumerate(zip(feature_type_lst, unit)):
        row = feat_idx + 1  
        
        # Add histogram for each joint (J0-J5)
        for joint_idx in range(6):
            col_name = f"{feature_type}_J{joint_idx}"
            
            fig.add_trace(
                go.Histogram(
                    x=df_cobots[col_name],
                    name=f'Joint {joint_idx}',
                    marker=dict(color=colors[joint_idx]),
                    opacity=0.7,
                    legendgroup=f'joint{joint_idx}',  # Group by joint for legend
                    showlegend=(feat_idx == 0)  # Only show legend once (on first subplot)
                ),
                row=row, col=1
            )
        
        fig.update_xaxes(title_text=f"{feature_type} ({unit_label})", row=row, col=1)

    fig.update_yaxes(title_text="Count", row=1, col=1)

    # Update layout
    fig.update_layout(
        height=1000,
    #  title_text="Joint Feature Distributions",
        barmode='overlay',  
        showlegend=True,
        legend=dict(
            title="Joints",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    return fig

# https://plotly.com/python/time-series/
#  Claude Sonnet 4.5, 10-11-25

def time_series_plots(df_cobots):
    # Define column groups
    feature_type_lst = ["Current", "Speed", "Temperature"]
    unit = ["A", "m/s", "Degrees C"]
    colors = px.colors.qualitative.Dark24

    fig_lst = []

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
        fig_lst.append(fig)

    return fig_lst

# Failure Events Heatmap
def failure_events_heatmap(df_cobots):
    # Transpose so time is on y-axis and features on x-axis
    df_failures = df_cobots[['Robot_ProtectiveStop', 'grip_lost']].T

    # for black background
    # plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(18, 4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    sns.heatmap(df_failures, cmap='viridis', cbar_kws={"label": "Failure Events"})
    plt.xlabel('Time Index')
    plt.title('Failures Over Time Heatmap')
    plt.tight_layout()

    return fig


# Correlation Heatmap
# Scaled using Cloud Sonnet 4.5, 10-11-25

def joint_correlation_heatmaps(df_cobots):
    feature_type_lst = ["Current", "Speed", "Temperature"]

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
    return fig

def feature_correlation_heatmaps(df_cobots):
    # Correlations by feature type
    feature_type_lst = ["Current", "Speed", "Temperature"]

    feature_pairs = list(combinations(feature_type_lst, 2))

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f'{pair[0]} vs {pair[1]}' for pair in feature_pairs],
        horizontal_spacing=0.125
    )

    for pair_idx, (feat1, feat2) in enumerate(feature_pairs):
        cols = []
        for joint_idx in range(6):
            cols.append(f"{feat1}_J{joint_idx}")
        for joint_idx in range(6):
            cols.append(f"{feat2}_J{joint_idx}")
        
        df_corr = df_cobots[cols].corr().round(2)
        
        # Mask upper triangle
        mask = np.zeros_like(df_corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        
        # Apply mask and drop empty rows/cols
        df_corr_viz = df_corr.mask(mask).dropna(how='all').dropna(axis='columns', how='all')

        # Create text array with blanks instead of nan
        text_values = df_corr_viz.values.astype(str)
        text_values[text_values == 'nan'] = ''
        
        col = pair_idx + 1  # 1, 2, or 3
        
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
                showscale=(pair_idx == 2)
            ),
            row=1, col=col
        )
        
        fig.update_xaxes(tickangle=-45, row=1, col=col)

    fig.update_layout(
        height=450,
    #  title_text="Cross-Feature Correlation Analysis",
        showlegend=False
    )

    return fig
