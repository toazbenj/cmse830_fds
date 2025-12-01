import streamlit as st
import numpy as np
import plotly.graph_objects as go
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import time
import glob
import os
import pandas as pd


# Code GPT 5.1 11-21-25
# --- Robot definition ---
@st.cache_resource
def get_robot_chain():
    """Returns an IKPy Chain approximating a UR3 robot arm."""
    return Chain(name="UR3_arm", links=[
        OriginLink(),  # Base

        # Joint 1: Shoulder Pan
        URDFLink(
            name="shoulder_pan",
            origin_translation=[0, 0, 0.15185],  # meters
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1]
        ),

        # Joint 2: Shoulder Lift
        URDFLink(
            name="shoulder_lift",
            origin_translation=[0, 0.1197, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0]
        ),

        # Joint 3: Elbow
        URDFLink(
            name="elbow",
            origin_translation=[0.24365, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0]
        ),

        # Joint 4: Wrist 1
        URDFLink(
            name="wrist_1",
            origin_translation=[0.21325, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0]
        ),

        # Joint 5: Wrist 2
        URDFLink(
            name="wrist_2",
            origin_translation=[0, 0.08535, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1]
        ),

        # Joint 6: Wrist 3 / End Effector
        URDFLink(
            name="wrist_3",
            origin_translation=[0, 0, 0.0819],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0]
        )
    ])

# @st.cache_data
# def animation_plot(x, y, z, animation_sequence, robot_chain):
#     """Creates a 3D plot of the robot arm given joint positions."""

#     # --- Create 3D plot ---
#     fig = go.Figure()

#     # Robot arm
#     fig.add_trace(go.Scatter3d(
#         x=x, y=y, z=z,
#         mode='lines+markers',
#         line=dict(width=10, color='rgb(59, 130, 246)'),
#         marker=dict(size=8, color='rgb(29, 78, 216)'),
#         name='Robot Arm',
#         showlegend=True
#     ))

#     # End effector highlight
#     fig.add_trace(go.Scatter3d(
#         x=[x[-1]], y=[y[-1]], z=[z[-1]],
#         mode='markers',
#         marker=dict(size=15, color='rgb(255, 255, 255)', 
#                     line=dict(width=2, color='rgb(0, 0, 0)')),
#         name='End Effector',
#         showlegend=True
#     ))

#     # Base
#     fig.add_trace(go.Scatter3d(
#         x=[0], y=[0], z=[0],
#         mode='markers',
#         marker=dict(size=10, color='rgb(34, 197, 94)', symbol='diamond'),
#         name='Base',
#         showlegend=True
#     ))

#     # Trajectory trace (show path of end effector)
#     trajectory_x = []
#     trajectory_y = []
#     trajectory_z = []
#     for angles_frame in animation_sequence[:st.session_state.frame_index + 1]:
#         frame_mats = robot_chain.forward_kinematics([0.0] + angles_frame, full_kinematics=True)
#         end_pos = frame_mats[-1][:3, 3]
#         trajectory_x.append(end_pos[0])
#         trajectory_y.append(end_pos[1])
#         trajectory_z.append(end_pos[2])

#     if len(trajectory_x) > 1:
#         fig.add_trace(go.Scatter3d(
#             x=trajectory_x, y=trajectory_y, z=trajectory_z,
#             mode='markers',
#             marker=dict(size=3, color='rgb(239, 68, 68)'),
#             name='Trajectory',
#             showlegend=True,
#         ))

#     return fig

def animation_plot(x, y, z, trajectory, frame_index):
    fig = go.Figure()

    # Arm segment
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        line=dict(width=10, color='rgb(59, 130, 246)'),
        marker=dict(size=8, color='rgb(29, 78, 216)'),
        name='Robot Arm'
    ))

    # End effector
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode='markers',
        marker=dict(size=15, color='white', line=dict(width=2, color='black')),
        name='End Effector'
    ))

    # Base
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='rgb(34, 197, 94)', symbol='diamond'),
        name='Base'
    ))

    # Precomputed trajectory (just slice it)
    if frame_index > 0:
        fig.add_trace(go.Scatter3d(
            x=trajectory[:frame_index, 0],
            y=trajectory[:frame_index, 1],
            z=trajectory[:frame_index, 2],
            mode='markers',
            marker=dict(size=3, color='rgb(239, 68, 68)'),
            name='Trajectory'
        ))

    return fig
