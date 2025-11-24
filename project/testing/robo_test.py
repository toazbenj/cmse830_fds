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
@st.cache_data
def rebase_time_gaps(df, time_col='time', gap_threshold=2.0, reset_gap=0.01):
    """
    Rebases timestamps when gaps exceed a threshold.
    
    Parameters:
        df : pd.DataFrame
            Must contain a numeric or datetime time column.
        time_col : str
            Name of the time column.
        gap_threshold : float
            Threshold (in seconds) for gap detection.
        reset_gap : float
            Gap to insert after rebasing (in seconds).
    Returns:
        pd.DataFrame with continuous time column.
    """
    df = df.copy()
    
    # Convert to numeric seconds if datetime
    if not pd.api.types.is_numeric_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
        df[time_col] = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds()
    
    # Sort by time
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # Detect large gaps
    time_vals = df[time_col].to_numpy()
    gaps = np.diff(time_vals)
    
    # Keep track of total offset applied
    offset = 0.0
    adjusted_times = [time_vals[0]]
    
    for i, gap in enumerate(gaps, start=1):
        if gap > gap_threshold:
            # Increase offset by the size of the gap minus desired reset gap
            offset += (gap - reset_gap)
        adjusted_times.append(time_vals[i] - offset)
    
    df[time_col] = adjusted_times
    return df

@st.cache_data
def aursad_data():
    data_path = "project/data/aursad"

    # Get all feather files, sorted in order (important for time series)
    feather_files = sorted(glob.glob(os.path.join(data_path, "part_*.feather")))

    # Load and concatenate
    start = 2
    stop = 6
    df_aursad = pd.concat([pd.read_feather(f) for f in feather_files[start:stop]], ignore_index=True)

    df_aursad = df_aursad.rename(columns={'timestamp': 'time'})
    df_aursad['time'] = df_aursad['time'] - df_aursad['time'].min()
    df_aursad = df_aursad.sort_values('time').reset_index(drop=True)
    df_aursad = rebase_time_gaps(df_aursad, time_col='time', gap_threshold=2.0, reset_gap=0.01)

    # Downsample
    df_aursad = df_aursad.iloc[::100]

    # Renaming to match CobotOps
    for i in range(6):
        df_aursad = df_aursad.rename(columns={f'actual_current_{i}': f'Current_J{i}'})
        df_aursad = df_aursad.rename(columns={f'actual_TCP_speed_{i}': f'Speed_J{i}'})
        df_aursad = df_aursad.rename(columns={f'joint_temperatures_{i}': f'Temperature_J{i}'})

    # Encode labels for screwing failures
    df_aursad = pd.get_dummies(df_aursad, columns=['label'], prefix='label')
    label_names = ["Normal operation", "Damaged screw", "Extra assembly component", "Missing screw", "Damaged thread samples", "Screw Loosening"]

    for i, label in enumerate(label_names):
        df_aursad = df_aursad.rename(columns={f'label_{i}': label})
    df_aursad.head()

    return df_aursad


df = aursad_data()
df_angles = df[['actual_q_0', 'actual_q_1', 'actual_q_2', 'actual_q_3', 'actual_q_4', 'actual_q_5']]

st.set_page_config(layout="wide")
st.title("🤖 Robot Animation – Joint Angle Sequence")

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

# Instantiate chain
robot_chain = get_robot_chain()


robot_chain = get_robot_chain()

# --- Define animation sequence ---
animation_sequence = df_angles.values.tolist()


# --- Session state for animation ---
if 'frame_index' not in st.session_state:
    st.session_state.frame_index = 0
if 'playing' not in st.session_state:
    st.session_state.playing = False

# --- Controls ---
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 2, 2])

with col1:
    if st.button("▶️ Play" if not st.session_state.playing else "⏸️ Pause"):
        st.session_state.playing = not st.session_state.playing
        st.rerun()

with col2:
    if st.button("⏮️ Reset"):
        st.session_state.frame_index = 0
        st.session_state.playing = False
        st.rerun()

with col3:
    if st.button("⏭️ Step"):
        st.session_state.frame_index = (st.session_state.frame_index + 1) % len(animation_sequence)
        st.rerun()

with col4:
    speed = st.slider("Speed (fps)", 1, 30, 10, key="speed")

with col5:
    frame = st.slider("Frame", 0, len(animation_sequence)-1, st.session_state.frame_index, key="frame_slider")
    if frame != st.session_state.frame_index:
        st.session_state.frame_index = frame
        st.session_state.playing = False

# --- Get current joint angles ---
angles = animation_sequence[st.session_state.frame_index]

# --- Forward kinematics ---
print("Total links:", len(robot_chain.links))
for i, link in enumerate(robot_chain.links):
    print(i, link.name)

print("Expected joint vector length:", len(robot_chain.links) - 1)
print("Your joint vector length:", len(angles))

frame_matrices = robot_chain.forward_kinematics([0.0] + angles, full_kinematics=True)

# Extract positions
points = []
for matrix in frame_matrices:
    points.append(matrix[:3, 3])

x, y, z = zip(*points)

# --- Create 3D plot ---
fig = go.Figure()

# Robot arm
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode='lines+markers',
    line=dict(width=10, color='rgb(59, 130, 246)'),
    marker=dict(size=8, color='rgb(29, 78, 216)'),
    name='Robot Arm',
    showlegend=True
))

# End effector highlight
fig.add_trace(go.Scatter3d(
    x=[x[-1]], y=[y[-1]], z=[z[-1]],
    mode='markers',
    marker=dict(size=12, color='rgb(239, 68, 68)', 
                line=dict(width=2, color='rgb(153, 27, 27)')),
    name='End Effector',
    showlegend=True
))

# Base
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers',
    marker=dict(size=10, color='rgb(34, 197, 94)', symbol='diamond'),
    name='Base',
    showlegend=True
))

# Trajectory trace (show path of end effector)
trajectory_x = []
trajectory_y = []
trajectory_z = []
for angles_frame in animation_sequence[:st.session_state.frame_index + 1]:
    frame_mats = robot_chain.forward_kinematics([0.0] + angles_frame, full_kinematics=True)
    end_pos = frame_mats[-1][:3, 3]
    trajectory_x.append(end_pos[0])
    trajectory_y.append(end_pos[1])
    trajectory_z.append(end_pos[2])

if len(trajectory_x) > 1:
    fig.add_trace(go.Scatter3d(
        x=trajectory_x, y=trajectory_y, z=trajectory_z,
        mode='lines',
        line=dict(width=3, color='rgba(239, 68, 68, 0.5)', dash='dash'),
        name='Trajectory',
        showlegend=True
    ))

# --- View selection ---
view_option = st.selectbox(
    "🔭 Select View",
    # ["Isometric", "Front", "Side", "Top"],
    ["Isometric", "Front", "Side"],
    index=0,
    key="view_option"
)

# --- Camera presets ---
camera_presets = {
    "Isometric": dict(x=1.5, y=1.5, z=1.2),
    "Front": dict(x=0.0, y=2.5, z=0.5),
    "Side": dict(x=2.5, y=0.0, z=0.5),
    "Top": dict(x=0.0, y=0.0, z=3.0)
}

# --- Initialize persistent state ---
if "camera_eye" not in st.session_state:
    st.session_state.camera_eye = camera_presets["Isometric"]
if "last_view_option" not in st.session_state:
    st.session_state.last_view_option = "Isometric"

# --- Only update when user changes dropdown ---
if view_option != st.session_state.last_view_option:
    st.session_state.camera_eye = camera_presets[view_option]
    st.session_state.last_view_option = view_option

camera_eye = st.session_state.camera_eye

# --- Layout update ---
fig.update_layout(
    scene=dict(
        aspectmode='cube',
        xaxis=dict(title='X', range=[-0.5, 0.5]),
        yaxis=dict(title='Y', range=[-0.5, 0.5]),
        zaxis=dict(title='Z', range=[0, 0.7]),
        camera=dict(
            eye=camera_eye,
            center=dict(x=0, y=0, z=0.2),
            up=dict(x=0, y=0, z=1)
        )
    ),
    height=600,
    margin=dict(l=0, r=0, t=0, b=0),
    uirevision='locked_camera'
)



st.plotly_chart(fig, use_container_width=True)

# --- Display current state ---
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Frame {st.session_state.frame_index + 1} / {len(animation_sequence)}")
    
    # Display joint angles
    st.write("**Joint Angles (radians):**")
    cols = st.columns(len(angles))
    for i, (col, angle) in enumerate(zip(cols, angles)):
        col.metric(f"θ{i}", f"{angle:.3f}")

with col2:
    st.subheader("End Effector Position")
    st.write(f"**X:** {x[-1]:.3f}")
    st.write(f"**Y:** {y[-1]:.3f}")
    st.write(f"**Z:** {z[-1]:.3f}")

# --- Input your own sequence ---
with st.expander("📝 Edit Animation Sequence"):
    st.write("Each line should have 4 values: [origin, θ1, θ2, θ3]")
    sequence_text = st.text_area(
        "Joint Angles (one configuration per line)",
        value="\n".join([", ".join(map(str, angles)) for angles in animation_sequence]),
        height=200
    )
    
    if st.button("Update Sequence"):
        try:
            new_sequence = []
            for line in sequence_text.strip().split("\n"):
                values = [float(x.strip()) for x in line.split(",")]
                if len(values) == 6:
                    new_sequence.append(values)
            
            if new_sequence:
                animation_sequence = new_sequence
                st.session_state.frame_index = 0
                st.success(f"✅ Updated sequence with {len(new_sequence)} frames")
                st.rerun()
            else:
                st.error("No valid frames found")
        except Exception as e:
            st.error(f"Error parsing sequence: {e}")

# --- Auto-play animation ---
if st.session_state.playing:
    time.sleep(1.0 / speed)
    st.session_state.frame_index = (st.session_state.frame_index + 1) % len(animation_sequence)
    st.rerun()

# --- Instructions ---
st.markdown("---")
st.markdown("""
### 🎮 Controls:
- **▶️ Play/⏸️ Pause**: Start/stop animation
- **⏮️ Reset**: Go back to first frame
- **⏭️ Step**: Advance one frame
- **Speed slider**: Control animation speed
- **Frame slider**: Jump to specific frame
- **Trajectory**: Red dashed line shows the path traveled by the end effector

### 📊 Features:
- Camera position is preserved during animation
- Shows trajectory of end effector
- Editable joint angle sequence
- Real-time forward kinematics
""")