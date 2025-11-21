import streamlit as st
import numpy as np
import plotly.graph_objects as go
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import time

st.set_page_config(layout="wide")
st.title("🤖 Robot Animation – Joint Angle Sequence")

# --- Robot definition ---
@st.cache_resource
def get_robot_chain():
    return Chain(name="arm", links=[
        OriginLink(),
        URDFLink(
            name="link1",
            origin_translation=[0, 0, 0.1],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1]
        ),
        URDFLink(
            name="link2",
            origin_translation=[0.2, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0]
        ),
        URDFLink(
            name="link3",
            origin_translation=[0.2, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0]
        ),
    ])

robot_chain = get_robot_chain()

# --- Define animation sequence ---
# Each row is [origin, joint1, joint2, joint3]
# You can replace this with your own list of joint angles
animation_sequence = [
    # Start at home
    [0, 0, 0, 0],
    
    # Rotate base slowly (interpolated)
    [0, 0.1, 0, 0],
    [0, 0.2, 0, 0],
    [0, 0.3, 0, 0],
    [0, 0.4, 0, 0],
    [0, 0.5, 0, 0],
    [0, 0.6, 0, 0],
    
    # Lift shoulder (keeping Z positive)
    [0, 0.6, -0.1, 0],
    [0, 0.6, -0.2, 0],
    [0, 0.6, -0.3, 0],
    [0, 0.6, -0.4, 0],
    [0, 0.6, -0.5, 0],
    
    # Bend elbow
    [0, 0.6, -0.5, 0.1],
    [0, 0.6, -0.5, 0.2],
    [0, 0.6, -0.5, 0.3],
    [0, 0.6, -0.5, 0.4],
    [0, 0.6, -0.5, 0.5],
    
    # Lift more
    [0, 0.6, -0.6, 0.6],
    [0, 0.6, -0.7, 0.7],
    [0, 0.6, -0.8, 0.8],
    [0, 0.6, -0.9, 0.9],
    [0, 0.6, -1.0, 1.0],
    
    # Hold and rotate base
    [0, 0.55, -1.0, 1.0],
    [0, 0.5, -1.0, 1.0],
    [0, 0.45, -0.95, 0.95],
    [0, 0.4, -0.9, 0.9],
    [0, 0.35, -0.85, 0.85],
    [0, 0.3, -0.8, 0.8],
    
    # Lower smoothly
    [0, 0.25, -0.7, 0.7],
    [0, 0.2, -0.6, 0.6],
    [0, 0.15, -0.5, 0.5],
    [0, 0.1, -0.4, 0.4],
    [0, 0.05, -0.3, 0.3],
    [0, 0, -0.2, 0.2],
    [0, 0, -0.1, 0.1],
    [0, 0, 0, 0],
    
    # Rotate other direction
    [0, -0.1, 0, 0],
    [0, -0.2, 0, 0],
    [0, -0.3, 0, 0],
    [0, -0.4, 0, 0],
    [0, -0.5, 0, 0],
    [0, -0.6, 0, 0],
    
    # Lift on other side (keeping Z positive)
    [0, -0.6, -0.1, 0],
    [0, -0.6, -0.2, 0],
    [0, -0.6, -0.3, 0],
    [0, -0.6, -0.4, 0],
    [0, -0.6, -0.5, 0],
    
    # Bend elbow
    [0, -0.6, -0.5, 0.1],
    [0, -0.6, -0.5, 0.2],
    [0, -0.6, -0.5, 0.3],
    [0, -0.6, -0.5, 0.4],
    [0, -0.6, -0.5, 0.5],
    
    # Return from other side
    [0, -0.5, -0.5, 0.5],
    [0, -0.4, -0.4, 0.4],
    [0, -0.3, -0.3, 0.3],
    [0, -0.2, -0.2, 0.2],
    [0, -0.1, -0.1, 0.1],
    [0, 0, 0, 0],
    
    # Wave pattern
    [0, 0.2, -0.3, 0.3],
    [0, 0.4, -0.5, 0.5],
    [0, 0.5, -0.6, 0.6],
    [0, 0.4, -0.5, 0.5],
    [0, 0.2, -0.3, 0.3],
    [0, 0, -0.1, 0.1],
    [0, -0.2, -0.3, 0.3],
    [0, -0.4, -0.5, 0.5],
    [0, -0.5, -0.6, 0.6],
    [0, -0.4, -0.5, 0.5],
    [0, -0.2, -0.3, 0.3],
    [0, 0, 0, 0],
    
    # Circle motion
    [0, 0.3, -0.4, 0.4],
    [0, 0.5, -0.6, 0.6],
    [0, 0.6, -0.7, 0.7],
    [0, 0.5, -0.6, 0.6],
    [0, 0.3, -0.4, 0.4],
    [0, 0, -0.2, 0.2],
    [0, -0.3, -0.4, 0.4],
    [0, -0.5, -0.6, 0.6],
    [0, -0.6, -0.7, 0.7],
    [0, -0.5, -0.6, 0.6],
    [0, -0.3, -0.4, 0.4],
    [0, 0, 0, 0],
]

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
frame_matrices = robot_chain.forward_kinematics(angles, full_kinematics=True)

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
    frame_mats = robot_chain.forward_kinematics(angles_frame, full_kinematics=True)
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

fig.update_layout(
    scene=dict(
        aspectmode='cube',
        xaxis=dict(title='X', range=[-0.5, 0.5]),
        yaxis=dict(title='Y', range=[-0.5, 0.5]),
        zaxis=dict(title='Z', range=[0, 0.7]),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2),
            center=dict(x=0, y=0, z=0.2),
            up=dict(x=0, y=0, z=1)
        )
    ),
    height=600,
    margin=dict(l=0, r=0, t=0, b=0),
    uirevision='constant'  # Preserve camera
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
                if len(values) == 4:
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