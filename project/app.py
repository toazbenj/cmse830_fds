import streamlit as st
from library import *
from text import *
from animation import *

# Main
df_cobots, df_cobots_original, df_cycle_issue, df_gaps = load_cobotops_data()
df_aursad = load_aursad_data()
df_rad = load_rad_data()

st.set_page_config(page_title="Robostats", layout="wide")

# Sidebar page selector
page_lst = ["Intro", "Data Processing", "EDA", 'Inverse Kinematics', 'Animation', 'LSTM', 'Error Prediction']
page = st.sidebar.radio("Select Page", page_lst)
page_idx = page_lst.index(page)

df_name = st.sidebar.radio("Select Dataset", ["CobotOps", "AURSAD", 'RAD'])


if df_name == "CobotOps":
   df = df_cobots
   hover_data = ['grip_lost','Robot_ProtectiveStop']
   color_lst = ['grip_lost','Robot_ProtectiveStop', 'Temperature', 'Speed', 'Current']
   feature_lst = ['Temperature', 'Speed', 'Current']
   unit_lst = ["A", "m/s", "Degrees C"]
   st.session_state.frame_index  = 0

elif df_name == "AURSAD":
   df = df_aursad 

   # hover_data  = ["Damaged screw", "Extra assembly component", "Missing screw", "Damaged thread samples"]
   hover_data  = ["Damaged screw", "Extra assembly component", "Missing screw"]

   # hover_data  = []
   # small_lst = ["Damaged screw", "Extra assembly component", "Missing screw", "Damaged thread samples", 'Temperature', 'Speed', 'Current']
   color_lst = ["Damaged screw", "Extra assembly component", "Missing screw", 'time']
   feature_lst = ['Temperature', 'Speed', 'Current', 'q', 'target_q_', 'target_qd_']
   unit_lst = ["A", "m/s", "Degrees C", "rad", 'rad', "rad/s"]
   st.session_state.frame_index  = 0

else:
   df = df_rad
   hover_data = []
   color_lst = ['time', 'x', 'y', 'z']
   feature_lst = ['x', 'y', 'z']
   unit_lst = ['m', 'm', 'm']
   st.session_state.frame_index  = 0



if page_idx == 0:
   st.title("Robot Performance Analysis and Failure Prediction")
   st.header("By Ben Toaz", divider="gray")
   st.header("Introduction")

   col1, col2, col3= st.columns([1.75, 0.25, 1])
   with col1: 
      st.markdown(intro_text(), unsafe_allow_html=True)

   with col3:
      st.image("project/media/ur3.png", width='stretch')

   st.header("Example UR3 Operation")

   video_file = open("project/media/ur3.mp4", "rb")
   video_bytes = video_file.read()

   st.video(video_bytes)

elif page_idx == 1:
   st.title("Data Processing")
   st.markdown(data_collection_text(), unsafe_allow_html=True)

   st.header(f"{df_name} Raw Data Samples")

   st.dataframe(df.head(10), width='stretch')

   # Generated with Claude Sonnet 4.5 10-19-25
   st.header(f"{df_name} Dataset Overview")
   # Summary metrics
   col1, col2, col3, col4 = st.columns(4)
   col1.metric("Total Rows", f"{len(df):,}")
   col2.metric("Total Columns", len(df.columns))
   col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
   col4.metric("Missing Values", f"{df.isnull().sum().sum():,}")
   # Detailed info table
   st.write("**Column Details:**")
   info_df = pd.DataFrame({
      'Column': df.columns,
      'Type': df.dtypes.astype(str),
      'Non-Null': df.count().values,
      'Null': df.isnull().sum().values,
      'Null %': (df.isnull().sum() / len(df_cobots_original) * 100).round(2).astype(str) + '%'
   })
   st.dataframe(
      info_df,
      width='stretch',
      hide_index=True,
      column_config={
         "Column": st.column_config.TextColumn("Column Name", width="medium"),
         "Type": st.column_config.TextColumn("Data Type", width="small"),
         "Non-Null": st.column_config.NumberColumn("Non-Null Count", format="%d"),
         "Null": st.column_config.NumberColumn("Null Count", format="%d"),
         "Null %": st.column_config.TextColumn("Missing %", width="small")
      }
   )

   st.header("Time Data Encoding")
   st.markdown(cycle_time_text(), unsafe_allow_html=True)
   fig = px.scatter(
        df_cycle_issue,
        x='time',
        y='cycle',
        title=f'Cycle Time',
        opacity=0.7,
        hover_data=['cycle', 'time']
    )
   st.plotly_chart(fig, width='stretch')
   st.markdown(cycle_time_text2(), unsafe_allow_html=True)
   fig = px.scatter(
        df_cobots,
        x='time',
        y='cycle',
        title=f'Cycle Time',
        opacity=0.7,
        hover_data=['cycle', 'time']
    )
   st.plotly_chart(fig, width='stretch', key="cycle_time_fixed")

   st.header("Missingness")
   fig = missingness_heatmap(df_cobots_original)
   st.pyplot(fig, clear_figure=True)

   st.markdown("Missingness occurs infrequently in CobotOps, and when it does, it happens across most of the features. " \
   "This suggests that it is missing completely at random, which doesn't require too advanced techniques to fix. The AURSAD dataset doesn't have any missingness. Lucky.")

   st.header("Imputation of Time Series Data - Pre and Post Interpolation Example")
   st.markdown(imputation_text(), unsafe_allow_html=True)
   option = st.selectbox(
      "Select a feature to interpolate:",
      ("Current", "Speed", "Temperature"),
   )
   fig = interpolation_example(df_cobots, df_gaps, option)
   st.plotly_chart(
    fig,
    width='stretch',
    config={
        "displayModeBar": True,     # show toolbar
        "scrollZoom": True,         # enable zoom with scroll
        "editable": False,          # disable direct edits
        "displaylogo": False        # hide Plotly logo
    })

elif page_idx == 2:
   st.title("Exploratory Data Analysis (EDA)")
   st.markdown(eda_text(), unsafe_allow_html=True)

   option = st.selectbox(
      "Select a graphic:",
      ('Stats', 'Scatter Plot', "Histogram", "Time Series", "Correlation Heatmaps"),
   )

   if option == "Stats":
      st.header("Statistical Summary")
      summary = detailed_summary(df)
      st.dataframe(summary)


   if option == "Scatter Plot":
      
      st.header("Scatter Plots")
      st.subheader("3D Scatter Plot")
      numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

      col1, col2, col3 = st.columns(3)
      with col1:
         x_3d = st.selectbox("Select X-axis (3D):", numeric_cols, index=0, key='x3d')
      with col2:
         y_3d = st.selectbox("Select Y-axis (3D):", numeric_cols, index=1, key='y3d')
      with col3:
         z_3d = st.selectbox("Select Z-axis (3D):", numeric_cols, index=2, key='z3d')

      color_3d = st.selectbox("Color by (3D):", color_lst, key='color3d')

      df_melted = melt_features(df)

      fig = px.scatter_3d(df_melted, x=x_3d, y=y_3d, z=z_3d, color=color_3d,
                           title=f'3D Scatter Plot: {x_3d} vs {y_3d} vs {z_3d}',
                           opacity=0.7, 
                           hover_data=df[hover_data],
                             color_continuous_scale='Viridis')
      fig.update_traces(marker=dict(size=5))
      fig.update_layout(height=700)
      st.plotly_chart(fig, width='stretch')

   if option == "Histogram":
      st.header("Feature Distributions")
      fig = histogram_plots(df, df_name)

      st.plotly_chart(fig, width='stretch')

   if option == "Time Series":
      st.header("Time Series Data")
      
      feature = st.selectbox(
      "Select a feature:",
         feature_lst,
      )
      
      error = st.radio(
         "Select an error type:",
         hover_data,
         horizontal=True
      )
      fig_lst = time_series_plots(df, error, feature, df_name, feature_type_lst=feature_lst, unit_lst =unit_lst)

      print(fig_lst)

      for fig in fig_lst:
         st.plotly_chart(fig, width='stretch')

   if option == "Correlation Heatmaps":

      if df_name != "RAD":
         st.header("Correlations by Robot Joint")

         is_cobotops = "CobotOps" == df_name
         fig = joint_correlation_heatmaps(df, df_name)
         st.plotly_chart(fig, width='stretch')

      st.header("Cross-Feature Correlations")

      fig = feature_correlation_heatmaps(df, df_name, feature_lst)
 
      st.plotly_chart(fig, width='stretch')

elif page_idx == 3:
   st.title("Inverse Kinematic Feature Engineering")
   st.header("Intro to Robot Movement")
   st.markdown(kinematic_text(), unsafe_allow_html=True)

   st.image('project/media/robot_schematic.png')

   st.markdown(inverse_kinematic_text(), unsafe_allow_html=True)
   fig = time_series_inv_kin(df_rad)
   st.plotly_chart(fig, width='stretch')

elif page_idx == 4:

   # --- Page setup ---
   st.set_page_config(layout="wide")
   st.title("Robot Movement Animation")

   # --- Data prep ---
   df_angles = df[['q0', 'q1', 'q2', 'q3', 'q4', 'q5']]
   animation_sequence = df_angles.values.tolist()
   robot_chain = get_robot_chain()

   # --- Session state ---
   if 'frame_index' not in st.session_state:
      st.session_state.frame_index = 0
   if 'playing' not in st.session_state:
      st.session_state.playing = False

   # === MAIN LAYOUT ===
   # ==========================================================
   # LEFT COLUMN — Controls
   # ==========================================================
   left_col, right_col = st.columns([1, 3])
   with left_col:
      st.markdown("### Controls")
      st.markdown("---")
      c1, c2, c3 = st.columns(3)
      with c1:
         if st.button("⏮ Reset"):
               st.session_state.frame_index = 0
               st.session_state.playing = False
               st.rerun()
      with c2:
         if st.button("▶ Play" if not st.session_state.playing else "⏸ Pause"):
               st.session_state.playing = not st.session_state.playing
               st.rerun()
      with c3:
         if st.button("Step ⏭"):
               st.session_state.frame_index = (st.session_state.frame_index + 1) % len(animation_sequence)
               st.rerun()

      speed = st.slider("Speed (fps)", 1, 30, 30, key="speed")
      frame = st.slider("Frame", 0, len(animation_sequence)-1, st.session_state.frame_index, key="frame_slider")
      if frame != st.session_state.frame_index:
         st.session_state.frame_index = frame
         st.session_state.playing = False

      view_option = st.selectbox("Camera View", ["Isometric", "Front", "Side"], index=0)

   # ==========================================================
   # RIGHT COLUMN — 3D Plot + Readouts
   # ==========================================================
   with right_col:
      # --- Forward kinematics ---
      angles = animation_sequence[st.session_state.frame_index]
      frame_matrices = robot_chain.forward_kinematics([0.0] + angles, full_kinematics=True)
      x, y, z = zip(*[m[:3, 3] for m in frame_matrices])

      # --- Camera presets ---
      camera_presets = {
         "Isometric": dict(x=1.5, y=1.5, z=1.2),
         "Front": dict(x=0.0, y=2.5, z=0.5),
         "Side": dict(x=2.5, y=0.0, z=0.5),
         "Top": dict(x=0.0, y=0.0, z=3.0)
      }

      if "camera_eye" not in st.session_state:
         st.session_state.camera_eye = camera_presets["Isometric"]
      if "last_view_option" not in st.session_state:
         st.session_state.last_view_option = "Isometric"

      if view_option != st.session_state.last_view_option:
         st.session_state.camera_eye = camera_presets[view_option]
         st.session_state.last_view_option = view_option

      camera_eye = st.session_state.camera_eye

      # --- Build 3D plot ---
      trajectory = df[['x', 'y', 'z']].to_numpy() 

      fig = animation_plot(x, y, z, trajectory, st.session_state.frame_index )
      fig.update_layout(
         autosize=False,
         width=750,             # fixed width
         height=450,
         margin=dict(l=0, r=0, t=0, b=0),
         legend=dict(
            x=0.98, y=0.02, xanchor='left', yanchor='top',
         ),
         scene=dict(
            aspectmode='cube',
            xaxis=dict(title='X', range=[-0.5, 0.5]),
            yaxis=dict(title='Y', range=[-0.5, 0.5]),
            zaxis=dict(title='Z', range=[0.0, 0.7]),
            camera=dict(
                  eye=camera_eye,
                  center=dict(x=0, y=0, z=0.2),
                  up=dict(x=0, y=0, z=1)
            )
         ),
         uirevision='static_scene'  # <— prevents re-layout across reruns
      )

      st.plotly_chart(fig, use_container_width=False, key="animation_plot")


      # st.markdown("<div style='width:650px; margin:auto;'>", unsafe_allow_html=True)
      # st.plotly_chart(fig,  key="animation_plot")
      # st.markdown("</div>", unsafe_allow_html=True)

      # --- Readouts ---
      st.markdown("### Robot State")
      # cols = st.columns([0.05]*9)
      # cols[0].metric("X", f"{x[-1]:.3f}")
      # cols[1].metric("Y", f"{y[-1]:.3f}")
      # cols[2].metric("Z", f"{z[-1]:.3f}")
      # for i, (col, angle) in enumerate(zip(cols[3:], angles)):
      #    col.metric(f"θ{i}", f"{angle:.3f}")
      cols = st.columns([1, 1, 1] + [1] * len(angles))

      cols[0].markdown(f"<small><b>X:</b> {x[-1]:.3f}</small>", unsafe_allow_html=True)
      cols[1].markdown(f"<small><b>Y:</b> {y[-1]:.3f}</small>", unsafe_allow_html=True)
      cols[2].markdown(f"<small><b>Z:</b> {z[-1]:.3f}</small>", unsafe_allow_html=True)

      for i, (col, angle) in enumerate(zip(cols[3:], angles)):
         col.markdown(f"<small><b>θ{i}:</b> {angle:.3f}</small>", unsafe_allow_html=True)

   # ==========================================================
   # AUTO-PLAY LOOP
   # ==========================================================
   if st.session_state.playing:
      time.sleep(1.0 / speed)
      st.session_state.frame_index = (st.session_state.frame_index + 1) % len(animation_sequence)
      st.rerun()

elif page_idx == 5:
   pass

elif page_idx == 6:

   pass