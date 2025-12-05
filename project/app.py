import streamlit as st
from library import *
from text import *
from animation import *


# Main
df_cobots, df_cobots_original, df_cycle_issue, df_gaps = load_cobotops_data()
df_aursad, df_pred_q, df_pred_all_good, df_pred_all_bad = load_aursad_data()
# df_aursad, df_pred_q = load_aursad_data()
df_rad = load_rad_data()

st.set_page_config(page_title="Robostats", layout="wide")

# Sidebar page selector
page_lst = ["Intro", "Data Processing", "EDA", 'Inverse Kinematics', 'Animation', 'Reconstruction', 'Error Prediction']

page = st.sidebar.radio("Select Page", page_lst)
page_idx = page_lst.index(page)

pages_with_dataset = ["EDA", "Animation", "Error Prediction"]
is_data_set_selectable = page in pages_with_dataset

if is_data_set_selectable:
   df_name = st.sidebar.radio("Select Dataset", ["CobotOps", "AURSAD", 'RAD'])
else:
   df_name = 'CobotOps'

if df_name == "CobotOps":
   df = df_cobots
   hover_data = ['grip_lost','Robot_ProtectiveStop']
   color_lst = ['grip_lost','Robot_ProtectiveStop', 'Temperature', 'Speed', 'Current']
   feature_lst = ['Temperature', 'Speed', 'Current']
   unit_lst = ["A", "m/s", "Degrees C"]

elif df_name == "AURSAD":
   df = df_aursad 

   # hover_data  = ["Damaged screw", "Extra assembly component", "Missing screw", "Damaged thread samples"]
   hover_data  = ["Damaged screw", "Extra assembly component", "Missing screw"]

   # hover_data  = []
   # small_lst = ["Damaged screw", "Extra assembly component", "Missing screw", "Damaged thread samples", 'Temperature', 'Speed', 'Current']
   color_lst = ["Damaged screw", "Extra assembly component", "Missing screw", 'time']
   feature_lst = ['Temperature', 'Speed', 'Current', 'q', 'target_q_', 'target_qd_']
   unit_lst = ["A", "m/s", "Degrees C", "rad", 'rad', "rad/s"]
else:
   df = df_rad
   hover_data = []
   color_lst = ['time', 'x', 'y', 'z']
   feature_lst = ['x', 'y', 'z']
   unit_lst = ['m', 'm', 'm']

# df transition issue with mismatched frames
if 'df_name_last' not in st.session_state:
   st.session_state.df_name_last = None

if df_name != st.session_state.df_name_last:
   st.session_state.frame_index = 0

if page_idx == 0:
   is_data_set_selectable = False

   st.title("Robot Performance Analysis and Failure Prediction")
   st.header("By Ben Toaz", divider="gray")
   st.header("Introduction")

   col1, col2, col3= st.columns([1.75, 0.25, 1])
   with col1: 
      st.markdown(intro_text(), unsafe_allow_html=True)

   with col3:
      st.image("project/media/ur3.png", use_container_width=False)

   st.header("Example UR3 Operation")

   video_file = open("project/media/ur3.mp4", "rb")
   video_bytes = video_file.read()

   st.video(video_bytes)

elif page_idx == 1:
   is_data_set_selectable = False

   st.title("Data Processing")
   st.markdown(data_collection_text(), unsafe_allow_html=True)

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
   st.plotly_chart(fig, use_container_width=False)
   st.markdown(cycle_time_text2(), unsafe_allow_html=True)
   fig = px.scatter(
        df_cobots,
        x='time',
        y='cycle',
        title=f'Cycle Time',
        opacity=0.7,
        hover_data=['cycle', 'time']
    )
   st.plotly_chart(fig, use_container_width=False, key="cycle_time_fixed")

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
    use_container_width=False,
    config={
        "displayModeBar": True,     
        "scrollZoom": True,         
        "editable": False,          
        "displaylogo": False        
    })

elif page_idx == 2:
   is_data_set_selectable = True

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
         x_3d = st.selectbox("Select X-axis:", numeric_cols, index=0, key='x3d')
      with col2:
         y_3d = st.selectbox("Select Y-axis:", numeric_cols, index=1, key='y3d')
      with col3:
         z_3d = st.selectbox("Select Z-axis:", numeric_cols, index=2, key='z3d')

      color_3d = st.selectbox("Color by:", color_lst, key='color3d')

      df_melted = melt_features(df)

      fig = px.scatter_3d(df_melted, x=x_3d, y=y_3d, z=z_3d, color=color_3d,
                           title=f'3D Scatter Plot: {x_3d} vs {y_3d} vs {z_3d}',
                           opacity=0.7, 
                           hover_data=df[hover_data],
                             color_continuous_scale='Viridis')
      fig.update_traces(marker=dict(size=5))
      fig.update_layout(height=700)
      st.plotly_chart(fig, use_container_width=False)

   if option == "Histogram":
      st.header("Feature Distributions")
      fig = histogram_plots(df, df_name)

      st.plotly_chart(fig, use_container_width=False)

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
         st.plotly_chart(fig, use_container_width=False)

   if option == "Correlation Heatmaps":

      if df_name != "RAD":
         st.header("Correlations by Robot Joint")

         is_cobotops = "CobotOps" == df_name
         fig = joint_correlation_heatmaps(df, df_name)
         st.plotly_chart(fig, use_container_width=False)

      st.header("Cross-Feature Correlations")

      fig = feature_correlation_heatmaps(df, df_name, feature_lst)
 
      st.plotly_chart(fig, use_container_width=False)

elif page_idx == 3:
   is_data_set_selectable = False

   st.title("Inverse Kinematic Feature Engineering")
   st.header("Intro to Robot Movement")
   st.markdown(kinematic_text(), unsafe_allow_html=True)

   st.image('project/media/robot_schematic.png')

   st.markdown(inverse_kinematic_text(), unsafe_allow_html=True)
   fig = time_series_inv_kin(df_rad)
   st.plotly_chart(fig, use_container_width=False)

elif page_idx == 4:
   is_data_set_selectable = True

   #  Page setup 
   st.set_page_config(layout="wide")
   st.title("Robot Movement Animation")

   #  Data prep 
   df_angles = df[['q0', 'q1', 'q2', 'q3', 'q4', 'q5']]
   animation_sequence = df_angles.values.tolist()
   robot_chain = get_robot_chain()

   #  Session state 
   if 'frame_index' not in st.session_state:
      st.session_state.frame_index = 0
   if 'playing' not in st.session_state:
      st.session_state.playing = False

   left_col, right_col = st.columns([1, 3])
   with left_col:
      st.markdown("### Controls")
      st.markdown("")

      # c1, c2, c3 = st.columns(3)
      c1, c3 = st.columns([1,1])

      with c1:
         if st.button("⏮ Reset", use_container_width=True):
               st.session_state.frame_index = 0
               st.session_state.playing = False
               st.rerun()
      # with c2:
         # if st.button("▶ Play" if not st.session_state.playing else "⏸ Pause", use_container_width=True):
         #       st.session_state.playing = not st.session_state.playing
         #       st.rerun()
         # pass
      with c3:
         if st.button("Step ⏭", use_container_width=True):
               st.session_state.frame_index = (st.session_state.frame_index + 1) % len(animation_sequence)
               st.rerun()

      # speed = st.slider("Speed (fps)", 1, 30, 30, key="speed")
      speed = 0
      frame = st.slider("Frame", 0, len(animation_sequence)-1, st.session_state.frame_index, key="frame_slider")
      if frame != st.session_state.frame_index:
         st.session_state.frame_index = frame
         st.session_state.playing = False

      view_option = st.selectbox("Camera View", ["Isometric", "Front", "Side"], index=0)

      st.markdown("")
      st.markdown("Step forward to move the robot through the dataset. " \
      "Move quickly through time second by second using the frame slider. Press reset to return to the first frame.")

   with right_col:
      # Forward kinematics
      angles = animation_sequence[st.session_state.frame_index]
      frame_matrices = robot_chain.forward_kinematics([0.0] + angles, full_kinematics=True)
      x, y, z = zip(*[m[:3, 3] for m in frame_matrices])

      # Camera presets
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

      # 3d plot 
      trajectory = df[['x', 'y', 'z']].to_numpy() 

      fig = animation_plot(x, y, z, trajectory, st.session_state.frame_index )
      fig.update_layout(
         autosize=False,
         width=750,        
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
         uirevision='static_scene' 
      )

      st.plotly_chart(fig, use_container_width=False, key="animation_plot")

      # Readouts 
      st.markdown("### Robot State")
      cols = st.columns([1, 1, 1] + [1] * len(angles))
      cols[0].markdown(f"<small><b>X:</b> {x[-1]:.3f}</small>", unsafe_allow_html=True)
      cols[1].markdown(f"<small><b>Y:</b> {y[-1]:.3f}</small>", unsafe_allow_html=True)
      cols[2].markdown(f"<small><b>Z:</b> {z[-1]:.3f}</small>", unsafe_allow_html=True)

      for i, (col, angle) in enumerate(zip(cols[3:], angles)):
         col.markdown(f"<small><b>θ{i}:</b> {angle:.3f}</small>", unsafe_allow_html=True)

   if st.session_state.playing:
      time.sleep(1.0 / speed)
      st.session_state.frame_index = (st.session_state.frame_index + 1) % len(animation_sequence)
      st.rerun()

elif page_idx == 5:
   is_data_set_selectable = False

   st.title("Long Short-Term Memory Feature Reconstruction")
   st.header("Recurrent Neural Networks")
   st.markdown(lstm_text(), unsafe_allow_html=True)

   st.image('project/media/lstm.png')

   st.header("Time Series Regression")

   st.markdown(q_training_text(), unsafe_allow_html=True)

   fig = time_series_prediction_plot(df_pred_q, [f'q{i}' for i in range(6)], 'rad', 'Actual vs. Predicted Joint Angles')
   st.plotly_chart(fig, use_container_width=False)

   st.markdown(sequence_training_text(), unsafe_allow_html=True)

   graph_features = ['Speed', 'Current', "Temperature"]
   units = ['rad/s', 'A', '°C']
   feature = st.radio("Select Failed Prediction Type", graph_features)
   unit = units[graph_features.index(feature)]

   fig = time_series_prediction_plot(df_pred_all_bad, [f'{feature}{i}' for i in range(6)], unit, 'Actual vs. Failed Prediction')
   st.plotly_chart(fig, use_container_width=False)

   feature = st.radio("Select Successful Prediction Type", graph_features)
   unit = units[graph_features.index(feature)]

   fig = time_series_prediction_plot(df_pred_all_good, [f'{feature}{i}' for i in range(6)], unit, 'Actual vs. Successful Prediction')
   st.plotly_chart(fig, use_container_width=False)

   st.header('Sample Training Curves')

   left_col, mid_col, right_col = st.columns([1, 1, 1])
   with left_col:
      st.subheader('Joint Angle Prediction (CobotOps)')
      st.image("project/media/lstm_q_pred1.png", use_container_width=False)
   with mid_col:
      st.subheader('All Features at Once (RAD)')
      st.image("project/media/bad_lstm.png", use_container_width=False)
   with right_col:
      st.subheader('Joint Temperature Prediciton (RAD)')
      st.image("project/media/lstm_temp_pred1.png", use_container_width=False)


elif page_idx == 6:
   is_data_set_selectable = True

   st.title("Error Prediction")
   st.header("Echo State Networks")
   st.markdown(esn_text(), unsafe_allow_html=True)

   st.image('project/media/esn.png')

   st.header("State Classification from Time Series Data")
   st.markdown(esn_training_text(), unsafe_allow_html=True)

   col1, col2 = st.columns(2)

   with col1:
      st.markdown(esn_training_stats())

   with col2:
      st.markdown(log_reg_training_stats())

   st.markdown(baseline_training_text(), unsafe_allow_html=True)


   model = load_esn_model('project/models/esn_fully_balanced.pt')

   st.subheader("Input Features")
   cols = st.columns(4)
   inputs = []

   if 'selected_index' not in st.session_state:
      st.session_state.selected_index = len(df)//2

   feature_type = st.selectbox(
      "Select feature to view (other features loaded to model automatically):",
      (df.columns),
   )

   st.session_state.selected_index = st.slider(
      "Select time index for prediction",
      min_value=0,
      max_value=len(df) - 1,
      value=len(df) // 2,
      step=1
   )

   color = px.colors.qualitative.Dark24[0]

   fig = go.Figure()
   fig.add_trace(go.Scatter(
      x=df["time"],
      y=df[feature_type],
      mode="lines",
      name=feature_type,
      line=dict(color=color, width=2)
   ))

   # Add vertical line at selected index
   fig.add_vline(
      x=st.session_state.selected_index,
      line_dash="dash",
      line_color="red",
      annotation_text=f"Selected: {st.session_state.selected_index}",
      annotation_position="top"
   )

   class_names = ['Damaged screw', 'Extra assembly component', 'Missing screw']
   class_colors = ['orange', 'purple', 'green', 'brown']

   for class_name, class_color in zip(class_names, class_colors):
      if class_name in df.columns:
         # Find rows where this class has value 1
         class_indices = df[df[class_name] == 1].index
         if len(class_indices) > 0:
               fig.add_trace(go.Scatter(
                  x=df.loc[class_indices, "time"],
                  y=df.loc[class_indices, feature_type],
                  mode="markers",
                  name=class_name,
                  marker=dict(color=class_color, size=8)
               ))

   fig.update_layout(
      xaxis_title="Time (s)",
      yaxis_title=feature_type,
      legend=dict(
         orientation="h", 
         yanchor="bottom", y=1.05, 
         xanchor="right", x=1
      ),
      height=500,
   )

   st.plotly_chart(fig, use_container_width=False)


   input_lst = ['q0','q1','q2','q3','q4','q5', 'x', 'y', 'z',
               'Current0','Current1','Current2','Current3','Current4','Current5',
               'Speed0','Speed1','Speed2','Speed3','Speed4','Speed5',
               'Temperature0','Temperature1','Temperature2','Temperature3','Temperature4','Temperature5']
   output_lst = ['Normal operation', 'Screw Loosening', 'Damaged screw', 'Extra assembly component', 'Missing screw']


   # Extract the feature vector at index
   df_selected = df.iloc[:st.session_state.selected_index]
   inputs = data_prep(df_selected, input_lst, output_lst, seq_len=50, target_ratio=1.0)

   # Run prediction
   with torch.no_grad():
      probs = np.array(model.predict(inputs))
      pred_class = np.argmax(probs[0]).item()
   st.subheader("Echo State Network Error Prediction (AURSAD Error Classes)")
   class_names = ['Normal operation', 'Damaged screw', 'Extra assembly component', 'Missing screw']

   # Create columns for probabilities
   cols = st.columns(len(class_names))
   for i, class_name in enumerate(class_names):
      with cols[i]:
         st.metric(label=class_name, value=f"{probs[0][i]*100:.2f}%")

   st.success(f"Predicted Class: **{class_names[pred_class]}**")

# For switching datasets
st.session_state.df_name_last = df_name
