import streamlit as st
from library import *

def intro_text():
   return """
   <p>\n\n\nThe Universal Robot 3 is one of the most common lab robots for use in university research. 
   Experiments using the UR3 can be compared to significant bodies of work which also use the same type of robot, 
   boosting reproducibility and easy comparison across different types of results. \n
   In this project, I chose three major datasets who use the UR3 in their experiments but gather different types of data about the robot.
   Since all of the data was gathered using the same hardware, I plan to use each dataset to fill in the missing features of the other 
   datasets in order to gather a more complete picture of how well the robot performed and what caused it to fail 
   (both because of hardware faults and at the given research task).

   See the datasets here:

   [CobotOps](https://archive.ics.uci.edu/dataset/963/ur3+cobotops)

   [AURSAD](https://zenodo.org/records/4559556)

   [RAD](https://github.com/ubc-systopia/dsn-2022-rad-artifact/tree/main)
   
   Note for the midterrm project I will only be working with the first two.</p>
   """

def data_collection_text():
   return """
   <p>I had to fill in some missing values using interpolation, but overall the data is high enough quality with few enough missing values to get a complete picture using this imputation method. 
   The timestamps for each sensor reading I had to pull apart into hours, minutes and seconds (all on the same day), 
   then condensed into a feature containing the total seconds from the start of the experiment. 
   The AURSAD dataset is 6 GB in its original form, so I had to do some compression as well.</p>
   """

def eda_text():
   return """
   <p>The data itself is almost entirely time series data of features like temperature, current, speed, and position of the various robot joints, as well as binary flags indicating various failure states. 
   CobotOps contains joint temperature, current and speed, while AURSAD also includes voltages, joint rotations, and robot link positions. 
   RAD only includes the position of the end effector tool, meaning I will have to rebuild these other features using inverse kinematics and models from the other datasets.
   
   Note that the AURSAD dataset is 6 GB total, with millions of entries. Streamlit cannot handle this volume, so I have downsampled down to about 0.5 GB for the web app.
   As a result, the AURSAD data is not as complete or representative as it would be if viewed locally.</p>
   """

def cycle_time_text():
   return """
   <p>Here we notice the cycle feature, or the robot's count of how many times it has iterated through a task, stops incrementing for a good period of time.
     While the cycle count stopped, the time feature kept counting, which masks a discontinuity in the experiment that effectively cuts the dataset in half. 
     If we rescale the second half of the time observations, we can eliminate this gap. Also, we should edit the starting value so that the experiment starts at time 0.</p>
   """

def cycle_time_text2():
   return """
   <p>Note that there are still discontinuities in the cycle counts themselves, but the timestamps are now continuous. 
   This could be becaused of revised counts in the robot's software after reassessing the completeness of a given task,
     or possibly lags in actually logging several backlogged cycle completions. There is no easy fix for this, so we will leave it for now.</p>
   """

def imputation_text():
   return """
   <p>Looking at the time series data confirms this. Small gaps occur between readings, indicating it is an occasional sensor error. 
   We can fix this easily with a nearest neighbor method or simply interpolating between the proceeding and following readings in time.
   Measurements are very stable so this won't affect the overall statistics by much.

   Managing the down time gap also introduces a visible discontinuity in the temperature data,
   because the a heat energy in the robot changes over time regardless of whether it is in operation or not. 
   We will leave this in for now, but will probably have to deal with it when we start building predictive models.</p>
   """


# Main
df_cobots, df_cobots_original, df_cycle_issue, df_gaps = cobots_data()
df_aursad = aursad_data()

# Sidebar page selector
page = st.sidebar.radio("Select Page", ["Intro", "Data Processing", "EDA"])
page_idx = ["Intro", "Data Processing", "EDA"].index(page)

df_name = st.sidebar.radio("Select Dataset", ["CobotOps", "AURSAD"])

if df_name == "CobotOps":
   df = df_cobots
   hover_data = ['grip_lost','Robot_ProtectiveStop']
   small_lst = ['grip_lost','Robot_ProtectiveStop', 'Temperature', 'Speed', 'Current']

else:
   df = df_aursad
   # hover_data  = ["Damaged screw", "Extra assembly component", "Missing screw", "Damaged thread samples"]
   hover_data  = ["Damaged screw", "Extra assembly component", "Missing screw"]

   # hover_data  = []
   # small_lst = ["Damaged screw", "Extra assembly component", "Missing screw", "Damaged thread samples", 'Temperature', 'Speed', 'Current']
   small_lst = ["Damaged screw", "Extra assembly component", "Missing screw", 'Temperature', 'Speed', 'Current']

st.set_page_config(page_title="Robot Joint Monitoring", layout="wide")

if page_idx == 0:
   st.title("Robot Performance Analysis and Failure Prediction")
   st.header("By Ben Toaz", divider="gray")
   st.header("Introduction")

   col1, col2, col3= st.columns([1.75, 0.25, 1])
   with col1: 
      st.markdown(intro_text(), unsafe_allow_html=True)

   with col3:
      st.image("project/media/ur3.png", use_container_width=True)

   st.header("Example UR3 Operation")

   video_file = open("project/media/ur3.mp4", "rb")
   video_bytes = video_file.read()

   st.video(video_bytes)

elif page_idx == 1:
   st.title("Data Processing")
   st.markdown(data_collection_text(), unsafe_allow_html=True)

   st.header(f"{df_name} Raw Data Samples")

   st.dataframe(df.head(10), use_container_width=True)

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
      use_container_width=True,
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
   st.plotly_chart(fig, use_container_width=True)
   st.markdown(cycle_time_text2(), unsafe_allow_html=True)
   fig = px.scatter(
        df_cobots,
        x='time',
        y='cycle',
        title=f'Cycle Time',
        opacity=0.7,
        hover_data=['cycle', 'time']
    )
   st.plotly_chart(fig, use_container_width=True, key="cycle_time_fixed")

   st.header("Missingness")
   fig = missingness_heatmap(df_cobots_original)
   st.pyplot(fig, clear_figure=True)

   st.markdown("Missingness occurs infrequently in CobotOps, and when it does, it happens across most of the features. " \
   "This suggests that it is MCAR, which doesn't require too advanced techniques to fix. The AURSAD dataset doesn't have any missingness. Lucky.")

   st.header("Imputation of Time Series Data - Pre and Post Interpolation Example")
   st.markdown(imputation_text(), unsafe_allow_html=True)
   option = st.selectbox(
      "Select a feature to interpolate:",
      ("Current", "Speed", "Temperature"),
   )
   fig = interpolation_example(df_cobots, df_gaps, option)
   st.plotly_chart(
    fig,
    use_container_width=True,
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
      ('Scatter Plot', "Histogram", "Time Series", "Correlation Heatmaps"),
   )

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

      color_3d = st.selectbox("Color by (3D):", small_lst, key='color3d')

      df_melted = melt_features(df)

      fig = px.scatter_3d(df_melted, x=x_3d, y=y_3d, z=z_3d, color=color_3d,
                           title=f'3D Scatter Plot: {x_3d} vs {y_3d} vs {z_3d}',
                           opacity=0.7, 
                           hover_data=df[hover_data],
                             color_continuous_scale='Viridis')
      fig.update_traces(marker=dict(size=5))
      fig.update_layout(height=700)
      st.plotly_chart(fig, use_container_width=True)

   if option == "Histogram":
      st.header("Joint Feature Distributions")
      fig = histogram_plots(df)
      st.plotly_chart(fig, use_container_width=True)

   if option == "Time Series":
      st.header("Time Series Data")
      
      feature = st.selectbox(
      "Select a feature:",
         ("Current", "Speed", "Temperature"),
      )
      
      error = st.radio(
         "Select an error type:",
         hover_data,
         horizontal=True
      )
      fig_lst = time_series_plots(df, error, feature)

      for fig in fig_lst:
         st.plotly_chart(fig, use_container_width=True)

   if option == "Correlation Heatmaps":
      st.header("Correlations by Robot Joint")

      is_cobotops = "CobotOps" == df_name
      fig = joint_correlation_heatmaps(df, is_cobotops)
      st.plotly_chart(fig, use_container_width=True)

      st.header("Cross-Feature Correlations")
      fig = feature_correlation_heatmaps(df)
      st.plotly_chart(fig, use_container_width=True)
