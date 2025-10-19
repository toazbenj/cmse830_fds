import streamlit as st
from library import *


df_cobots, df_cobots_original = data_cleaning()

# Sidebar page selector
page = st.sidebar.radio("Select Page", ["Intro", "Data Processing", "EDA"])
page_idx = ["Intro", "Data Processing", "EDA"].index(page)

st.set_page_config(page_title="Robot Joint Monitoring", layout="wide")

def intro_text():
   return """
   <p>\n\n\nThe Universal Robot 3 is one of the most common lab robots for use in university research. 
   Experiments using the UR3 can be compared to significant bodies of work which also use the same type of robot, 
   boosting reproducibility and easy comparison across different types of results. \n
   In this project, I chose three major datasets who use the UR3 in their experiments but gather different types of data about the robot.
   Since all of the data was gathered using the same hardware, I plan to use each dataset to fill in the missing features of the other 
   datasets in order to gather a more complete picture of how well the robot performed and what caused it to fail 
   (both because of hardware faults and at the given research task). </p>
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
   RAD only includes the position of the end effector tool, meaning I will have to rebuild these other features using inverse kinematics and models from the other datasets.</p>
   """

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

   st.header("Raw Data Samples")
   # st.subheader("CobotOps Sample Data")
   st.dataframe(df_cobots.head(10), use_container_width=True)

   st.header("Time Data Encoding - Original, Intermediate, and Final Timestamp")

   df_times = df_cobots_original[['Timestamp']].copy()
   cols = ['hour', 'minute', 'second', 'time']
   df_times[cols] = df_cobots[cols]
   st.dataframe(df_times.head(10), use_container_width=True)


   st.header("Missingness")
   fig = missingness_heatmap(df_cobots_original)
   st.pyplot(fig, clear_figure=True)

   st.header("Imputation of Time Series Data - Pre and Post Interpolation Example")

   option = st.selectbox(
      "Select a feature to interpolate:",
      ("Current", "Speed", "Temperature"),
   )

   fig = interpolation_example(df_cobots, df_cobots_original, option)
   # st.plotly_chart(fig, clear_figure=True)
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
      ("Histogram", "Time Series", "Correlation Heatmaps"),
   )

   if option == "Histogram":
      st.header("Joint Feature Distributions")
      fig = histogram_plots(df_cobots)
      st.plotly_chart(fig, use_container_width=True)

      stolen_plots(df_cobots)

   if option == "Time Series":
      st.header("Time Series Data")
      fig_lst = time_series_plots(df_cobots)
      for fig in fig_lst:
         st.plotly_chart(fig, use_container_width=True)

      st.header("Failure Events")
      fig = failure_events_heatmap(df_cobots)
      st.pyplot(fig, clear_figure=True)

   if option == "Correlation Heatmaps":
      st.header("Correlations by Robot Joint")
      fig = joint_correlation_heatmaps(df_cobots)
      st.plotly_chart(fig, use_container_width=True)

      st.header("Cross-Feature Correlations")
      fig = feature_correlation_heatmaps(df_cobots)
      st.plotly_chart(fig, use_container_width=True)
