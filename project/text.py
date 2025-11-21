def intro_text():
   return """
   <p>\n\n\nThe Universal Robot 3 is one of the most common lab robots for use in university research. 
   Experiments using the UR3 can be compared to significant bodies of work which also use the same type of robot, 
   boosting reproducibility and easy comparison across different types of results. \n
   In this project, I chose three major datasets who use the UR3 in their experiments but gather different types of data about the robot.
   Since all of the data was gathered using the same hardware, I plan to use each dataset to fill in the missing features of the other 
   datasets in order to gain a more complete picture of how well the robot performed and what caused it to fail 
   (both because of hardware faults and at the given research task).

   See the datasets here:

   [CobotOps](https://archive.ics.uci.edu/dataset/963/ur3+cobotops)

   [AURSAD](https://zenodo.org/records/4559556)

   [RAD](https://github.com/ubc-systopia/dsn-2022-rad-artifact/tree/main)
   
   Note for the midterm project I will only be working with the first two. Also, view this app in dark mode if you are a cool person. Use the settings drop down in the upper right.</p>
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
   
   Note that the AURSAD dataset is 6 GB total, with millions of entries. Streamlit cannot handle this volume, so I have downsampled significantly for the web app.
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

   Managing the down time gap also introduces a visible discontinuity in the temperature data
   because the a heat energy in the robot changes over time regardless of whether it is in operation or not. 
   We will leave this in for now, but will probably have to deal with it when we start building predictive models.</p>
   """

