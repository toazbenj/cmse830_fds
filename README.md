# UR3 Robot Performance Analysis and Failure Prediction

## Datasets

[CobotOps](https://archive.ics.uci.edu/dataset/963/ur3+cobotops)

[AURSAD](https://zenodo.org/records/4559556)

[RAD](https://github.com/ubc-systopia/dsn-2022-rad-artifact/tree/main)

## Progress Summary - Midterm

### Why UR3?

The Universal Robot 3 is one of the most common lab robots for use in university research. 
Experiments using the UR3 can be compared to significant bodies of work which also use the same type of robot, boosting reproducibility and easy comparison across different types of results.
In this project, I chose three major datasets who use the UR3 in their experiments but gather different types of data about the robot. 
Since all of the data was gathered using the same hardware, I plan to use each dataset to fill in the missing features of the other datasets in order to gather a more complete picture of how well the robot performed and what caused it to fail (both because of hardware faults and at the given research task).

### IDA and EDA Insights

The data itself is almost entirely time series data of features like temperature, current, speed, and position of the various robot joints, as well as binary flags indicating various failure states. 
CobotOps contains joint temperature, current and speed, while AURSAD also includes voltages, joint rotations, and robot link positions. 
RAD only includes the position of the end effector tool, meaning I will have to rebuild these other features using inverse kinematics and models from the other datasets. 

### Current Data Processing

I had to fill in some missing values using interpolation, but overall the data is high enough quality with few enough missing values to get a complete picture using this imputation method.
The timestamps for each sensor reading I had to pull apart into hours, minutes and seconds (all on the same day), then condensed into a feature containing the total seconds from the start of the experiment. 
The AURSAD dataset is 6 GB in its original form, so I had to do some compression as well.

### Streamlit

My app contains visualizations of the time series data and corrlation matrices for the robot joint features. You can view it here: [Robostats](https://robotstats.streamlit.app/). 

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/toazbenj/cmse830_fds.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd ~/cmse_fds
   ```

3. **Install Required Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
   
4. **Run the app locally**:

   ```bash
   cd ~/cmse_fds
   streamlit run project/streamlit.py
   ```
