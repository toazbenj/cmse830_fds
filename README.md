# UR3 Robot Performance Analysis and Failure Prediction

## Summary

### Why UR3?

The Universal Robot 3 is one of the most common lab robots for use in university research. 
Experiments using the UR3 can be compared to significant bodies of work which also use the same type of robot, boosting reproducibility and easy comparison across different types of results.
In this project, I chose three major datasets who use the UR3 in their experiments but gather different types of data about the robot. 
Since all of the data was gathered using the same hardware, I plan to use each dataset to fill in the missing features of the other datasets in order to gather a more complete picture of how well the robot performed and what caused it to fail (both because of hardware faults and at the given research task).

### IDA and EDA Insights

The data itself is almost entirely time series data of features like temperature, current, speed, and position of the various robot joints, as well as binary flags indicating various failure states. 
CobotOps contains joint temperature, current and speed, while AURSAD also includes voltages, joint rotations, and robot link positions. 
RAD only includes the position of the end effector tool, meaning I will have to rebuild these other features using inverse kinematics and models from the other datasets. 

### Data Processing

I had to fill in some missing values using interpolation, but overall the data is high enough quality with few enough missing values to get a complete picture using this imputation method.
The timestamps for each sensor reading I had to pull apart into hours, minutes and seconds (all on the same day), then condensed into a feature containing the total seconds from the start of the experiment. 
The AURSAD dataset is 6 GB in its original form, so I had to do some compression as well.

### Feature Engineering and Reconstruction

The RAD and CobotOps datasets were not nearly as complete as AURSAD. CobotOps completely omits position data from the robot, and RAD only has position data with no additional physical measurements.
In order to fix this, I used inverse kinematics to calculate the joint angles from the end effector position for the RAD dataset. Then I trained several LSTM neural networks to rebuild an approximation of current, speed, and temperature features given the joint positions as the inputs. For CobotOps, I used an LSTM to estimate the joint angles themselves and calculated the end effector position using forward kinematics. 

The LSTMs I used were implemented in PyTorch with a width of 128 neurons and two stacked LSTM cells. The model uses 50 timesteps worth of data across 6-22 different input features (depending on what was being forecasted). The output for all LSTMs was 6 floats, physical quantities pertaining to each of the 6 robot joints.

### Error Prediction Models

I trained an echo state network to take in the time series data and predict the failure states of the robot as seen in the AURSAD dataset, which was gathered while using the robot for screwdriving tasks. The model takes in 50 timesteps worth of data across 27 different input features (speed, current, temperature, angular position for 6 joints, then x, y, z position of the end effector). The network had a single large reservoir of 300 neurons, a spectral radius of 0.9, a leak rate of 1.0, and a ridge regression parameter of 1e-6. It was able to identify failures in the 4 operating state classes with a recall of about 60%, which was not as effective as expected. It still outperformed the logistic regression model in this metric, which could not identify the failure states at all.

## Datasets

[CobotOps](https://archive.ics.uci.edu/dataset/963/ur3+cobotops)

[AURSAD](https://zenodo.org/records/4559556)

[RAD](https://github.com/ubc-systopia/dsn-2022-rad-artifact/tree/main)

## Data Dictionaries (Original Data)

### CobotOps

| Type                         | Label                 | Samples |
|------------------------------|-----------------------|---------|
| Current Amperage Joints 1-6  | Current_J{0-5}        | 7409    |
| Angular Speed Joints 1-6     | Speed_J{0-5}          | 7409    |
| Temperature (C) Joints 1-6   | Temperature_J{0-5}    | 7409    |
| Robot Grip Failure           | grip_lost             | 7409    |
| Emergency Stop               | Robot_ProtectiveStop  | 7409    |


### AURSAD (Only Target Features Due to Size)

| Type                     | Label | Samples | %     |
|--------------------------|-------|---------|-------|
| Normal operation         | 0     | 1420    | 69.44 |
| Damaged screw            | 1     | 221     | 10.81 |
| Extra assembly component | 2     | 183     | 8.95  |
| Missing screw            | 3     | 218     | 10.65 |
| Damaged thread samples   | 4     | 3       | 0.15  |

### RAD (After Robot Data Extraction)

| Type                     | Label | Samples |
|--------------------------|-------|---------|
| End Effector X Position  | x     | 11103   |
| End Effector Y Position  | y     | 11103   |
| End Effector Z Position  | z     | 11103   |

## Streamlit App

My app contains visualizations of the time series data and correlation matrices for the robot joint features. You can view it here: [Robostats](https://robotstats.streamlit.app/). 

### Installation

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
   streamlit run project/app.py
   ```


