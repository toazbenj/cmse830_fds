def intro_text():
   return """
   <p>\n\n\nThe Universal Robot 3 is one of the most common lab robots for use in university research.
   Experiments using the UR3 can be compared to significant bodies of work which also use the same type of robot,
   boosting reproducibility and easy comparison across different types of results. \n
   In this project, I chose three major datasets who use the UR3 in their experiments but gather different types of data about the robot.
   Since all of the data was gathered using the same hardware, I used each dataset to fill in the missing features of the other
   datasets with a Long Short-Term Memory neural network. This helps us gain a more complete picture of how well the robot performed and what caused it to fail
   (both because of hardware faults and at the given research task).
   I then implemented an Echo State Network (ESN) for the task of classifying failures based on the time series data.

   See the datasets here:

   [CobotOps](https://archive.ics.uci.edu/dataset/963/ur3+cobotops)

   [Universal Robot Screwdriving Anomaly Detection Dataset (AURSAD)](https://zenodo.org/records/4559556)

   [Robotic Arm Dataset (RAD)](https://github.com/ubc-systopia/dsn-2022-rad-artifact/tree/main)
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
   This could be because of revised counts in the robot's software after reassessing the completeness of a given task,
     or possibly lags in actually logging several backlogged cycle completions. There is no easy fix for this, so we will leave it for now.</p>
   """


def imputation_text():
   return """
   <p>Looking at the time series data confirms this. Small gaps occur between readings, indicating it is an occasional sensor error.
   We can fix this easily with a nearest neighbor method or simply interpolating between the proceeding and following readings in time.
   Measurements are very stable so this won't affect the overall statistics by much.


   Managing the down time gap also introduces a visible discontinuity in the temperature data
   because the heat energy in the robot changes over time regardless of whether it is in operation or not.
   We will leave this in for now, but will probably have to deal with it when we start building predictive models.</p>
   """


def kinematic_text():
   return """
   <p>Kinematics is the mathematics behind how a robot moves through space.
   Given some inputs, such as the rotations of the robot joints, the end effector, or the tip of the robot, will end up in a given xyz position.
   If you gave the robot the correct rotations, this position will be in the spot that you want.</p>
   """


def inverse_kinematic_text():
   return """
   <p>We can also do this in reverse.
   Given the robot end point is in a position in space, we can calculate the most likely rotations of the joints and determine the configuration of the robot.
   The RAD dataset only had the xyz position of the robot's end effector, but I wanted to know what was happening with each of the robot's joints.
   To find the rotations, I used a library called Ikpy, which allows you to define your robot limbs and dimensions.
   Ikpy also has a built-in optimizer to do the inverse kinematics for you.
   I used this to reconstruct the missing joint angles in the dataset.
   
   After finding the most likely rotations, if we perform the forward kinematics on these, we receive the original positions with only a slight error introduced.
   With this done, we can visualize the positions and movements of the entire robot on the Animation page</p>
   """


def lstm_text():
   return """
   <p>I used three different large-scale datasets in this project. Out of all of these, the most substantial is the AURSAD, which has 6 GB of detailed robot data with no missingness.
   The Cobotops and RAD datasets are much smaller and have limited features. The Cobotops dataset for instance has speed, current, and temperature data, but crucially omits all position data.
   The RAD dataset contains logs from many devices, which is not useful for our application, and only includes robot position data.
   Here I leveraged the sheer size of the AURSAD data and the fact that all the logs come from the same type of robot in order to rebuild the missing features of position,
   speed, current, and temperature for these smaller datasets.
   
   Given the large amount of information that needs to be parsed, we chose to use a Long Short-Term Memory (LSTM) neural network.
   This architecture is useful for time series forecasting, or in this case reconstruction.
   The gated structure of the cells within the LSTM allow for training sets of weights that learn what to remember about how previous states affect the current output,
   as well as what to states to forget when they no longer have a quantifiable effect on the prediction.</p>
   """


def q_training_text():
      return """
   <p>The first step was to reconstruct the position data using the speed, current, and temperature data of the Cobotops dataset.
   Many of the physical relationships from these inputs apply directly to the angular positions of the robot joints over time, which made this task the easiest.</p>
   """


def sequence_training_text():
   return """
   <p>Next I attempted to reconstruct the speed, current, and temperature of each robot joint in the RAD dataset using a single LSTM.
   Since the input was only the position of the end effector, there was not enough information to generalize to these other physical quantities, and the model was not successful.
   To combat this, I trained LSTMs for each of these features separately one after the other. I started with current, then speed, then temperature.
   After each successful reconstruction, I added the new feature data to the RAD dataset in order to become an additional input for the next model.
   This approach was much more successful. The result was that there were two much more complete datasets to use for error prediction.</p>
   """


def esn_text():
   return """
   <p>An echo state network (ESN) is an architecture that takes advantage of a large number of random weights in a network in order to create spontaneous and useful feature engineering.
   Only the input and output weights are trained, combining the random eddies of signals in the reservoir of neurons in useful ways.
   The weights in the reservoir are not trained, allowing for fast model creation and scalability.</p>
   """


def esn_training_text():
   return """
   <p>Here I used the ESN as a classification model in order to identify the time instants where the robot is most likely to fail at its given task.
   The inputs are end effector position, joint angle configuration, and speeds, temperatures, currents for each joint.
   The outputs are probabilities that the robot operates normally or falls into any of the failure categories.

   For the AURSAD, these categories are various ways you can mess up screwing in a screw.
   For Cobotops, they include dropping the item that was being gripped or throwing the emergency stop, possibly before an imminent collision with another object.
   The RAD dataset only had position data originally and was made in a pick and place scenario for transporting containers of suspicious liquid in a chemistry lab.
   CobotOps also has its own gripping task error states, though I chose not to build models for these due to time constraints.
   I can apply the same ESN models to the other datasets just for fun to see what the errors could have been. This is only for grins since there isn't any ground truth.
   
   The Cobotops and AURSAD data suffers from a crippling class imbalance, which I was only able to partially alleviate using undersampling.
   I reduced the amount of good runs by a ratio of 0.1, meaning I adjusted the ratios of the classes so that the largest one was no more than 10
   times larger than the smallest. This improved the recall of the model for finding the error cases, but only somewhat.</p>
   """


def baseline_training_text():
   return """
   <p>As a baseline, I also trained a logistic regression to compare to the ESN. The regression model took the same inputs of joint and position information from the undersampled data.
   It does not take any of the time dependencies into account and fails to find the errors in almost all cases.
   Notice that while the logistic regression has higher precision, it has very low recall, which is the most effective metric for catching the rare but important error cases.
   While not as effective as I'd hoped, the ESN is still the superior method for forecasting robot faults from time series data.</p>
   """


def esn_training_stats():
   return """
      ### Model 1: Echo State Network
           
      | Class | Precision | Recall | F1-Score |
      |-------|-----------|--------|----------|
      | Normal operation | 0.493 | 0.226 | 0.310 |
      | Damaged screw | 0.138 | 0.436 | 0.210 |
      | Extra assembly component | 0.116 | 0.610 | 0.195 |
      | Missing screw | 0.116 | 0.604 | 0.194 |"""


def log_reg_training_stats():
   return """
      ### Model 2: Logistic Regression
           
      | Class | Precision | Recall | F1-Score |
      |-------|-----------|--------|----------|
      | Normal operation | 0.502 | 0.221 | 0.307 |
      | Damaged screw | 0.400 | 0.001 | 0.003 |
      | Extra assembly component | 0.500 | 0.004 | 0.007 |
      | Missing screw | 0.857 | 0.005 | 0.011 |"""
