# SOLARWIND2GIC
With the help of machine learning algorithms, this package takes data on solar wind at the L1 point and makes predictions on levels of geomagnetically induced currents (GICs) at the ground.

This script is split into five parts, with each part relying on the functions and objects defined in the sw2gic package.

## 1. Data Preparation

This code takes the last 26 years of geomagnetic field measurements from the Fürstenfeldbruck geomagnetic observatory (at the minute cadence), applies some basic data cleaning, and saves the data alongside OMNI solar wind data for the same period in a DataFrame format. These data are used as a proxy for geomagnetic variations in Austria, specifically at the Conrad Observatory. The geomagnetic variations at both stations are used to calculate the geoelectric field using 1D plane-wave modelling, and a fit of the two geoelectric field components is applied to GICs measured in power grid substations in Austria to determine the contribution of each component to the GICs.

## 2. Historical Analysis

An analysis of the data gathered in the data preparation is carried out. Distribution of values in both the geomagnetic field variations and the modelled GICs are plotted, and a case study of the 2003 Halloween storm is carried out. A table of the most geomagnetically active days (by three different measures) throughout the past 25 years is compiled. Some periods with high cumulative GICs (and the kind of signals that lead to them) are highlighted.

## 3. Extract Samples

This code takes geomagnetic field, modelled geoelectric field and OMNI data that have been conbined into a single DataFrame and extracts samples for training of machine learning models. The training data set and the test dataset are extracted and save separately for each target (geoelectric field components Ex and Ey or GICs from two substations, SS1 and SS2). In addition to the sampling, data from the testing years is also extracted as if the model had been run in 15-minute intervals to create a virtual "real-time" application from past data. Predictions from a "persistence" model are also created from a baseline comparison - this model works under the assumption that the geoelectric field or GICs that will be seen in the future are the same as what is being seen at the time of solar wind measurement.

## 4. Model Training

This script first tests some basic model architectures (a neural network, a gradient boosting regressor and an LSTM) to see which performs best at predicting the targets. After this basic analysis, it goes on to specifically train LSTMs to predict four different targets from OMNI-processed solar wind data. The four targets are the geoelectric field components Ex and Ey, and the modelled GICs at two specific substations, SS1 and SS5

## 5. Model Evaluation

This script loads LSTMs trained in 4_ModelTraining and evaluates and compares their forecasting skill. The different outputs can be compared to three different targets: the geoelectric field components, the modelled GICs (GIC_fit), or the measured GICs. Since GIC_fit functions only as an intermediary for lack of more GIC measurements, the only results considered in the study are ability in predicting the geoelectric field components (Ex and Ey) and the GICs at two substations (#1 and #5).

