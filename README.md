# transitpredictiontool
Honda Capstone Project SP2025



# Honda Capstone Transit Times Prediction Tool

This repository forecasts the expected arrival times for ongoing trips.

This set of scripts uses 4 different scripts: (1) SplitDataset.py, (2) TripTraining411.py, (3) PredictionModel411.py, (4) MetricsEval.py




## Installation

Install these libraries to set up the environment:
    
    pandas – data loading & manipulation
    numpy – numerical operations
    openpyxl – Excel I/O support in pandas
    scikit‑learn – train/validation splitting along with evaluation metrics 
    xgboost – the gradient‑boosted tree model
    shap – explainability (SHAP summary plots)
    matplotlib – plotting residuals and SHAP outputs
    haversine – haversine distance calcuation used for DistanceToDestination/Origin



```bash
  pip install pandas
  pip install numpy
  pip install openpyxl
  pip install scikit‑learn
  pip install xgboost
  pip install shap
  pip install matplotlib
  pip install haversine
  
```
    
## Features

Input Features used to predict remaining transit time:

    StartHour: Scheduled hour of trip start
    StartMinute: Scheduled minute of trip start
    StartDayOfWeek: Scheduled day of week (Mon = 0, Sun = 6)
    WeekendTrip: integer value = 1 if StartDayOfWeek >= (Sat/Sun) else 0
    ElapsedTime: Minutes since ActualStart at each GPS Ping
    DepartureDelay: Minutes since ActualStart and ScheduledStart
    ScheduledTripDuration: Scheduled duration
    Latitude, Longitude: Raw GPS coordinates
    LatLongVector: Euclidean norm of latitude & longitude 
    DistanceToDestination: Haversine distance from current ping to destination
    DistanceFromOrigin: Haversine distance from latest ping to origin
    CarrierCode: Categorical encoding of the Transit Carrier Company
    RouteNumberCode: Categorical encoding of RouteNumber


## Environment Variables

Folders within the environment: 

InputData: Holds all the raw source files: 

    - All_trips.xlsx (Master Trip Records)
    - GPS_data.xlss (time stamped gps logs)
    - CAPSTONE_ROUTES.xlsx (21 longest delayed routes)
SPLIT_DATA_FOLDER: Path for reading & writing the split datasets

    C:\Capstone_Project\SplitData

OutputData: Path for model outputs & plots

    Trained Model Files: transit_time_model_<timestamp>.xlsx
    SHAP Plots: SHAP_Summary_<timestamp>.png
    Prediction Workbooks: PredictionResults_<timestamp>.xlsx

Evaluation Metrics: Final Diagnostics & Model Comparison

    Metrics Spreadsheet: Evaluation_Metrics_<timestamp>.xlsx
    Residual Plot: Residual_Plot_<timestamp>.png
    Error Distribution Plot: Error_Distribution_<timestamp>.png

PING_PERCENTAGE: Fraction of GPS Pings to retain. Purpose is to simulate partial trip data and to remove any possibilities 

random_state: This ensures that the data splitting and shuffling produces the same results on every run. The fixed seed allows for stability in the model.

n_estimators=100

learning_rate=0.1

To contextualize this, with 100 boosting trees, there is a proposition for each decision tree there is a correction to the predicted remaining minutes, and with a learning rate of 0.1, there is a 10% adjustment. These adjustments prevent the model from making large and unstable jumps. For all 100 trees, the model will gradually converge to the true remaining time without overshooting.

## Deployment

To deploy this project, download all input_data files from Ceva Matrix and all four python scripts: 

    (1) SplitDataset.py
    (2) TripTraining411.py
    (3) PredictionModel411.py
    (4) MetricsEval.py


## Project Contacts

- Rohan Madan: rohanmadan02@gmail.com
- Juan Galvis (Sponsor): juan_galviscamargo@na.honda.com


## Documentation

[Sharepoint Folder](https://buckeyemailosu.sharepoint.com/:f:/s/4900Capstone398/El4aRU-sJwFJosYAiksrUiIBEQ39cvVyRRpDIxMKKejbWA?e=pxd6B2)


## Demo
[Demo Video](https://buckeyemailosu.sharepoint.com/:v:/s/4900Capstone398/EaBUMQkS0j5EjuQT5Keb7HIBHw6B2mD_CwGApALMPNwrUQ?e=4msJ88)


## Appendix

Any additional information goes here

https://www.nvidia.com/en-us/glossary/xgboost/
https://scikit-learn.org/stable/install.html
https://medium.com/@venujkvenk/anomaly-detection-techniques-a-comprehensive-guide-with-supervised-and-unsupervised-learning-67671cdc9680
https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html
https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py
https://www.osti.gov/servlets/purl/1733262
## Optimizations

What optimizations did you make in your code? E.g. refactors, performance improvements, accessibility


## Troubleshooting & FAQ

#### Question 1

Answer 1

#### Question 2

Answer 2


## Roadmap & Next Steps


Automate delay alerts based on the geofence triggers

Quantify the financial and inventory implications of each trip delay

Challenge scheduled start times to improve scheduling efficiency and eliminate early start trips

Contextualize the input data further to implement more refined features in the ML model – this will increase the accuracy of the model

Automate the ML Prediction Script to run every morning for streamlined efficiency

Automate the data scraping process with Selenium


