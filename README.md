We start with a naive baseline and a simple logistic regression model to compare with the main model.
   - Data cleaning script to help with data processing included. 

Credit: We use Oracle's Elixer's Lol esports match data by Tim Sevenhuysen, here
https://drive.google.com/drive/u/1/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH
Currently using 2014-2025 datasets

Download the datasets first and export to your local repository.

The pipeline progresses as follows:
data_loading -> exploratory_data_analysis -> data_processing -> feature_engineering -> baseline_model

Note that data_cleaning.ipynb is outdated and was renamed to exploratory_data_analysis