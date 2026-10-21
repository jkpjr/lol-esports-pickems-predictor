<p><b>Credit</b>: We use Oracle's Elixer's Lol esports match data by Tim Sevenhuysen at https://drive.google.com/drive/u/1/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH</p>

<div align="center">
  <h1>🏆 League of Legends Esports Match Predictor</h1>
  <p>A machine learning pipeline for forecasting professional match outcomes.</p>
</div>

<div>
   <p> We start with baseline models to compare with the main model. The pipeline includes: </p>
      <ol>
         <li> Data Loading: Choosing what dataset to use for the modeling. For now, we manually download the dataset into a local directory.</li>
         <li> Exploratory Data Analysis: Describe and visualize the data. Formerly named as `data_cleaning.ipynb`.</li>
         <li> Data Processing: Data imputation, outlier trimming, and dropping features based on EDA.</li>
         <li> Feature Engineering: Rolling metrics and "diff" features were engineered.</li>
         <li> Modeling: Logistic Regression, Random Forest, and XGboost models are trained, tested, and compared with a naive baseline model.</li>
      </ol>
</div>

<div>
   <h2></h2>
   <p><b>Alpha v0.1 Version</b>: Completed an end-to-end pipeline that predicts whether a team will win their next match based on their historical match data.</p>
</div>