import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

# Formats the global decimal places to 2
pd.options.display.float_format = '{:.2f}'.format

# EDA and Data Processing
def null_map(df, figsize=(14, 8)):
    """
    Maps the missing values using the missingno library
    """
    plt.figure(figsize=figsize)
    msno.matrix(df, labels=True, fontsize=8)
    plt.title('MISSING VALUES MAP', fontsize=24, fontweight="bold")
    plt.show()

    print(f"\n{" PERCENTAGE OF MISSING VALUES ":-^50}") 
    missing_percentage = (df.isnull().mean() * 100).sort_values(ascending=False)
    print(missing_percentage[missing_percentage > 0])   

def data_profiling(df):
    """
    Present the profile of the dataframe, including
    description, number of unique values, and unique values of object series
    """  
    df_copy = df.copy()
    with pd.option_context('display.max_seq_items', None): # list all columns of the df with no limits
        print(f"List of Columns: {df_copy.columns}")
    
    with pd.option_context('display.max_rows', None): # list all rows of the df with no limits
        print(f"\n{" COUNT OF UNIQUE VALUES ":-^50}")
        display(df_copy.nunique())

        print(f"\n{" UNIQUE VALUES OF OBJECT SERIES ":-^50}")
        for object_column in df_copy.select_dtypes(include=['object']):
            if object_column == 'gameid': # skip gameid series
                continue
            print("\n",df_copy[object_column].value_counts())

def detect_outliers(df, col, IQR_multiplier=2.0):
    # Setting the IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Standard 1.5 multiplier for outlier bounds
    lower_bound = Q1 - IQR_multiplier * IQR
    upper_bound = Q3 + IQR_multiplier * IQR

    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    return outliers

def count_outliers(df, numcols_to_drop, IQR_multiplier=2.0):
    """
    Utilizes IQR to detect outliers.
    """
    num_cols = df.select_dtypes(include=["number"]).columns.drop(numcols_to_drop).tolist()

    print(f"\n{" COUNT OF OUTLIERS ":-^50}")
    for col in num_cols:
        outliers = df[detect_outliers(df, col, IQR_multiplier)]
        if (len(outliers)/len(df[col])) > 0.00:
            print(f"{f"{col}:":<23} {len(outliers):>5} ({len(outliers)/len(df[col]):<.2%})")


def visualize_outliers(df, numcols_to_drop, plot_title="BOXPLOTS OF NUMERICAL COLUMNS", nrows=20, ncols=4, figsize=(20, 80)):
    """
    Utilize box plots to detect outliers.
    """
    num_cols = df.select_dtypes(include=["number"]).columns.drop(numcols_to_drop)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize) # Adjust depending on the number of numerical columns
    axes = axes.flatten()
    fig.suptitle(plot_title.upper(), fontsize=16, fontweight="bold", y=1.02)

    for i, col in enumerate(num_cols):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(col, fontsize=12, fontweight="bold")
        axes[i].set_ylabel("")

    # Hide unused subplots
    for j in range(len(num_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def visualize_num_distribution(df, numcols_to_drop, plot_title="DISTRIBUTION OF NUMERICAL COLUMNS", nrows=20, ncols=4, figsize=(20, 80)):
    """
    Utilize histograms to visualize numerical distribution.
    """
    num_cols = df.select_dtypes(include=["number"]).columns.drop(numcols_to_drop)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize) # Adjust depending on the number of numerical columns
    axes = axes.flatten()
    fig.suptitle(plot_title.upper(), fontsize=16, fontweight="bold", y=1.02)

    for i, col in enumerate(num_cols):
        sns.histplot(x=df[col], ax=axes[i])
        axes[i].set_title(col, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("")

    # Hide unused subplots
    for j in range(len(num_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def visualize_cat_distribution(df, catcols_to_drop, plot_title="DISTRIBUTION OF CATEGORICAL DATA", nrows=20, ncols=4, figsize=(15, 15)):
    """
    Utilize bar plots to visualize categorical distribution.
    """
    cat_cols = df.select_dtypes(include=["str"]).columns.drop(catcols_to_drop)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize) # Adjust depending on the number of numerical columns
    axes = axes.flatten()
    fig.suptitle(plot_title.upper(), fontsize=16, fontweight="bold", y=1.02)

    for i, col in enumerate(cat_cols):
        sns.countplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(col, fontsize=12, fontweight="bold")
        axes[i].set_ylabel("")
        axes[i].tick_params(axis="x", rotation=45)

    # Hide unused subplots
    for j in range(len(cat_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()