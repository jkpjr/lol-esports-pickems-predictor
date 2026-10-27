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

    # List the exact counts and percentages of the missing values
    missing_count = df.isnull().sum()
    missing_pct = df.isnull().mean() * 100

    missing_df = pd.DataFrame({
        'Count': missing_count,
        '(%)': missing_pct.round(2)
    })

    missing_df = missing_df[missing_df['Count'] > 0].sort_values(
        by='(%)', ascending=False
    )

    print(f"\n{' MISSING VALUES REPORT ':-^50}")
    print(missing_df)  

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
        ban_cols = ['ban1', 'ban2', 'ban3', 'ban4', 'ban5']
        for object_column in df_copy.select_dtypes(include=['object']):
            if object_column in ['date','teamid','gameid', 'ban1'] or object_column in ban_cols:
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
    num_cols = (
        df.select_dtypes(include=["number"])
        .columns.drop(numcols_to_drop, errors="ignore")
        .tolist()
    )

    results = []
    for col in num_cols:
        outlier_mask = detect_outliers(df, col, IQR_multiplier)
        count = outlier_mask.sum()
        percentage = (count / len(df)) * 100

        results.append({
            "Column": col,
            "Count": count,
            "(%)": round(percentage, 2),
        })

    outliers_df = pd.DataFrame(results)

    outliers_df = outliers_df[outliers_df['Count'] > 0].sort_values(
        by="Count", ascending=False
    ).reset_index(drop=True)

    print(f"\n{" OUTLIER COUNTS AND PERCENTAGES PER COL ":-^50}")
    print(outliers_df)


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