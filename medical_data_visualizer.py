import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
# Import the data from medical_examination.csv and assign it to the df variable.
df = pd.read_csv('medical_examination.csv')

# 2
# Add an overweight column to the data. To determine if a person is overweight, 
# first calculate their BMI by dividing their weight in kilograms by the square of their height in meters. 
# If that value is > 25 then the person is overweight. 
# Use the value 0 for NOT overweight and the value 1 for overweight.
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)


# 3
# Normalize data by making 0 always good and 1 always bad. 
# If the value of cholesterol or gluc is 1, set the value to 0. 
# If the value is more than 1, set the value to 1.
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
# Draw Categorical Plot
def draw_cat_plot():
    # 5
    # Create DataFrame for cat plot using `pd.melt` 
    # using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # 6
    # Group and reformat the data to split it by 'cardio'. 
    # Show the counts of each feature.
    
    
    # 7
    # Convert the data into long format and create a chart 
    # that shows the value counts of the categorical features 
    # using the following method provided by the seaborn library 
    # import: sns.catplot().
    df_cat = pd.DataFrame(df_cat.groupby(['cardio', 'variable', 'value'])['value'].count()).rename(columns={'value': 'total'}).reset_index()


    # 8
    # Get the figure for the output and store it in the fig variable.
    
    fig = sns.catplot(data=df_cat, x='variable', y='total', hue='value', col='cardio', kind='bar').figure
    plt.show()


    # 9
    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# 10
# Draw Heat Map
def draw_heat_map():
    # 11
    # Clean the data
    # Filter out the following patient segments that represent incorrect data:
    # diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
    # height is less than the 2.5th percentile (Keep the correct data with (df['height'] >= df['height'].quantile(0.025)))
    # height is more than the 97.5th percentile
    # weight is less than the 2.5th percentile
    # weight is more than the 97.5th percentile
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
                 (df['height'] >= df['height'].quantile(0.025)) & 
                 (df['height'] <= df['height'].quantile(0.975)) & 
                 (df['weight'] >= df['weight'].quantile(0.025)) & 
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    # Calculate the correlation matrix and store it in the variable corr.
    corr = df_heat.corr()

    # 13
    # Generate a mask for the upper triangle and store it in the variable mask.
    mask = np.triu(corr)

    # 14
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15
    # Plot the correlation matrix using seaborn's heatmap().
    sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, square=True, center=0, linewidths=.5, cbar_kws={'shrink': .5}, ax=ax).figure
    plt.show()

    # 16
    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
 