import pandas as pd
import matplotlib.pyplot as plt

# Load data into pandas DataFrame
df = pd.read_csv("Data.csv")

def get_top_10_diseases(df):
    # Calculate the total number of cases for each disease
    top_ten = df.groupby('Disease')['Cases'].sum().nlargest(10).index
    return df[df['Disease'].isin(top_ten)]

def disease_time_trend(df):
    # Get top 10 diseases by number of cases
    top_ten = df.groupby('Disease')['Cases'].sum().nlargest(10).index
    df_top_10 = df[df['Disease'].isin(top_ten)]
    
    # Group by year and disease to see trends over time
    time_trend = df_top_10.groupby(['Year', 'Disease'])['Cases'].sum().unstack()

    # Figure size
    plt.figure(figsize=(18, 10))

    # Line plot for top 10 diseases by number of cases
    time_trend.plot(kind='line', marker='o')

    # Add title and labels
    plt.title('Trend of Top 10 Disease Cases Over Years in California', fontsize=20)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Number of Cases', fontsize=16)

    # Move the legend outside the plot
    plt.legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Adjust the margins to accommodate the legend and title
    plt.tight_layout()
    plt.subplots_adjust(right=0.5, top=0.9, left=0.1, bottom=0.15)
    
    # Show plot
    plt.show()

def disease_by_county(df, disease_name):
    # Get top 10 diseases by number of cases
    df_top_10 = get_top_10_diseases(df)

    # Ensure the disease is one of the top 10
    if disease_name not in df_top_10['Disease'].unique():
        print(f"Disease '{disease_name}' is not in the top 10 diseases by cases.")
        return

    # Filter for the specific disease and group by county
    county_disease_data = df_top_10[df_top_10['Disease'] == disease_name].groupby('County')['Cases'].sum()

    # Plot a bar graph to show cases by county
    plt.figure(figsize=(10, 6))
    county_disease_data.plot(kind='bar')

    plt.title(f'Cases of {disease_name} by County (Top 10 Disease)')
    plt.xlabel('County')
    plt.ylabel('Number of Cases')

    # Rotate x-axis 
    plt.xticks(rotation=90, ha='right')

    # Adjust layout to ensure labels are not cut off
    plt.tight_layout()

    plt.show()

def disease_by_gender(df):
    # Get top 10 diseases by number of cases
    df_top_10 = get_top_10_diseases(df)
    
    # Group by disease and gender
    gender_data = df_top_10.groupby(['Disease', 'Sex'])['Cases'].sum().unstack()

    # Use a stacked bar chart for gender distribution
    plt.figure(figsize=(12, 6)) 
    gender_data.plot(kind='bar', stacked=True)

    plt.title('Gender Distribution of Top 10 Disease Cases')
    plt.xlabel('Disease')
    plt.ylabel('Number of Cases')

    # Rotate x-axis labels to avoid overlap
    plt.xticks(rotation=45, ha='right')

    # Adjust layout
    plt.tight_layout()

    plt.show()


# For visualizing time trend
disease_time_trend(df)
# Replace disease with the name of a disease you want to focus on
disease_by_county(df, 'Campylobacteriosis') 
# For visualizing gender distribution
disease_by_gender(df)

