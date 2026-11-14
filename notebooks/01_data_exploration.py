import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the flight dataset
df = pd.read_csv("data/Clean_Dataset.csv")

# Display the first five rows
print(df.head())
# 1. Check the number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# 2. Display all column names
print("\nColumn Names:")
print(df.columns)

# 3. Understand the data types and non-null values
print("\nDataset Information:")
df.info()

# 4. Get statistical information
print("\nStatistical Summary:")
print(df.describe())

# 5. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 6. Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Explore categorical columns

categorical_columns = [
    'airline',
    'flight',
    'source_city',
    'departure_time',
    'stops',
    'arrival_time',
    'destination_city',
    'class'
]

for column in categorical_columns:
    print("\nColumn:", column)
    print("Unique values:", df[column].nunique())
    print(df[column].unique())


# ---------------------------------------
# STEP 4: DATA VISUALIZATION
# ---------------------------------------

# Graph 1: Distribution of flight prices

plt.figure(figsize=(10, 5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Distribution of Flight Prices")
plt.xlabel("Ticket Price (₹)")
plt.ylabel("Number of Flights")
plt.tight_layout()
plt.savefig("results/price_distribution.png")
plt.close()


# Graph 2: Average price by airline

plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='airline', y='price', errorbar=None)
plt.title("Average Flight Price by Airline")
plt.xlabel("Airline")
plt.ylabel("Average Price (₹)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("results/airline_prices.png")
plt.close()


# Graph 3: Price comparison by ticket class

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='class', y='price')
plt.title("Flight Prices by Ticket Class")
plt.xlabel("Ticket Class")
plt.ylabel("Ticket Price (₹)")
plt.tight_layout()
plt.savefig("results/class_prices.png")
plt.close()


# Graph 4: How prices change with booking days left

daily_prices = df.groupby('days_left')['price'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=daily_prices, x='days_left', y='price', marker='o')
plt.title("Average Flight Price vs Days Left Before Departure")
plt.xlabel("Days Left Before Departure")
plt.ylabel("Average Price (₹)")
plt.tight_layout()
plt.savefig("results/days_left_prices.png")
plt.close()


# Graph 5: Average price by number of stops

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x='stops',
    y='price',
    errorbar=None,
    order=['zero', 'one', 'two_or_more']
)

plt.title("Average Flight Price by Number of Stops")
plt.xlabel("Number of Stops")
plt.ylabel("Average Ticket Price (₹)")
plt.tight_layout()
plt.savefig("results/stops_prices.png")
plt.close()


# Graph 6: Average price by source city

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x='source_city',
    y='price',
    errorbar=None
)

plt.title("Average Flight Price by Source City")
plt.xlabel("Source City")
plt.ylabel("Average Ticket Price (₹)")
plt.tight_layout()
plt.savefig("results/source_city_prices.png")
plt.close()
