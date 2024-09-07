import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import datetime

# Load the dataset from an Excel file
file_path = r'C:\Input_Data.xlsx'
data = pd.read_excel(file_path)

# Convert the 'Date' column from string to datetime format
data['Date'] = pd.to_datetime(data['Tarih'], format='%d.%m.%Y')

# Extract additional time-based features from the 'Date' column
data['DayOfYear'] = data['Date'].dt.dayofyear  # Day of the year (1-366)
data['DayOfWeek'] = data['Date'].dt.dayofweek  # Day of the week (0-6, where Monday=0)
data['WeekOfYear'] = data['Date'].dt.isocalendar().week  # Week number of the year
data['Month'] = data['Date'].dt.month  # Month (1-12)
data['Year'] = data['Date'].dt.year  # Year

# Encode categorical variables to numeric values
location_encoder = LabelEncoder()
data['Location'] = location_encoder.fit_transform(data['Location'])  # Encode location names to numbers

Type_encoder = LabelEncoder()
data['Type'] = Type_encoder.fit_transform(data['Type'])  # Encode Type types to numbers

# Create a dictionary to map 'ProductCode' to 'Type'
product_Type_mapping = data[['ProductCode', 'Type']].drop_duplicates().set_index('ProductCode')['Type'].to_dict()

# Define features (X) and target variable (y) for the model
X = data[['DayOfYear', 'DayOfWeek', 'WeekOfYear', 'Month', 'Year', 'Location', 'ProductCode', 'Type']]
y = data['Quantity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Set up a grid search for hyperparameter tuning of the Random Forest model
param_grid = {
    'n_estimators': [29],  # Number of trees in the forest
    'max_depth': [10, 20],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required to be at a leaf node
    'bootstrap': [True, False]  # Whether to use bootstrap samples when building trees
}

# Perform grid search with cross-validation to find the best model
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=0), param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Select the best model from grid search results
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate the Root Mean Squared Error (RMSE) of the model's predictions
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model RMSE: {rmse}")

# Prepare a DataFrame to compare actual and predicted values
results_df = X_test.copy()
results_df['Actual'] = y_test.values  # Actual values from the test set
results_df['Predicted'] = y_pred  # Predicted values from the model

# Convert the numerical day of year and year back to date format
def day_of_year_to_date(year, day_of_year):
    return datetime.datetime(int(year), 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)

results_df['Date'] = results_df.apply(lambda row: day_of_year_to_date(row['Year'], row['DayOfYear']), axis=1)
results_df['Date'] = results_df['Date'].dt.strftime('%d.%m.%Y')  # Format the date for display

# Decode the encoded categorical variables back to their original values
results_df['Location'] = location_encoder.inverse_transform(results_df['Location'])
results_df['Type'] = Type_encoder.inverse_transform(results_df['Type'])

print("\nActual vs. Predicted values:")
print(results_df.head(10))  # Display the first 10 rows for review

# Generate future dates for forecasting
last_date = data['Date'].max()
last_date = pd.to_datetime(last_date, format='%d.%m.%Y')
future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 8)]  # Forecast for the next 7 days

# Calculate the distribution of 'ProductCode' for each day of the year and location
product_code_distribution = data.groupby(['DayOfYear', 'Location', 'ProductCode']).size().reset_index(name='Count')

# Function to select the most frequent product codes for a given day and location
def select_product_code_set(day_of_year, location, top_n=13):
    relevant_distribution = product_code_distribution[
        (product_code_distribution['DayOfYear'] == day_of_year) & 
        (product_code_distribution['Location'] == location)
    ]
    
    # Select the top 'top_n' product codes based on frequency
    top_product_codes = relevant_distribution.nlargest(top_n, 'Count')['ProductCode']
    
    if len(top_product_codes) < top_n:
        # If not enough product codes are available, randomly sample additional ones
        additional_product_codes = data['ProductCode'].sample(top_n - len(top_product_codes)).values
        top_product_codes = pd.concat([top_product_codes, pd.Series(additional_product_codes)])
    
    return top_product_codes.unique()  # Return unique product codes

# Create data for future predictions
future_data = []
for date in future_dates:
    day_of_year = date.timetuple().tm_yday
    day_of_week = date.weekday()
    week_of_year = date.isocalendar()[1]
    month = date.month
    year = date.year
    
    for loc in data['Location'].unique():
        selected_product_codes = select_product_code_set(day_of_year, loc, top_n=13)
        for code in selected_product_codes:
            Type = product_Type_mapping[code]
            future_data.append([day_of_year, day_of_week, week_of_year, month, year, loc, code, Type])

future_df = pd.DataFrame(future_data, columns=['DayOfYear', 'DayOfWeek', 'WeekOfYear', 'Month', 'Year', 'Location', 'ProductCode', 'Type'])

# Predict future quantities using the trained model
future_df['Quantity'] = best_model.predict(future_df)

# Optionally add noise to the predicted quantities to simulate variability
noise_scale = y_train.std() * 0.05
future_df['Quantity'] += np.random.normal(0, noise_scale, len(future_df))

# Remove duplicate entries for 'Date' and 'ProductCode'
def filter_duplicates(df):
    return df.drop_duplicates(subset=['Date', 'ProductCode'], keep='first')  # Keep the first occurrence of each combination

# Convert day of year and year back to date format
future_df['Date'] = future_df.apply(lambda row: day_of_year_to_date(row['Year'], row['DayOfYear']), axis=1)
future_df['Date'] = future_df['Date'].dt.strftime('%d.%m.%Y')

# Decode the encoded categorical variables back to their original values
future_df['Location'] = location_encoder.inverse_transform(future_df['Location'])
future_df['Type'] = Type_encoder.inverse_transform(future_df['Type'])

# Remove duplicates from the future data
filtered_future_df = filter_duplicates(future_df)

# Add empty columns for additional data that might be required
filtered_future_df['ReferenceNo'] = ""
filtered_future_df['InvoiceDescription'] = ""

# Add product descriptions based on 'ProductCode'
product_description_mapping = data.set_index('ProductCode')['UrunAciklamasi'].to_dict()
filtered_future_df['ProductDescription'] = filtered_future_df['ProductCode'].map(product_description_mapping)

# Convert 'Date' to datetime for sorting purposes
filtered_future_df['Date'] = pd.to_datetime(filtered_future_df['Date'], format='%d.%m.%Y')

# Sort the data by 'Date' and 'Location'
filtered_future_df.sort_values(by=['Date', 'Location'], ascending=[False, False], inplace=True)

# Convert 'Date' back to string format after sorting
filtered_future_df['Date'] = filtered_future_df['Date'].dt.strftime('%d.%m.%Y')

# Reorder the columns as specified
filtered_future_df = filtered_future_df[['Date', 'ReferenceNo', 'InvoiceDescription', 'Location', 'ProductDescription', 'ProductCode', 'Type', 'Quantity']]

# Save the final dataset to an Excel file
final_output_path = r'C:\Forecast.xlsx'
filtered_future_df.to_excel(final_output_path, index=False)

print("DONE")
