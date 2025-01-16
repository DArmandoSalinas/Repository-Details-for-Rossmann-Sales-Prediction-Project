import pandas as pd  # Data manipulation 
import numpy as np  # Numerical operations

# Assigning the csv files to read them and to process the info
store_data = pd.read_csv(r"C:\Users\Salin\OneDrive\Documentos\ESSEX\Neural Networks\store.csv")  
test_data = pd.read_csv(r"C:\Users\Salin\OneDrive\Documentos\ESSEX\Neural Networks\test.csv")  
train_data = pd.read_csv(r"C:\Users\Salin\OneDrive\Documentos\ESSEX\Neural Networks\train.csv")  

# For 'store.csv'
# There are some missing values in 'CompetitionDistance', so they can be replaced with the median value 
store_data['CompetitionDistance'] = store_data['CompetitionDistance'].fillna(store_data['CompetitionDistance'].median())

# for 'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear', they can be filled with 0 since there is no data
store_data['CompetitionOpenSinceMonth'] = store_data['CompetitionOpenSinceMonth'].fillna(0)
store_data['CompetitionOpenSinceYear'] = store_data['CompetitionOpenSinceYear'].fillna(0)

# for 'Promo2SinceWeek' and 'Promo2SinceYear' same
store_data['Promo2SinceWeek'] = store_data['Promo2SinceWeek'].fillna(0)
store_data['Promo2SinceYear'] = store_data['Promo2SinceYear'].fillna(0)

# for missing values in 'PromoInterval' i fill with an empty string because there is no periodic promtion intervals 
store_data['PromoInterval'] = store_data['PromoInterval'].fillna('')

# Now for 'test.csv' 
# Filling the blanks in 'Open' with 1 assuming is open
test_data['Open'] = test_data['Open'].fillna(1)

# For easy manipulation, I convert 'Date' column in training and test data to datetime objects 
train_data['Date'] = pd.to_datetime(train_data['Date'], format='%d/%m/%Y')  
test_data['Date'] = pd.to_datetime(test_data['Date'], format='%d/%m/%Y')

# Applying Featuring Engineering at extracting useful date related features
# LOoping through both train and test datasets in order to create new important columns in the train_data and test_data
for dataset in [train_data, test_data]:
    dataset['Year'] = dataset['Date'].dt.year  # Extracting the year 
    dataset['Month'] = dataset['Date'].dt.month  # Extracting the month 
    dataset['Day'] = dataset['Date'].dt.day  # Extracting the day 
    dataset['WeekOfYear'] = dataset['Date'].dt.isocalendar().week  # Extracting the week number of the year
    dataset['DayOfYear'] = dataset['Date'].dt.dayofyear  # Extracting the day of the year

# Merging store data into train and test datasets
# For combining and build both files, i use the specific data of 'store' using 'Store' as the key
train_data = pd.merge(train_data, store_data, how='left', on='Store')  
test_data = pd.merge(test_data, store_data, how='left', on='Store')  

# To calculate Competition Open duration as number of months and leave this as a new feature

train_data['CompetitionOpen'] = 12 * (train_data['Year'] - train_data['CompetitionOpenSinceYear']) + \
                                 (train_data['Month'] - train_data['CompetitionOpenSinceMonth'])
test_data['CompetitionOpen'] = 12 * (test_data['Year'] - test_data['CompetitionOpenSinceYear']) + \
                                (test_data['Month'] - test_data['CompetitionOpenSinceMonth'])

# If it has not started, if there is negative values, replacing with zero
train_data['CompetitionOpen'] = train_data['CompetitionOpen'].apply(lambda x: max(0, x))
test_data['CompetitionOpen'] = test_data['CompetitionOpen'].apply(lambda x: max(0, x))

# Calculating duration for active Promo2 in months to create this new feature

# /4 to convert to months 
train_data['Promo2Open'] = 12 * (train_data['Year'] - train_data['Promo2SinceYear']) + \
                            (train_data['WeekOfYear'] - train_data['Promo2SinceWeek']) / 4.0
test_data['Promo2Open'] = 12 * (test_data['Year'] - test_data['Promo2SinceYear']) + \
                           (test_data['WeekOfYear'] - test_data['Promo2SinceWeek']) / 4.0

# If it has not started, if there is negative values, replacing with zero
train_data['Promo2Open'] = train_data['Promo2Open'].apply(lambda x: max(0, x))
test_data['Promo2Open'] = test_data['Promo2Open'].apply(lambda x: max(0, x))

# Setting the now unnecessary columns for modelling
columns_to_drop = ['Date', 'Customers', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                   'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

# Dropping these columns if they exist in each file
train_data = train_data.drop(columns=[col for col in columns_to_drop if col in train_data.columns])
test_data = test_data.drop(columns=[col for col in columns_to_drop if col in test_data.columns])
store_data = store_data.drop(columns=[col for col in columns_to_drop if col in store_data.columns])

# Saving the preprocessed datasets and creating the new cleaned files
train_data.to_csv(r"C:\Users\Salin\OneDrive\Documentos\ESSEX\Neural Networks\train_preprocessed.csv", index=False)
test_data.to_csv(r"C:\Users\Salin\OneDrive\Documentos\ESSEX\Neural Networks\test_preprocessed.csv", index=False)
store_data.to_csv(r"C:\Users\Salin\OneDrive\Documentos\ESSEX\Neural Networks\store_preprocessed.csv", index=False)

print("Data preprocessing complete. Preprocessed files saved.")
