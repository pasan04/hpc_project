import yaml
import pandas as pd

# Get the config.yml file
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

data_file = config['data_file']
processed_data_file = config['processed_data_file'] 

# Get the data to dataframe
df = pd.read_csv(data_file, encoding='latin1')

print(df.head(10))

# Get the U.S bounding data
us_bounds = {
    'min_lat': 24.396308, 'max_lat': 49.384358,
    'min_long': -125.000000, 'max_long': -66.934570
}

us_df = df[
    (df['latitude'].between(us_bounds['min_lat'], us_bounds['max_lat'])) &
    (df['longitude'].between(us_bounds['min_long'], us_bounds['max_long']))
].copy()

# Now the data is with the US bound
print(us_df.head(10))

# Convert timestamps to hourly/daily format
us_df['timestamp'] = pd.to_datetime(us_df['timestamp'], errors='coerce')
us_df['hour'] = us_df['timestamp'].dt.hour.astype('uint8')
us_df['day'] = us_df['timestamp'].dt.dayofweek.astype('uint8')  # Monday=0, Sunday=6
us_df['date'] = us_df['timestamp'].dt.date

us_df = us_df.drop(columns=['timestamp'])

print(f"Filtered data shape: {us_df.shape}")
print(us_df.head())

# Write to a new csv file
us_df.to_csv(processed_data_file)

