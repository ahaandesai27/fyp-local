"""
Process raw Kaggle / UCI Individual Household Electric Power Consumption dataset:
- Cleans missing values ('?')
- Parses timestamps
- Resamples from 1-minute to 15-minute intervals
- Converts kW to Watts
- Computes cyclic time features
- Exports to dataset/kaggle_household_power_15min.csv
"""
import os
import numpy as np
import pandas as pd


def process_kaggle_data(input_path="dataset/household_power_consumption.txt",
                        output_path="dataset/kaggle_household_power_15min.csv",
                        sample_days=180):
    print(f"Reading raw data from {input_path}...")

    # Read CSV with ';' separator and '?' as NaN
    # We take sample_days of continuous real data (e.g. 180 days = ~6 months)
    # 180 days * 24 * 60 = 259,200 rows
    max_rows = sample_days * 24 * 60

    df = pd.read_csv(
        input_path,
        sep=';',
        nrows=max_rows,
        na_values=['?'],
        dtype={
            'Global_active_power': float,
            'Global_reactive_power': float,
            'Voltage': float,
            'Global_intensity': float,
            'Sub_metering_1': float,
            'Sub_metering_2': float,
            'Sub_metering_3': float,
        },
        low_memory=False
    )

    print(f"Loaded {len(df):,} raw 1-minute rows.")

    # Combine Date and Time
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('datetime', inplace=True)
    df.drop(['Date', 'Time'], axis=1, inplace=True)

    # Forward fill small gaps in measurements
    df.interpolate(method='linear', limit=10, inplace=True)
    df.bfill(inplace=True)

    print("Resampling to 15-minute intervals...")
    resampled = df.resample('15min').mean()

    # Convert kW to Watts for active power
    resampled['global_active_power'] = resampled['Global_active_power'] * 1000.0

    # Sub-metering conversion
    resampled['sub_meter_kitchen'] = resampled['Sub_metering_1']
    resampled['sub_meter_laundry_bath'] = resampled['Sub_metering_2']
    resampled['sub_meter_hvac'] = resampled['Sub_metering_3']
    resampled['voltage'] = resampled['Voltage']
    resampled['global_intensity'] = resampled['Global_intensity']
    resampled['global_reactive_power'] = resampled['Global_reactive_power']

    # Cyclic time features
    hours = resampled.index.hour + resampled.index.minute / 60.0
    resampled['hour_sin'] = np.sin(hours / 24.0 * 2.0 * np.pi)
    resampled['hour_cos'] = np.cos(hours / 24.0 * 2.0 * np.pi)
    resampled['day_of_week'] = resampled.index.dayofweek
    resampled['is_weekend'] = (resampled.index.dayofweek >= 5).astype(int)

    final_cols = [
        'global_active_power',
        'sub_meter_kitchen',
        'sub_meter_laundry_bath',
        'sub_meter_hvac',
        'voltage',
        'global_intensity',
        'global_reactive_power',
        'hour_sin',
        'hour_cos',
        'day_of_week',
        'is_weekend'
    ]

    out_df = resampled[final_cols].reset_index()
    out_df.rename(columns={'datetime': 'timestamp'}, inplace=True)
    out_df.dropna(inplace=True)

    out_df.to_csv(output_path, index=False)
    print(f"Processed dataset saved successfully to: {output_path}")
    print(f"Total 15-minute intervals: {len(out_df):,} | Days: {len(out_df)/(24*4):.1f}")
    print(f"Power (W) - Mean: {out_df['global_active_power'].mean():.1f} W | Min: {out_df['global_active_power'].min():.1f} W | Max: {out_df['global_active_power'].max():.1f} W")
    peaks = len(out_df[out_df['global_active_power'] >= 3500.0])
    print(f"High power peaks (>= 3500W): {peaks}")


if __name__ == "__main__":
    process_kaggle_data()
