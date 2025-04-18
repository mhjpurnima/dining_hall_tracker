from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime

app = Flask(__name__)
print("TEST",app.url_map)
# Load data from CSV file
def load_data():
    try:
        df = pd.read_csv("data/graze_data.csv", header=None, dtype=str)  # Read as strings to avoid type issues
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)  # Remove trailing spaces
        return df
    except Exception as e:
        print("Error loading CSV file:", e)
        return None

# Prepare data for training
def prepare_data(df):
    data = []
    for col in range(2, df.shape[1]):
        day = df.iloc[1, col]
        for row in range(2, df.shape[0]):
            time = df.iloc[row, 1]
            count = df.iloc[row, col]
            
            # Handle empty strings and NaN values
            if pd.isna(count) or count.strip() in ['', 'nan', 'NaN']:
                print(f"Skipping invalid entry at column {col}, row {row}")
                continue
                
            try:
                # Convert to integer safely
                count_int = int(float(count))
                data.append({
                    "day": day,
                    "time": time,
                    "count": count_int
                })
            except (ValueError, TypeError) as e:
                print(f"Skipping invalid count '{count}' at {time} on {day}: {str(e)}")
    
    return pd.DataFrame(data)

# Train a model to predict counts based on the day of the week
def train_model(df):
    # Prepare features and target
    df['day'] = df['day'].astype('category')
    
    # Features and target
    X = df[['day']]
    y = df['count']
    
    # Debugging: Print the DataFrame before training
    print("\n=== Debugging Training Data ===")
    print("DataFrame used for training:")
    print(df)
    
    # Check for missing values
    print("Missing values in DataFrame:")
    print(df.isnull().sum())
    
    # Count valid entries
    valid_entries = df.dropna().shape[0]
    print(f"Number of valid entries for training: {valid_entries}")
    
    # Log features and target values
    print("Features (X):")
    print(X)
    print("Target (y):")
    print(y)
    
    # Encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('day', OneHotEncoder(), ['day'])
        ],
        remainder='passthrough'
    )
    
    # Create a pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Train the model
    model.fit(X, y)
    return model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_status/<day>")
def get_status(day):
    df = load_data()
    if df is None:
        return jsonify({"error": "Failed to load data"}), 500

    print(f"\n=== DEBUGGING GET_STATUS FOR {day.upper()} ===")
    
    try:
        day = day.upper()
        print(f"1. Looking for day column: {day}")
        
        day_columns = df.iloc[1, :]
        print(f"2. Available days in data: {list(day_columns[2:])}")  # Skip first two empty columns
        
        day_column_index = day_columns[day_columns == day].index[0]
        print(f"3. Found {day} at column index: {day_column_index}")
        
        counts = []
        print("4. Scanning rows for valid counts:")
        for row in range(2, df.shape[0]):
            count = df.iloc[row, day_column_index]
            time_slot = df.iloc[row, 1]
            
            if pd.isna(count) or not str(count).isnumeric():
                print(f"   - Row {row} ({time_slot}): Invalid count '{count}' → Skipped")
                continue
                
            count_int = int(count)
            counts.append(count_int)
            print(f"   - Row {row} ({time_slot}): Valid count {count_int}")
        
        print(f"\n5. Analysis for {day}:")
        print(f"   - Total valid readings: {len(counts)}")
        print(f"   - All counts: {counts}")
        
        if not counts:
            print("   - ERROR: No valid counts found")
            return jsonify({"error": "No data available for this day"}), 404

        avg_count = sum(counts) / len(counts)
        print(f"   - Calculated average: {avg_count:.2f}")
        
        # Determine status
        if avg_count < 20:
            status = "Busy"
        elif 20 <= avg_count < 40:
            status = "Moderate"
        else:
            status = "Slow"
            
        print(f"   - Final status: {status}")
        
        return jsonify({
            "day": day,
            "status": status,
            "average_count": avg_count,
            "total_readings": len(counts)
        })

    except Exception as e:
        print(f"\n!!! ERROR PROCESSING {day}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Invalid day or data format"}), 400

@app.route("/get_peak_hour/<day>")
def get_peak_hour(day):
    df = load_data()
    if df is None:
        return jsonify({"error": "Failed to load data"}), 500

    print(f"\n=== DEBUGGING GET_PEAK_HOUR FOR {day.upper()} ===")
    
    try:
        day = day.upper()
        print(f"1. Looking for day column: {day}")
        
        day_columns = df.iloc[1, :]
        print(f"2. Available days in data: {list(day_columns[2:])}")
        
        day_column_index = day_columns[day_columns == day].index[0]
        print(f"3. Found {day} at column index: {day_column_index}")
        
        time_counts = []
        print("4. Scanning time slots:")
        for row in range(2, df.shape[0]):
            time_str = df.iloc[row, 1]
            count = df.iloc[row, day_column_index]
            
            if pd.isna(count) or not str(count).isnumeric():
                print(f"   - {time_str}: Invalid count '{count}' → Skipped")
                continue
                
            try:
                count_int = int(count)
                time_obj = datetime.strptime(time_str, "%H:%M").time()
                formatted_time = time_obj.strftime("%I:%M %p").lstrip('0')
                time_counts.append({
                    "time": formatted_time,
                    "count": count_int
                })
                print(f"   - {time_str} → {formatted_time}: Valid count {count_int}")
            except Exception as e:
                print(f"   - {time_str}: Error processing → {str(e)}")
                continue

        print(f"\n5. Peak hour analysis for {day}:")
        print(f"   - All valid time counts: {time_counts}")
        
        if not time_counts:
            print("   - ERROR: No valid time slots found")
            return jsonify({"error": "No data available for this day"}), 404

        max_count = max(tc['count'] for tc in time_counts)
        print(f"   - Maximum count found: {max_count}")
        
        peak_times = [tc['time'] for tc in time_counts if tc['count'] == max_count]
        print(f"   - Peak time(s) with max count: {peak_times}")
        
        return jsonify({
            "day": day,
            "peak_hours": peak_times,
            "peak_count": max_count,
            "total_readings": len(time_counts)
        })

    except Exception as e:
        print(f"\n!!! ERROR PROCESSING {day}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Invalid day or data format"}), 400

@app.route("/tracker")
def tracker():
    return render_template("tracker.html")

# @app.route("/get_quiet_hours")
# def get_quiet_hours():
#     df = load_data()
#     if df is None:
#         return jsonify({"error": "Failed to load data"}), 500

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        cleaned_df = prepare_data(df)
        trained_model = train_model(cleaned_df)
    else:
        print("Failed to load data for training.")

    app.run(debug=True)

