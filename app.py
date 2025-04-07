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

    try:
        day = day.upper()
        # Find the column for the requested day
        day_columns = df.iloc[1, :]
        day_column_index = day_columns[day_columns == day].index[0]
        
        # Get all counts for this day
        counts = []
        for row in range(2, df.shape[0]):
            count = df.iloc[row, day_column_index]
            if pd.isna(count) or not str(count).isnumeric():
                continue
            counts.append(int(count))
        
        if not counts:
            return jsonify({"error": "No data available for this day"}), 404

        avg_count = sum(counts) / len(counts)
        
        # Determine status
        if avg_count < 20:
            status = "Busy"
        elif 20 <= avg_count < 40:
            status = "Moderate"
        else:
            status = "Slow"

        return jsonify({
            "day": day,
            "status": status,
            "average_count": avg_count,
            "total_readings": len(counts)
        })

    except Exception as e:
        print(f"Error processing day {day}:", e)
        return jsonify({"error": "Invalid day or data format"}), 400

@app.route("/get_peak_hour/<day>")
def get_peak_hour(day):
    df = load_data()
    if df is None:
        return jsonify({"error": "Failed to load data"}), 500

    try:
        day = day.upper()
        # Find the column for the requested day
        day_columns = df.iloc[1, :]
        day_column_index = day_columns[day_columns == day].index[0]
        
        # Get all time slots and counts for this day
        time_counts = []
        for row in range(2, df.shape[0]):
            time_str = df.iloc[row, 1]
            count = df.iloc[row, day_column_index]
            
            if pd.isna(count) or not str(count).isnumeric():
                continue
                
            try:
                # Convert to datetime and format as AM/PM
                time_obj = datetime.strptime(time_str, "%H:%M").time()
                formatted_time = time_obj.strftime("%I:%M %p").lstrip('0')
                time_counts.append({
                    "time": formatted_time,
                    "count": int(count)
                })
            except Exception as e:
                print(f"Error processing time {time_str}: {str(e)}")
                continue

        if not time_counts:
            return jsonify({"error": "No data available for this day"}), 404

        # Find peak hour(s)
        max_count = max(tc['count'] for tc in time_counts)
        peak_times = [tc['time'] for tc in time_counts if tc['count'] == max_count]
        print(peak_times)
        return jsonify({
            "day": day,
            "peak_hours": peak_times,
            "peak_count": max_count,
            "total_readings": len(time_counts)
        })

    except Exception as e:
        print(f"Error processing day {day}:", e)
        return jsonify({"error": "Invalid day or data format"}), 400

# @app.route("/get_quiet_hours")
# def get_quiet_hours():
#     df = load_data()
#     if df is None:
#         return jsonify({"error": "Failed to load data"}), 500

if __name__ == "__main__":
    app.run(debug=True)
