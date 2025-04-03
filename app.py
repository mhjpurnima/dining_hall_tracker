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
    print("=== get_status endpoint called ===")
    df = load_data()
    if df is None:
        return jsonify({"error": "Failed to load data"}), 500

    # Prepare data for training
    data_df = prepare_data(df)
    if data_df.empty:
        return jsonify({"error": "No valid data found"}), 500

    try:
        # Train the model
        model = train_model(data_df)

        # Prepare input for prediction based on the provided day
        input_data = pd.DataFrame({
            'day': [day]
        })

        # Predict the count
        predicted_count = model.predict(input_data)[0]
        print(f"Predicted count for {day}: {predicted_count}")

        # Determine status based on predicted count
        if predicted_count < 20:
            current_status = "SLOW"
        elif 20 <= predicted_count < 40:
            current_status = "MODERATE"
        else:
            current_status = "BUSY"

        return jsonify({"status": current_status, "predicted_count": predicted_count})

    except Exception as e:
        print("Error predicting status:", e)
        return jsonify({"error": "Failed to predict status"}), 500

@app.route("/get_peak_hour")    
def get_peak_hour():
    df = load_data()
    if df is None:
        return jsonify({"error": "Failed to load data"}), 500

    try:
        now = pd.Timestamp.now()
        current_day = now.strftime('%a').upper()
        current_date = f"{now.month}.{now.day}"  # Format date as 'M.D' (e.g., '2.2' for Feb 2)
        
        print(f"Current Date: {current_date}, Current Day: {current_day}")

        # Identify date and day columns
        day_columns = df.iloc[1, :]  # Get day names from second row
        print("day_columns: ", day_columns)
        
        # Check if current day exists in data
        matching_days = day_columns[day_columns == current_day]
        if matching_days.empty:
            print(f"Day {current_day} not found in data")
            return jsonify({"peak_hour": f"No data available for {current_day}"})
            
        day_column_index = matching_days.index[0]  # Find matching day column

        # Filter rows for the current date - make sure you're checking the right column
        date_columns = df.iloc[0, :]  # First row contains dates
        print("date_columns: ", date_columns)
        date_mask = date_columns.astype(str) == current_date
        
        # If current date not found, find most recent date
        if not any(date_mask):
            print(f"Date {current_date} not found in data")
            available_dates = date_columns[2:].dropna()  # Skip first two columns
            if not available_dates.empty:
                # Use the most recent date
                most_recent = available_dates.iloc[-1]
                current_date = most_recent
                date_mask = date_columns.astype(str) == current_date
                print(f"Using most recent date: {current_date}")
            else:
                return jsonify({"peak_hour": "No date data available"})
        
        # Find all columns matching the date (should be only one)
        date_col_indices = date_mask[date_mask].index.tolist()
        if not date_col_indices:
            return jsonify({"peak_hour": "No data available for this date"})
            
        date_col_index = date_col_indices[0]
        
        # Create a dataframe with time and count
        # Only include rows that have time values (skip meal headers)
        times = df.iloc[:, 1].astype(str)
        valid_time_mask = times.str.contains(':')
        time_values = times[valid_time_mask]
        count_values = df.loc[valid_time_mask, df.columns[date_col_index]]

        
        time_counts = pd.DataFrame({
            "time": time_values,
            "count": count_values
        })
        print("time_counts: ", time_counts)

        # Convert count to numeric
        time_counts["count"] = pd.to_numeric(time_counts["count"], errors="coerce").fillna(0).astype(int)

        # Find the max count
        max_count = time_counts["count"].max()
        if max_count <= 0:
            return jsonify({"peak_hour": "No significant data"})

        # Get peak time(s)
        peak_rows = time_counts[time_counts["count"] == max_count]
        if peak_rows.empty:
            return jsonify({"peak_hour": "No peak data found"})
            
        peak_time = peak_rows.iloc[0]["time"]  # Take the first peak time

        # Format peak time
        try:
            formatted_peak_time = pd.to_datetime(peak_time, format="%H:%M").strftime("%-I:%M %p")
        except:
            formatted_peak_time = peak_time  # Use as-is if formatting fails
        
        peak_hour = f"{formatted_peak_time}"

        print("Peak Hour:", peak_hour)
        return jsonify({
            "peak_hour": peak_hour,
            "date": current_date,
            "day": current_day
        })

    except Exception as e:
        print("Error finding peak hour:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"peak_hour": "Unknown", "error": str(e)})
        
# @app.route("/get_quiet_hours")
# def get_quiet_hours():
#     df = load_data()
#     if df is None:
#         return jsonify({"error": "Failed to load data"}), 500

if __name__ == "__main__":
    app.run(debug=True)
