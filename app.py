from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import smtplib 
from email.message import EmailMessage
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

def mail_template(user_email,message):
    admin_email = 'maharjanpurnima880@gmail.com'
    msg = EmailMessage()
    msg['Subject'] = f'Feedback from {user_email}'
    msg['From'] = admin_email
    msg['To'] = admin_email
    submission_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_body = message
    email_template = f"""
Subject: [TooGrazy] New Feedback from {user_email}

Hi Team,

A new feedback entry has been submitted via the “Feedback page” form.

─ Submission Details ──────────────────────────────────────
Date & Time : {submission_timestamp}  (server time)
Email       : {user_email}

Message:
-----------------------------------------------------------
{message_body}
-----------------------------------------------------------

"""
    msg.set_content(email_template)
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(admin_email, 'hrrtbowkcookrwic')
        smtp.send_message(msg)
@app.route("/send_email", methods=["POST"])
def send_email():
    email = request.form.get("email")
    message = request.form.get("message")
    print(message)
    mail_template(email, message)
    print("Email sent successfully")
    return render_template("index.html")

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

@app.route("/hours")
def hours():
    return render_template("hours.html")
@app.route("/more-info")
def more_info():
    return render_template("more-info.html")

@app.route("/get_status/<day>")
def get_status(day):
    df = load_data()
    if df is None:
        return jsonify({"error": "Failed to load data"}), 500
    
    try:
        # Round current time to nearest 15 minutes
        now = datetime.now()
        rounded_minute = (now.minute // 15) * 15
        rounded_time = now.replace(minute=rounded_minute, second=0, microsecond=0)
        formatted_time = rounded_time.strftime("%#I:%M")  # Windows format
        current_hour = rounded_time.hour

        # Determine meal period based on current hour
        # Determine meal period based on current hour
        if 6 <= current_hour < 11:
            meal_period = "BREAKFAST"
        elif 11 <= current_hour < 16:
            meal_period = "LUNCH"
        elif 16 <= current_hour < 23:  # Now includes 10:00 PM (22:00)
            meal_period = "DINNER"
        else:
            meal_period = "LATE NIGHT"

        day = day.upper()
        print(f"\n=== Analyzing {day} at {rounded_time} ({meal_period}) ===")

        # Find matching meal period and time slot
        meal_column = df.iloc[:, 0].fillna(method='ffill').str.strip().str.upper()
        time_column = df.iloc[:, 1].astype(str).str.strip().str.lower()
        
        # Find rows matching both meal period and formatted time
        matched_rows = df[(meal_column == meal_period) & 
                         (time_column == formatted_time.lower())]
        
        if matched_rows.empty:
            return jsonify({
                "error": f"No data for {formatted_time} {meal_period}",
                "suggestion": "Check nearby times",
                "available_times": df[meal_column == meal_period].iloc[:, 1].dropna().unique().tolist()
            }), 404

        time_row_index = matched_rows.index[0]

        # Find all columns for the requested day
        day_columns = df.iloc[1, :]
        day_indices = day_columns[day_columns.str.upper() == day].index.tolist()

        if not day_indices:
            return jsonify({"error": f"No data for {day}"}), 404

        # Collect historical counts from matching day columns and time row
        counts = []
        valid_dates = []
        for col in day_indices:
            count = df.iloc[time_row_index, col]
            date = df.iloc[0, col]
            if pd.notna(count) and str(count).isnumeric():
                counts.append(int(count))
                valid_dates.append(date)

        if not counts:
            return jsonify({"error": "No historical data for this time slot"}), 404

        # Calculate metrics and determine status
        avg_count = sum(counts) / len(counts)
        status = "Slow" if avg_count < 20 else "Moderate" if avg_count < 40 else "Busy"

        return jsonify({
            "day": day,
            "time": f"{formatted_time} {meal_period}",
            "status": status,
            "average_count": round(avg_count, 1),
            "historical_weeks": len(counts),
            "min_count": min(counts),
            "max_count": max(counts),
            "analysis_dates": valid_dates
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Processing failed"}), 500

@app.route("/get_peak_hour/<day>")
def get_peak_hour(day):
    df = load_data()
    if df is None:
        return jsonify({"error": "Failed to load data"}), 500

    try:
        day = day.upper()
        day_columns = df.iloc[1, :]
        day_indices = day_columns[day_columns == day].index.tolist()
        
        if not day_indices:
            return jsonify({"error": f"No data for {day}"}), 404

        time_stats = []
        current_meal = "BREAKFAST"

        for row in range(2, df.shape[0]):
            # Track meal period
            meal_cell = df.iloc[row, 0]
            if pd.notna(meal_cell):
                current_meal = meal_cell.strip().upper()
            
            time_str = str(df.iloc[row, 1]).strip()
            if not time_str:
                continue

            # Collect counts from all matching day columns
            counts = []
            for col in day_indices:
                val = df.iloc[row, col]
                if pd.notna(val) and str(val).isnumeric():
                    counts.append(int(val))

            if not counts:
                continue  # Skip slots with no data

            avg_count = sum(counts) / len(counts)

            # Determine AM/PM based on meal period
            try:
                hour = int(time_str.split(':')[0])
                if current_meal == "BREAKFAST":
                    meridiem = "AM"
                elif current_meal == "LUNCH":
                    meridiem = "AM" if hour < 12 else "PM"
                elif current_meal == "DINNER":
                    meridiem = "PM"
                else:
                    meridiem = "PM"  # Default

                # Format time with correct AM/PM
                time_obj = datetime.strptime(f"{time_str} {meridiem}", "%I:%M %p")
                formatted_time = time_obj.strftime("%I:%M %p").lstrip('0').replace(" 0", " ", 1)

                time_stats.append({
                    "time": formatted_time,
                    "count": avg_count
                })
            except Exception as e:
                print(f"Time parse error: {str(e)}")
                continue

        if not time_stats:
            return jsonify({"error": "No time slots found"}), 404

        # Find peak hour(s)
        max_count = max(ts["count"] for ts in time_stats)
        peak_times = [ts["time"] for ts in time_stats if ts["count"] == max_count]

        return jsonify({
            "day": day,
            "peak_hours": peak_times,
            "average_peak_count": round(max_count, 1),
            "total_sampled_weeks": len(day_indices)
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Processing failed"}), 500


@app.route("/tracker")
def tracker():
    return render_template("tracker.html")

@app.route("/get_quiet_hours")
def get_quiet_hours():
    df = load_data()
    if df is None:
        return jsonify({"error": "Failed to load data"}), 500

    try:
        time_stats = {}
        current_meal = "BREAKFAST"
        
        # First pass: Create time slot map with proper AM/PM
        time_slot_map = {}
        for row in range(2, df.shape[0]):
            meal_cell = df.iloc[row, 0]
            if pd.notna(meal_cell):
                current_meal = meal_cell.strip().upper()
            
            time_str = str(df.iloc[row, 1]).strip()
            if not time_str:
                continue

            meridiem = "AM" if current_meal == "BREAKFAST" else "PM"
            
            try:
                time_obj = datetime.strptime(f"{time_str} {meridiem}", "%I:%M %p")
                formatted_time = time_obj.strftime("%I:%M %p").lstrip('0').replace(" 0", " ", 1)
                time_slot_map[row] = formatted_time
            except:
                continue

        # Second pass: Analyze counts with zero filtering
        for row, formatted_time in time_slot_map.items():
            total = 0
            count = 0
            zero_entries = 0
            
            for col in range(2, df.shape[1]):
                cell_value = df.iloc[row, col]
                if pd.notna(cell_value) and str(cell_value).isnumeric():
                    numeric_value = int(cell_value)
                    if numeric_value == 0:
                        zero_entries += 1
                    else:
                        total += numeric_value
                        count += 1

            # Only consider slots with meaningful activity
            if count > 0:  # At least some non-zero readings
                avg = total / count
                if avg > 0:  # Exclude slots with zero average
                    if formatted_time in time_stats:
                        # Merge with existing data
                        existing = time_stats[formatted_time]
                        time_stats[formatted_time] = {
                            "average": (existing["average"] * existing["count"] + total) / (existing["count"] + count),
                            "count": existing["count"] + count,
                            "zero_percentage": (zero_entries + existing.get("zero_entries", 0)) / 
                                             (count + existing["count"] + zero_entries + existing.get("zero_entries", 0)) * 100
                        }
                    else:
                        time_stats[formatted_time] = {
                            "average": avg,
                            "count": count,
                            "zero_percentage": zero_entries / (count + zero_entries) * 100
                        }

        # Get current time
        now = datetime.now()
        current_time = now.time()
        
        # Filter and sort results based on time proximity
        valid_times = []
        for k, v in time_stats.items():
            if v["average"] > 0:
                try:
                    # Parse stored time string to time object
                    slot_time = datetime.strptime(k, "%I:%M %p").time()
                    
                    # Calculate time difference in hours
                    time_diff = abs((current_time.hour * 60 + current_time.minute) - 
                                  (slot_time.hour * 60 + slot_time.minute)) / 60
                    
                    # Consider times within 1.5-2.5 hour window
                    if 1.5 <= time_diff <= 2.5:
                        valid_times.append({
                            "time": k,
                            "average": v["average"],
                            "readings": v["count"],
                            "zero_pct": round(v["zero_percentage"], 1),
                            "time_diff": time_diff
                        })
                except Exception as e:
                    print(f"Error processing time {k}: {e}")

        # Sort by closest to 2 hours difference first, then by quietest
        valid_times.sort(key=lambda x: (abs(x["time_diff"] - 2), x["average"]))
        sorted_times = valid_times[:3]

        return jsonify({
            "quiet_hours": [
                {
                    "time": t["time"],
                    "average_count": round(t["average"], 1),
                    "readings_used": t["readings"],
                    "zero_percentage": t["zero_pct"],
                    "hours_from_now": round(t["time_diff"], 1)
                } for t in sorted_times
            ],
            "current_time": now.strftime("%I:%M %p").lstrip('0')
        })

    except Exception as e:
        print(f"Error calculating quiet hours: {str(e)}")
        return jsonify({"error": "Failed to calculate quiet hours"}), 500
    
@app.route("/get_historical_trends")
def get_historical_trends():
    df = load_data()
    if df is None:
        return jsonify({"error": "Failed to load data"}), 500

    try:
        time_stats = {}
        current_meal = "BREAKFAST"
        
        # Create time slot map with proper AM/PM
        time_slot_map = {}
        for row in range(2, df.shape[0]):
            meal_cell = df.iloc[row, 0]
            if pd.notna(meal_cell):
                current_meal = meal_cell.strip().upper()
            
            time_str = str(df.iloc[row, 1]).strip()
            if not time_str:
                continue

            meridiem = "AM" if current_meal == "BREAKFAST" else "PM"
            
            try:
                time_obj = datetime.strptime(f"{time_str} {meridiem}", "%I:%M %p")
                formatted_time = time_obj.strftime("%I:%M %p").lstrip('0').replace(" 0", " ", 1)
                time_slot_map[row] = formatted_time
            except:
                continue

        # Analyze counts for all time slots
        for row, formatted_time in time_slot_map.items():
            total = 0
            count = 0
            
            for col in range(2, df.shape[1]):
                cell_value = df.iloc[row, col]
                if pd.notna(cell_value) and str(cell_value).isnumeric():
                    numeric_value = int(cell_value)
                    total += numeric_value
                    count += 1

            if count > 0:
                avg = total / count
                time_stats[formatted_time] = avg

        # Convert to sorted list
        sorted_times = sorted(
            time_stats.items(),
            key=lambda x: datetime.strptime(x[0], "%I:%M %p").time()
        )

        return jsonify({
            "labels": [t[0] for t in sorted_times],
            "data": [round(t[1], 1) for t in sorted_times]
        })

    except Exception as e:
        print(f"Error generating trends: {str(e)}")
        return jsonify({"error": "Failed to load historical data"}), 500

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        cleaned_df = prepare_data(df)
        trained_model = train_model(cleaned_df)
    else:
        print("Failed to load data for training.")

    app.run(debug=True)

