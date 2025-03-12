from flask import Flask, render_template, jsonify
import pandas as pd

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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_status")
def get_status():
    df = load_data()
    if df is None:
        return jsonify({"error": "Failed to load data"}), 500

    # Get current date, hour, and time slot
    now = pd.Timestamp.now()
    print("now: ", now)
    current_hour = now.hour
    print("current_hour: ", current_hour)
    current_day = now.strftime('%a').upper()
    print("current_day: ", current_day)

    try:
        # Ensure time is formatted correctly for comparison
        time_slot = f"{current_hour}:{now.minute // 15 * 15:02d}"
        print("time_slot: ", time_slot)

        # Match time_slot against second column (index 1)
        time_column = df.iloc[:, 1].astype(str).str.strip()  # Ensure no spaces or mismatches
        print("time_column: ", time_column)
        time_row = df[time_column == time_slot]  # Match time correctly
        print("time_row: ", time_row)

        date_columns = df.iloc[0, :]  # First row contains dates
        print("STATUS date_columns: ", date_columns)

        if not time_row.empty:
            # Find the column for the current day
            day_columns = df.iloc[1, :]  # Get day names from second row
            day_column_index = day_columns[day_columns == current_day].index[0]  # Find matching day column
            
            # Get the count for current time and day
            count = time_row.iloc[0, day_column_index]
            print("Count at this time:", count)

            # Determine status based on count
            count = int(count) if count.isnumeric() else 0  # Convert to integer safely
            if count > 50:
                current_status = "Busy"
            elif count > 20:
                current_status = "Moderate"
            else:
                current_status = "Slow"
        else:
            current_status = "Closed"

    except Exception as e:
        print("Error determining status:", e)
        current_status = "Unknown"

    return jsonify({"status": current_status})

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
