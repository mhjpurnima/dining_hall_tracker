from flask import Flask, render_template, jsonify
import pandas as pd

app = Flask(__name__)

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
        time_row = df[time_column == time_slot]  # Match time correctly
        print("time_row: ", time_row)

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
        day_column_index = day_columns[day_columns == current_day].index[0]  # Find matching day column

        # Filter rows for the current date
        print("df: ", df.iloc[:, 3])
        date_mask = df.iloc[:, 0].astype(str) == current_date
        filtered_df = df[date_mask]
        

        if filtered_df.empty:
            return jsonify({"peak_hour": "No data available"})

        # Get time slots and counts
        time_counts = filtered_df.iloc[:, [1, day_column_index]].copy()
        time_counts.columns = ["time", "count"]

        # Convert count to numeric
        time_counts["count"] = pd.to_numeric(time_counts["count"], errors="coerce").fillna(0).astype(int)

        # Find the max count
        max_count = time_counts["count"].max()
        if max_count <= 0:
            return jsonify({"peak_hour": "No significant data"})

        # Get peak time(s)
        peak_rows = time_counts[time_counts["count"] == max_count]
        peak_time = peak_rows.iloc[0]["time"]  # Take the first peak time

        # Find adjacent time (previous or next row)
        peak_index = peak_rows.index[0]
        adjacent_time = None

        if peak_index > time_counts.index[0]:  # Check previous row
            adjacent_time = time_counts.loc[peak_index - 1, "time"]
        elif peak_index < time_counts.index[-1]:  # Check next row
            adjacent_time = time_counts.loc[peak_index + 1, "time"]

        # Format peak and adjacent times
        formatted_peak_time = pd.to_datetime(peak_time, format="%H:%M").strftime("%-I %p")
        formatted_adjacent_time = (
            pd.to_datetime(adjacent_time, format="%H:%M").strftime("%-I %p") if adjacent_time else None
        )

        # Combine result
        if formatted_adjacent_time:
            peak_hour = f"{formatted_peak_time} & {formatted_adjacent_time}"
        else:
            peak_hour = formatted_peak_time

        print("Peak Hour:", peak_hour)
        return jsonify({"peak_hour": peak_hour})

    except Exception as e:
        print("Error finding peak hour:", e)
        return jsonify({"peak_hour": "Unknown"})

# @app.route("/get_quiet_hours")
# def get_quiet_hours():
#     df = load_data()
#     if df is None:
#         return jsonify({"error": "Failed to load data"}), 500

if __name__ == "__main__":
    app.run(debug=True)
