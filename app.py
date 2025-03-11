from flask import Flask, render_template, jsonify
import pandas as pd

app = Flask(__name__)

# Load data from the Excel file
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\mahar\dining_hall_tracker\data\graze_data.csv")  # Change to CSV
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
    
    # Example: Extract relevant data from the DataFrame
    current_status = "Moderate"  # Replace with logic to determine the current status
    return jsonify({"status": current_status})

@app.route("/get_busiest_hour")
def get_busiest_hour():
    df = load_data()
    if df is None:
        return jsonify({"error": "Failed to load data"}), 500

    busiest_hour = "12 PM"  # Replace with logic to get the actual busiest hour
    return jsonify({"busiest_hour": busiest_hour})

@app.route("/get_quiet_hours")
def get_quiet_hours():
    df = load_data()
    if df is None:
        return jsonify({"error": "Failed to load data"}), 500

    quiet_hours = ["9 AM", "3 PM"]  # Replace with logic to get quiet hours
    return jsonify({"quiet_hours": quiet_hours})

if __name__ == "__main__":
    app.run(debug=True)