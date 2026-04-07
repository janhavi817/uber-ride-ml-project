from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import datetime

app = Flask(__name__)

# Load the trained model
model = None
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print("Error loading model:", e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validation
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        required_fields = ['pickup_datetime', 'trip_distance', 'passenger_count']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Parse inputs
        pickup_dt_str = data['pickup_datetime']
        trip_distance = float(data['trip_distance'])
        passenger_count = int(data['passenger_count'])
        
        # Input value validation
        if trip_distance <= 0:
            return jsonify({'error': 'Trip distance must be positive'}), 400
        if passenger_count <= 0:
            return jsonify({'error': 'Passenger count must be positive'}), 400
            
        # Parse datetime
        try:
            pickup_dt = datetime.datetime.strptime(pickup_dt_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # Fallback if the user just inputs datetime-local (e.g., YYYY-MM-DDTHH:MM)
            try:
                pickup_dt = datetime.datetime.strptime(pickup_dt_str, "%Y-%m-%dT%H:%M")
            except ValueError:
                return jsonify({'error': 'Invalid datetime format. Use YYYY-MM-DD HH:MM:SS or YYYY-MM-DDTHH:MM'}), 400
        
        # Extract features
        pickup_hour = pickup_dt.hour
        day_of_week = pickup_dt.weekday()
        
        # Prepare input for model
        input_features = pd.DataFrame([{
            'trip_distance': trip_distance,
            'passenger_count': passenger_count,
            'pickup_hour': pickup_hour,
            'day_of_week': day_of_week
        }])
        
        if model is None:
            return jsonify({'error': 'Model is not loaded on the server'}), 500
        
        # Predict
        predicted_duration = model.predict(input_features)[0]
        
        return jsonify({
            'predicted_duration': round(predicted_duration, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ready for AWS EC2/local (host 0.0.0.0 usually)
    app.run(host='0.0.0.0', port=5000, debug=True)
