# Uber Ride Duration Predictor

An end-to-end Machine Learning web application to predict Uber ride durations.

## Project Structure
- `app/`: Flask application containing the backend (`app.py`), the saved ML model (`model.pkl`), and the frontend files (`templates/`, `static/`).
- `data/`: Contains the original dataset (`uber_data.csv`).
- `notebooks/`: Contains the ML training script (`train_model.py`).

## Machine Learning
1. Navigate to the project root directory.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the training script: `python notebooks/train_model.py`
This process will handle missing values, engineer features, train a baseline Linear Regression model and a final Random Forest model, and finally save it to `app/model.pkl`.

## Web Application (Local / AWS Ready)
1. Navigate to the `app` directory: `cd app`
2. Start the server: `python app.py`
3. Open your browser and go to `http://localhost:5000`

### API Endpoint (`/predict`)
A POST endpoint for predicting duration via API.

**Request:**
```json
{
  "pickup_datetime": "2024-01-01 10:00:00",
  "trip_distance": 3.5,
  "passenger_count": 2
}
```

**Response:**
```json
{
  "predicted_duration": 18.24
}
```
