import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Load the trained model and preprocessing components
model_data = joblib.load(r'C:\Users\admin\Desktop\D2K-Vanilla\f\output\kaggle\working\facility_task_prediction_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
encoder = model_data['encoder']
numerical_features = model_data['numerical_features']
categorical_features = model_data['categorical_features']
combined_feature_names = model_data['combined_feature_names']

# Function to prepare input data for prediction
def prepare_input_data(team_name, task_type, building, floor, priority, scheduled_time, estimated_duration, task_complexity, concurrent_tasks, resource_availability):
    # Create a dictionary with the input data
    input_data = {
        'task_type': task_type,
        'building': building,
        'floor': floor,
        'team': team_name,
        'priority': priority,
        'scheduled_time': scheduled_time,
        'estimated_duration': estimated_duration,
        'task_complexity': task_complexity,
        'concurrent_tasks': concurrent_tasks,
        'resource_availability': resource_availability
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Extract datetime features
    input_df['scheduled_date'] = input_df['scheduled_time'].dt.date
    input_df['scheduled_hour'] = input_df['scheduled_time'].dt.hour
    input_df['scheduled_day'] = input_df['scheduled_time'].dt.day
    input_df['scheduled_month'] = input_df['scheduled_time'].dt.month
    input_df['scheduled_year'] = input_df['scheduled_time'].dt.year
    input_df['day_of_week'] = input_df['scheduled_time'].dt.dayofweek
    input_df['is_weekend'] = input_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Feature engineering (similar to preprocessing)
    # Calculate task frequency by type (how many tasks of this type in the past week)
    # For simplicity, we assume no past tasks in this example
    input_df['daily_type_count'] = 1
    
    # Calculate workload by team per day
    # For simplicity, we assume no other tasks for the team on this day
    input_df['team_daily_workload'] = 1
    
    # Calculate building workload per day
    # For simplicity, we assume no other tasks for the building on this day
    input_df['building_daily_workload'] = 1
    
    # Calculate rolling 7-day miss rate for each team
    # For simplicity, we assume a rolling miss rate of 0.1
    input_df['rolling_miss_rate'] = 0.1
    
    # Priority encoding
    priority_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    input_df['priority_level'] = input_df['priority'].map(priority_map)
    
    # Time since last maintenance for building-floor combination
    # For simplicity, we assume 30 days since last maintenance
    input_df['days_since_last_maintenance'] = 30
    
    return input_df

# Function to predict task miss rate for a team
def predict_task_miss_rate(team_name, task_type, building, floor, priority, scheduled_time, estimated_duration, task_complexity, concurrent_tasks, resource_availability):
    # Prepare input data
    input_df = prepare_input_data(team_name, task_type, building, floor, priority, scheduled_time, estimated_duration, task_complexity, concurrent_tasks, resource_availability)
    
    # Select numerical and categorical features
    X_numerical = input_df[numerical_features]
    X_categorical = input_df[categorical_features]
    
    # Encode categorical features
    X_categorical_encoded = encoder.transform(X_categorical)
    
    # Combine numerical and encoded categorical features
    X_combined = np.hstack([X_numerical.values, X_categorical_encoded])
    
    # Scale the numerical features
    X_combined_scaled = X_combined.copy()
    X_combined_scaled[:, :len(numerical_features)] = scaler.transform(X_combined[:, :len(numerical_features)])
    
    # Predict the probability of task miss
    y_pred_proba = model.predict_proba(X_combined_scaled)[:, 1]
    
    return y_pred_proba[0]

# Define the list of teams to compare
teams = ['Maintenance A', 'Maintenance B', 'HVAC Team', 'Electrical Team', 'Plumbing Team', 'Security Team', 'Janitorial Team']

# Define common task details for all teams
task_type = 'Preventive Maintenance'
building = 'Building_4'
floor = 9
priority = 'Low'
scheduled_time = datetime.now() + timedelta(days=1)
estimated_duration = 2.0
task_complexity = 2
concurrent_tasks = 5
resource_availability = 0.9

# Store results for each team
results = []

# Loop through each team and predict the task miss rate
for team in teams:
    miss_rate = predict_task_miss_rate(team, task_type, building, floor, priority, scheduled_time, estimated_duration, task_complexity, concurrent_tasks, resource_availability)
    results.append({
        'Team': team,
        'Task Miss Rate': miss_rate
    })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)

# Print results for all teams
print("Task Miss Rate for All Teams:")
print(results_df)

# Determine the best-performing team (lowest task miss rate)
best_team = results_df.loc[results_df['Task Miss Rate'].idxmin()]
print(f"\nBest Performing Team: {best_team['Team']} with a Task Miss Rate of {best_team['Task Miss Rate']:.2%}")
