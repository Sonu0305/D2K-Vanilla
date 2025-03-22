import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

def find_best_team_for_task(
    task_type,
    building,
    floor,
    priority,
    scheduled_time,
    estimated_duration,
    task_complexity,
    concurrent_tasks,
    resource_availability,
    model_path=r'C:\Users\Kevin Shah\OneDrive\Desktop\D2K\output\kaggle\working\facility_task_prediction_model.pkl',
    teams=None
):
    """
    Find the best team for a given task based on the lowest predicted miss rate.
    
    Parameters:
    -----------
    task_type : str
        Type of maintenance task
    building : str
        Building identifier
    floor : int
        Floor number
    priority : str
        Task priority ('Low', 'Medium', 'High', 'Critical')
    scheduled_time : datetime
        When the task is scheduled
    estimated_duration : float
        Estimated hours to complete the task
    task_complexity : int
        Complexity rating (1-5)
    concurrent_tasks : int
        Number of other tasks occurring at the same time
    resource_availability : float
        Resource availability as a percentage (0.0-1.0)
    model_path : str
        Path to the trained model file
    teams : list, optional
        List of teams to evaluate. If None, uses default teams
        
    Returns:
    --------
    tuple
        (best_team_name, results_dataframe)
    """
    # Validate inputs
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    if not isinstance(scheduled_time, datetime):
        raise ValueError("scheduled_time must be a datetime object")
    
    if resource_availability < 0 or resource_availability > 1:
        raise ValueError("resource_availability must be between 0 and 1")
    
    # Default teams if none provided
    if teams is None:
        teams = [
            'Maintenance A', 'Maintenance B', 'HVAC Team', 
            'Electrical Team', 'Plumbing Team', 'Security Team', 'Janitorial Team'
        ]
    
    try:
        # Load model data
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        encoder = model_data['encoder']
        numerical_features = model_data['numerical_features']
        categorical_features = model_data['categorical_features']
        
        # Store results
        results = []
        
        # Predict for each team
        for team in teams:
            miss_rate = predict_task_miss_rate(
                team, task_type, building, floor, priority,
                scheduled_time, estimated_duration, task_complexity,
                concurrent_tasks, resource_availability,
                model, scaler, encoder, numerical_features, categorical_features
            )
            
            results.append({
                'Team': team,
                'Task Miss Rate': miss_rate
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results).sort_values('Task Miss Rate')
        
        # Find best team
        best_team = results_df.iloc[0]
        
        return best_team['Team'], results_df
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

def predict_task_miss_rate(
    team_name, task_type, building, floor, priority, scheduled_time,
    estimated_duration, task_complexity, concurrent_tasks, resource_availability,
    model, scaler, encoder, numerical_features, categorical_features
):
    """Predict miss rate for a specific team and task"""
    # Prepare input data
    input_df = prepare_input_data(
        team_name, task_type, building, floor, priority,
        scheduled_time, estimated_duration, task_complexity,
        concurrent_tasks, resource_availability
    )
    
    # Select features
    X_numerical = input_df[numerical_features]
    X_categorical = input_df[categorical_features]
    
    # Encode and scale
    X_categorical_encoded = encoder.transform(X_categorical)
    X_combined = np.hstack([X_numerical.values, X_categorical_encoded])
    X_combined_scaled = X_combined.copy()
    X_combined_scaled[:, :len(numerical_features)] = scaler.transform(X_combined[:, :len(numerical_features)])
    
    # Predict
    y_pred_proba = model.predict_proba(X_combined_scaled)[:, 1]
    
    return y_pred_proba[0]

def prepare_input_data(
    team_name, task_type, building, floor, priority, scheduled_time,
    estimated_duration, task_complexity, concurrent_tasks, resource_availability
):
    """Prepare and feature engineer input data"""
    # Create input dataframe
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
    
    input_df = pd.DataFrame([input_data])
    
    # Extract datetime features
    input_df['scheduled_date'] = input_df['scheduled_time'].dt.date
    input_df['scheduled_hour'] = input_df['scheduled_time'].dt.hour
    input_df['scheduled_day'] = input_df['scheduled_time'].dt.day
    input_df['scheduled_month'] = input_df['scheduled_time'].dt.month
    input_df['scheduled_year'] = input_df['scheduled_time'].dt.year
    input_df['day_of_week'] = input_df['scheduled_time'].dt.dayofweek
    input_df['is_weekend'] = input_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Add derived features
    input_df['daily_type_count'] = 1  # Simplified placeholder
    input_df['team_daily_workload'] = 1  # Simplified placeholder
    input_df['building_daily_workload'] = 1  # Simplified placeholder
    input_df['rolling_miss_rate'] = 0.1  # Simplified placeholder
    
    # Encode priority
    priority_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    input_df['priority_level'] = input_df['priority'].map(priority_map)
    
    # Add maintenance history feature
    input_df['days_since_last_maintenance'] = 30  # Simplified placeholder
    
    return input_df

def visualize_team_comparison(results_df):
    """Create a visual comparison of team miss rates"""
    plt.figure(figsize=(10, 6))
    
    # Sort by miss rate for better visualization
    sorted_df = results_df.sort_values('Task Miss Rate')
    
    # Create bar chart
    bars = plt.bar(
        sorted_df['Team'], 
        sorted_df['Task Miss Rate'] * 100,  # Convert to percentage
        color=['green' if i == 0 else 'skyblue' for i in range(len(sorted_df))]
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.5,
            f'{height:.1f}%',
            ha='center', va='bottom'
        )
    
    plt.title('Task Miss Rate by Team')
    plt.xlabel('Team')
    plt.ylabel('Miss Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return plt

# Example usage
if __name__ == "__main__":
    # Define task parameters
    task_details = {
        'task_type': 'Preventive Maintenance',
        'building': 'Building_4',
        'floor': 9,
        'priority': 'Low',
        'scheduled_time': datetime.now() + timedelta(days=1),
        'estimated_duration': 2.0,
        'task_complexity': 2,
        'concurrent_tasks': 5,
        'resource_availability': 0.9
    }
    
    # Find best team
    try:
        best_team, results = find_best_team_for_task(**task_details)
        
        # Print results
        print("Task Miss Rate for All Teams:")
        print(results)
        print(f"\nBest Performing Team: {best_team} with a Task Miss Rate of {results.iloc[0]['Task Miss Rate']:.2%}")
        
        # Create and show visualization
        plt_figure = visualize_team_comparison(results)
        plt_figure.show()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")