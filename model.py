import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

def generate_task_dataset(start_date='2023-01-01', end_date='2025-01-01'):
    """Generate synthetic facility management task data."""
    np.random.seed(42)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Task types relevant to facility management
    task_types = [
        'Preventive Maintenance', 'Corrective Maintenance', 'Safety Inspection',
        'Cleaning', 'Filter Replacement', 'Equipment Calibration', 'Security Check',
        'HVAC Service', 'Plumbing Repair', 'Electrical Inspection', 'Pest Control',
        'Landscape Maintenance', 'Fire System Check', 'Elevator Maintenance'
    ]
    
    # Location/Buildings
    buildings = [f'Building_{i}' for i in range(1, 6)]
    floors = list(range(1, 11))
    
    # Team assignments
    teams = ['Maintenance A', 'Maintenance B', 'HVAC Team', 'Electrical Team', 
             'Plumbing Team', 'Security Team', 'Janitorial Team']
    
    # Priority levels
    priorities = ['Low', 'Medium', 'High', 'Critical']
    
    # Generate data
    data = []
    task_id = 1000
    
    # Workload distribution patterns (tasks per day)
    weekday_workload = {0: 40, 1: 45, 2: 50, 3: 48, 4: 42, 5: 20, 6: 15}  # Mon-Sun
    
    for date in pd.date_range(start=start_date, end=end_date, freq='D'):
        day_of_week = date.dayofweek
        
        # Seasonal factor (more maintenance in extreme weather months)
        month = date.month
        seasonal_factor = 1.0 + 0.3 * abs((month - 6.5) / 5.5)  # Higher in Jan/Dec, lower in Jul
        
        # Number of tasks for this day based on day of week and seasonal factors
        n_tasks = int(weekday_workload[day_of_week] * seasonal_factor)
        
        # Special events that might increase workload
        if (date.month == 12 and date.day > 15) or (date.month == 1 and date.day < 10):
            n_tasks = int(n_tasks * 1.2)  # Holiday season
        
        # Resource availability - weekends have fewer staff
        if day_of_week >= 5:  # Weekend
            resource_availability = np.random.uniform(0.5, 0.7)
        else:  # Weekday
            resource_availability = np.random.uniform(0.8, 1.0)
        
        # Generate tasks for this day
        for _ in range(n_tasks):
            task_id += 1
            task_type = np.random.choice(task_types)
            building = np.random.choice(buildings)
            floor = np.random.choice(floors)
            team = np.random.choice(teams)
            priority = np.random.choice(priorities, p=[0.4, 0.3, 0.2, 0.1])
            
            # Task scheduled time
            hour = np.random.randint(7, 19) if day_of_week < 5 else np.random.randint(8, 16)
            scheduled_time = datetime.combine(date.date(), datetime.min.time()) + timedelta(hours=hour)
            
            # Estimated duration (in hours)
            if task_type in ['Preventive Maintenance', 'HVAC Service', 'Elevator Maintenance']:
                duration = np.random.uniform(1.5, 4.0)
            else:
                duration = np.random.uniform(0.5, 2.0)
            
            # Task completion factors
            task_complexity = np.random.uniform(1, 10)  # 1-10 scale
            
            # Concurrent tasks for the team at this time (workload)
            concurrent_tasks = np.random.randint(0, 8)
            
            # Calculate completion probability based on various factors
            completion_prob = 0.95  # Base probability
            
            # Adjust based on factors
            if priority == 'Critical':
                completion_prob *= 1.1
            elif priority == 'Low':
                completion_prob *= 0.9
            
            # Task complexity decreases completion probability
            completion_prob *= (1 - 0.05 * task_complexity/10)
            
            # Resource availability affects completion
            completion_prob *= resource_availability
            
            # Workload (concurrent tasks) decreases completion probability
            completion_prob *= (1 - 0.03 * concurrent_tasks)
            
            # Cap at 0.99
            completion_prob = min(completion_prob, 0.99)
            
            # Determine if task was completed
            completed = np.random.random() < completion_prob
            
            # If completed, calculate actual completion time
            if completed:
                delay_hrs = np.random.exponential(1) if np.random.random() < 0.3 else 0
                completion_time = scheduled_time + timedelta(hours=delay_hrs + duration)
                completion_status = 'Completed'
            else:
                completion_time = None
                
                # Differentiate between missed and in-progress
                if date < pd.Timestamp(end_date) - pd.Timedelta(days=7):
                    completion_status = 'Missed'
                else:
                    completion_status = np.random.choice(['Missed', 'In Progress'], p=[0.7, 0.3])
            
            # Create task entry
            task = {
                'task_id': f'TASK-{task_id}',
                'task_type': task_type,
                'building': building,
                'floor': floor,
                'team': team,
                'priority': priority,
                'scheduled_time': scheduled_time,
                'estimated_duration': duration,
                'completion_time': completion_time,
                'completion_status': completion_status,
                'task_complexity': task_complexity,
                'concurrent_tasks': concurrent_tasks,
                'resource_availability': resource_availability
            }
            
            data.append(task)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add was_delayed feature
    df['was_delayed'] = False
    mask = df['completion_status'] == 'Completed'
    df.loc[mask, 'was_delayed'] = (df.loc[mask, 'completion_time'] - df.loc[mask, 'scheduled_time']).dt.total_seconds() / 3600 > df.loc[mask, 'estimated_duration']
    
    # Compute on-time completion rate
    df['on_time_completion'] = (df['completion_status'] == 'Completed') & (~df['was_delayed'])
    
    # Target variable - task missed (binary)
    df['task_missed'] = (df['completion_status'] == 'Missed').astype(int)
    
    return df

# Generate dataset
task_df = generate_task_dataset()
print(f"Generated dataset with {len(task_df)} tasks")

# Save to CSV
task_df.to_csv('facility_tasks_dataset.csv', index=False)
print("Dataset saved to 'facility_tasks_dataset.csv'")

def preprocess_task_data(df):
    """Preprocess the task data for analysis and modeling."""
    # Create copy to avoid modifying original
    df_processed = df.copy()
    
    # Handle missing values
    df_processed['completion_time'] = df_processed['completion_time'].fillna(pd.NaT)
    
    # Extract datetime features
    df_processed['scheduled_date'] = df_processed['scheduled_time'].dt.date
    df_processed['scheduled_hour'] = df_processed['scheduled_time'].dt.hour
    df_processed['scheduled_day'] = df_processed['scheduled_time'].dt.day
    df_processed['scheduled_month'] = df_processed['scheduled_time'].dt.month
    df_processed['scheduled_year'] = df_processed['scheduled_time'].dt.year
    df_processed['day_of_week'] = df_processed['scheduled_time'].dt.dayofweek
    df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6]).astype(int)
    
    # Feature engineering
    # Calculate task frequency by type (how many tasks of this type in the past week)
    task_counts = df_processed.groupby(['task_type', 'scheduled_date']).size().reset_index(name='daily_type_count')
    df_processed = pd.merge(df_processed, task_counts, on=['task_type', 'scheduled_date'], how='left')
    
    # Calculate workload by team per day
    team_workload = df_processed.groupby(['team', 'scheduled_date']).size().reset_index(name='team_daily_workload')
    df_processed = pd.merge(df_processed, team_workload, on=['team', 'scheduled_date'], how='left')
    
    # Calculate building workload per day
    building_workload = df_processed.groupby(['building', 'scheduled_date']).size().reset_index(name='building_daily_workload')
    df_processed = pd.merge(df_processed, building_workload, on=['building', 'scheduled_date'], how='left')
    
    # Calculate rolling 7-day miss rate for each team
    team_miss_history = df_processed.groupby(['team', 'scheduled_date'])['task_missed'].mean().reset_index()
    team_miss_history = team_miss_history.sort_values(['team', 'scheduled_date'])
    team_miss_history['rolling_miss_rate'] = team_miss_history.groupby('team')['task_missed'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean())
    df_processed = pd.merge(df_processed, team_miss_history[['team', 'scheduled_date', 'rolling_miss_rate']], 
                           on=['team', 'scheduled_date'], how='left')
    
    # Priority encoding
    priority_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    df_processed['priority_level'] = df_processed['priority'].map(priority_map)
    
    # Time since last maintenance for building-floor combination
    df_processed['building_floor'] = df_processed['building'] + '-' + df_processed['floor'].astype(str)
    
    # Compute maintenance intervals for each building-floor
    maintenance_history = df_processed[df_processed['completion_status'] == 'Completed'].copy()
    maintenance_history = maintenance_history.sort_values(['building_floor', 'completion_time'])
    maintenance_history['prev_completion'] = maintenance_history.groupby('building_floor')['completion_time'].shift(1)
    maintenance_history['days_since_last_maintenance'] = (
        maintenance_history['completion_time'] - maintenance_history['prev_completion']).dt.total_seconds() / (24 * 3600)
    
    # Merge back with limited fields
    days_since_maint = maintenance_history[['task_id', 'days_since_last_maintenance']].dropna()
    df_processed = pd.merge(df_processed, days_since_maint, on='task_id', how='left')
    df_processed['days_since_last_maintenance'] = df_processed['days_since_last_maintenance'].fillna(30)  # Default to 30 days
    
    return df_processed

# Process the data
processed_df = preprocess_task_data(task_df)
print("Data preprocessing complete")

# Feature Selection for Modeling
def prepare_modeling_data(df):
    """Prepare data for predictive modeling."""
    # Select relevant numerical features
    numerical_features = [
        'estimated_duration', 'task_complexity', 'concurrent_tasks', 
        'resource_availability', 'scheduled_hour', 'day_of_week', 
        'is_weekend', 'scheduled_month', 'daily_type_count',
        'team_daily_workload', 'building_daily_workload', 
        'rolling_miss_rate', 'priority_level', 'days_since_last_maintenance'
    ]
    
    # Categorical features to encode
    categorical_features = ['task_type', 'building', 'floor', 'team', 'priority']
    
    # Select data for completed or missed tasks (exclude in-progress)
    modeling_df = df[df['completion_status'] != 'In Progress'].copy()
    
    # Define X and y
    X_numerical = modeling_df[numerical_features]
    X_categorical = modeling_df[categorical_features]
    y = modeling_df['task_missed']
    
    return X_numerical, X_categorical, y, numerical_features, categorical_features

# Prepare modeling data
X_numerical, X_categorical, y, numerical_features, categorical_features = prepare_modeling_data(processed_df)
print(f"Prepared modeling data with numerical features: {X_numerical.shape[1]}, categorical features: {X_categorical.shape[1]}")

# First, let's encode categorical features manually before splitting
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_categorical_encoded = encoder.fit_transform(X_categorical)

# Get feature names after encoding
categorical_feature_names = []
for i, feature in enumerate(categorical_features):
    for category in encoder.categories_[i]:
        categorical_feature_names.append(f"{feature}_{category}")

# Combine numerical and encoded categorical features
X_combined = np.hstack([X_numerical.values, X_categorical_encoded])
combined_feature_names = numerical_features + categorical_feature_names
print(f"Combined data shape: {X_combined.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

# Check class imbalance
print("\nClass distribution in training set:")
print(pd.Series(y_train).value_counts(normalize=True) * 100)

# Now apply SMOTE to the encoded data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts(normalize=True) * 100)

# Scale the numerical features for models that need it
scaler = StandardScaler()
# Create a copy to avoid modifying the original
X_train_scaled = X_train_resampled.copy()
X_test_scaled = X_test.copy()

# Scale only the numerical features (first len(numerical_features) columns)
X_train_scaled[:, :len(numerical_features)] = scaler.fit_transform(X_train_resampled[:, :len(numerical_features)])
X_test_scaled[:, :len(numerical_features)] = scaler.transform(X_test[:, :len(numerical_features)])

def train_and_evaluate_models(X_train, y_train, X_test, y_test, feature_names):
    """Train multiple models and evaluate performance."""
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, 
                                  scale_pos_weight=sum(y_train==0)/sum(y_train==1))
    }
    
    # Train and evaluate each model
    results = {}
    best_auc = 0
    best_model = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = (y_pred == y_test).mean()
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Save feature importances if applicable
        if hasattr(model, 'feature_importances_'):
            feature_importances = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            print("\nTop 15 features by importance:")
            print(feature_importances)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'y_pred_proba': y_pred_proba
        }
        
        if auc > best_auc:
            best_auc = auc
            best_model = model
    
    return results, best_model

# Train models with scaled data
model_results, best_model = train_and_evaluate_models(X_train_scaled, y_train_resampled, 
                                                    X_test_scaled, y_test, combined_feature_names)
print(f"\nModel training complete. Best model selected.")

# Save best model
model_data = {
    'model': best_model,
    'scaler': scaler,
    'encoder': encoder,
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
    'combined_feature_names': combined_feature_names
}
joblib.dump(model_data, 'facility_task_prediction_model.pkl')
print("Best model and preprocessing components saved as 'facility_task_prediction_model.pkl'")

# SHAP Analysis for explainability
def generate_shap_analysis(model, X_train, X_test, feature_names):
    """Generate SHAP values for model explainability."""
    # Check if model supports SHAP explainability
    if hasattr(model, 'predict_proba'):
        try:
            # For tree-based models
            if hasattr(model, 'feature_importances_'):
                explainer = shap.TreeExplainer(model)
            else:
                # For other models
                explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
            
            # Calculate SHAP values (limit to 100 samples for efficiency)
            shap_values = explainer.shap_values(X_test[:100])
            
            # If shap_values is a list (for multi-class), take the values for class 1
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
                
            # Create a DataFrame with SHAP values
            shap_df = pd.DataFrame(shap_values, columns=feature_names)
            
            # Get the mean absolute SHAP value for each feature
            mean_shap = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False)
            
            print("\nTop 15 features by SHAP importance:")
            print(mean_shap.head(15))
            
            return explainer, shap_values, mean_shap
        except Exception as e:
            print(f"Error generating SHAP values: {e}")
            return None, None, None
    else:
        print("Model does not support probability predictions for SHAP analysis")
        return None, None, None

# Generate SHAP analysis
explainer, shap_values, mean_shap = generate_shap_analysis(
    best_model, X_train_scaled[:500], X_test_scaled[:100], combined_feature_names)

if shap_values is not None:
    print("SHAP analysis generated successfully")
    # Save SHAP values for dashboard
    shap_data = {
        'values': shap_values,
        'feature_names': combined_feature_names,
        'mean_importance': mean_shap
    }
    joblib.dump(shap_data, 'task_shap_values.pkl')

# Time Series Forecasting
def build_forecasting_model(df):
    """Build a Prophet forecasting model for task completion rates."""
    # Prepare data for Prophet
    # Aggregate daily task completion rates
    daily_rates = df.groupby('scheduled_date')['task_missed'].mean().reset_index()
    daily_rates['miss_rate'] = daily_rates['task_missed'] * 100  # Convert to percentage
    
    # Format for Prophet
    prophet_df = daily_rates.rename(columns={'scheduled_date': 'ds', 'miss_rate': 'y'})
    
    # Train Prophet model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    
    # Generate future dataframe
    future = model.make_future_dataframe(periods=30)  # Forecast 30 days ahead
    
    # Make predictions
    forecast = model.predict(future)
    
    return model, forecast

# Build forecasting model
forecast_model, forecast_results = build_forecasting_model(processed_df)
print("Time series forecasting model built")

import pickle

with open('task_forecast_model.pkl', 'wb') as f:
    pickle.dump(forecast_model, f)

# Anomaly Detection
def detect_task_miss_anomalies(df):
    """Detect anomalies in task miss patterns."""
    # Prepare data
    # Get daily miss rates
    daily_miss = df.groupby('scheduled_date')['task_missed'].mean().reset_index()
    
    # Train isolation forest
    model = IsolationForest(contamination=0.05, random_state=42)
    daily_miss['anomaly'] = model.fit_predict(daily_miss[['task_missed']])
    daily_miss['anomaly'] = (daily_miss['anomaly'] == -1).astype(int)  # Convert to binary (1 = anomaly)
    
    # Add anomaly score
    daily_miss['anomaly_score'] = model.score_samples(daily_miss[['task_missed']])
    
    return daily_miss

# Detect anomalies
anomaly_results = detect_task_miss_anomalies(processed_df)
anomaly_results.to_csv('task_anomalies.csv', index=False)
print("Anomaly detection complete")

# Generate aggregated data for dashboard
def prepare_dashboard_data(df):
    """Prepare aggregated data for the dashboard."""
    # Daily aggregates
    daily_stats = df.groupby('scheduled_date').agg({
        'task_missed': 'mean',
        'task_id': 'count',
        'on_time_completion': 'mean'
    }).reset_index()
    daily_stats.columns = ['date', 'miss_rate', 'task_count', 'on_time_rate']
    daily_stats['miss_rate'] = daily_stats['miss_rate'] * 100
    daily_stats['on_time_rate'] = daily_stats['on_time_rate'] * 100
    
    # Weekly aggregates
    df['scheduled_week'] = df['scheduled_time'].dt.isocalendar().week
    df['scheduled_year_week'] = df['scheduled_time'].dt.strftime('%Y-%U')
    weekly_stats = df.groupby('scheduled_year_week').agg({
        'task_missed': 'mean',
        'task_id': 'count',
        'on_time_completion': 'mean'
    }).reset_index()
    weekly_stats.columns = ['year_week', 'miss_rate', 'task_count', 'on_time_rate']
    weekly_stats['miss_rate'] = weekly_stats['miss_rate'] * 100
    weekly_stats['on_time_rate'] = weekly_stats['on_time_rate'] * 100
    
    # Team performance
    team_stats = df.groupby('team').agg({
        'task_missed': 'mean',
        'task_id': 'count',
        'on_time_completion': 'mean'
    }).reset_index()
    team_stats.columns = ['team', 'miss_rate', 'task_count', 'on_time_rate']
    team_stats['miss_rate'] = team_stats['miss_rate'] * 100
    team_stats['on_time_rate'] = team_stats['on_time_rate'] * 100
    
    # Task type performance
    task_type_stats = df.groupby('task_type').agg({
        'task_missed': 'mean',
        'task_id': 'count',
        'on_time_completion': 'mean'
    }).reset_index()
    task_type_stats.columns = ['task_type', 'miss_rate', 'task_count', 'on_time_rate']
    task_type_stats['miss_rate'] = task_type_stats['miss_rate'] * 100
    task_type_stats['on_time_rate'] = task_type_stats['on_time_rate'] * 100
    
    # Building performance
    building_stats = df.groupby('building').agg({
        'task_missed': 'mean',
        'task_id': 'count',
        'on_time_completion': 'mean'
    }).reset_index()
    building_stats.columns = ['building', 'miss_rate', 'task_count', 'on_time_rate']
    building_stats['miss_rate'] = building_stats['miss_rate'] * 100
    building_stats['on_time_rate'] = building_stats['on_time_rate'] * 100
    
    # Heatmap data - create hourly/daily task miss rates
    heatmap_data = df.groupby(['day_of_week', 'scheduled_hour'])['task_missed'].mean().reset_index()
    heatmap_data['miss_rate'] = heatmap_data['task_missed'] * 100
    
    # Monthly heatmap
    monthly_heatmap = df.groupby(['scheduled_month', 'day_of_week'])['task_missed'].mean().reset_index()
    monthly_heatmap['miss_rate'] = monthly_heatmap['task_missed'] * 100
    
    # Save aggregated data
    daily_stats.to_csv('dashboard_daily_stats.csv', index=False)
    weekly_stats.to_csv('dashboard_weekly_stats.csv', index=False)
    team_stats.to_csv('dashboard_team_stats.csv', index=False)
    task_type_stats.to_csv('dashboard_task_type_stats.csv', index=False)
    building_stats.to_csv('dashboard_building_stats.csv', index=False)
    heatmap_data.to_csv('dashboard_hourly_heatmap.csv', index=False)
    monthly_heatmap.to_csv('dashboard_monthly_heatmap.csv', index=False)
    
    print("Dashboard data prepared and saved")
    return {
        'daily_stats': daily_stats,
        'weekly_stats': weekly_stats,
        'team_stats': team_stats,
        'task_type_stats': task_type_stats,
        'building_stats': building_stats,
        'heatmap_data': heatmap_data,
        'monthly_heatmap': monthly_heatmap
    }

# Prepare dashboard data
dashboard_data = prepare_dashboard_data(processed_df)
print("All data processing and model training complete!")

# Get a small sample of original data for loading into the dashboard
sample_tasks = task_df.sample(min(1000, len(task_df))).to_csv('sample_tasks.csv', index=False)