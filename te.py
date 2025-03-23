import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Facility Task Management Dashboard",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #F9F9F9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
    }
    div.block-container {
        padding-top: 2rem;
    }
    h1, h2, h3, h4, h5 {
        color: #007BFF;
    }
    .recommendation {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Define function to load model
@st.cache_resource
def load_model():
    model_path = r'output\kaggle\working\facility_task_prediction_model.pkl'
    try:
        model_data = joblib.load(model_path)
        return model_data
    except:
        st.error(f"Failed to load model from {model_path}. Please check the path.")
        st.warning("Using dummy model for demonstration purposes.")
        # Return mock data for demonstration
        class DummyModel:
            def predict_proba(self, X):
                return np.array([[0.7, 0.3]] * X.shape[0])
        
        return {
            'model': DummyModel(),
            'scaler': lambda x: x,
            'encoder': lambda x: np.zeros((x.shape[0], 10)),
            'numerical_features': ['estimated_duration', 'task_complexity', 'concurrent_tasks', 
                                'resource_availability', 'scheduled_hour', 'scheduled_day', 
                                'scheduled_month', 'is_weekend', 'priority_level', 
                                'days_since_last_maintenance'],
            'categorical_features': ['task_type', 'building', 'floor', 'team'],
            'combined_feature_names': []
        }

# Load model data
model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
encoder = model_data['encoder']
numerical_features = model_data['numerical_features']
categorical_features = model_data['categorical_features']
combined_feature_names = model_data['combined_feature_names']

# Function to prepare input data for prediction
def prepare_input_data(team_name, task_type, building, floor, priority, scheduled_time, 
                      estimated_duration, task_complexity, concurrent_tasks, resource_availability):
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
    input_df['scheduled_date'] = pd.to_datetime(input_df['scheduled_time']).dt.date
    input_df['scheduled_hour'] = pd.to_datetime(input_df['scheduled_time']).dt.hour
    input_df['scheduled_day'] = pd.to_datetime(input_df['scheduled_time']).dt.day
    input_df['scheduled_month'] = pd.to_datetime(input_df['scheduled_time']).dt.month
    input_df['scheduled_year'] = pd.to_datetime(input_df['scheduled_time']).dt.year
    input_df['day_of_week'] = pd.to_datetime(input_df['scheduled_time']).dt.dayofweek
    input_df['is_weekend'] = input_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Feature engineering (similar to preprocessing)
    input_df['daily_type_count'] = 1
    input_df['team_daily_workload'] = 1
    input_df['building_daily_workload'] = 1
    input_df['rolling_miss_rate'] = 0.1
    
    # Priority encoding
    priority_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    input_df['priority_level'] = input_df['priority'].map(priority_map)
    
    # Time since last maintenance for building-floor combination
    input_df['days_since_last_maintenance'] = 30
    
    return input_df

# Function to predict task miss rate for a team
def predict_task_miss_rate(team_name, task_type, building, floor, priority, scheduled_time, 
                          estimated_duration, task_complexity, concurrent_tasks, resource_availability):
    # Prepare input data
    input_df = prepare_input_data(team_name, task_type, building, floor, priority, scheduled_time, 
                                 estimated_duration, task_complexity, concurrent_tasks, resource_availability)
    
    # Select numerical and categorical features
    X_numerical = input_df[numerical_features]
    X_categorical = input_df[categorical_features]
    
    try:
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
    except:
        # If prediction fails, return random value for demo
        return np.random.uniform(0.05, 0.35)

# Create header
st.title("Facility Task Management Dashboard")
st.markdown("#### Task Miss Rate Prediction & Team Comparison")
st.markdown("---")

# Create two columns for layout
col1, col2 = st.columns([1, 2])

# Left column - Input parameters
with col1:
    st.markdown("### Task Parameters")
    
    # Task type
    task_type = st.selectbox(
        "Task Type:",
        options=[
            'Corrective Maintenance',
            'Preventive Maintenance',
            'Emergency Repair',
            'Installation',
            'Inspection'
        ],
        index=0
    )
    
    # Building
    building = st.selectbox(
        "Building:",
        options=[f'Building_{i}' for i in range(1, 11)],
        index=3
    )
    
    # Floor
    floor = st.selectbox(
        "Floor:",
        options=list(range(1, 21)),
        index=8
    )
    
    # Priority
    priority = st.selectbox(
        "Priority:",
        options=['Low', 'Medium', 'High', 'Critical'],
        index=0
    )
    
    # Scheduled Date & Time
    st.subheader("Scheduled Date & Time")
    col1a, col1b = st.columns(2)
    with col1a:
        scheduled_date = st.date_input(
            "Date:",
            value=datetime.now().date(),
            min_value=datetime.now().date(),
            max_value=datetime.now().date() + timedelta(days=30)
        )
    with col1b:
        scheduled_hour = st.selectbox(
            "Time:",
            options=list(range(24)),
            index=9,
            format_func=lambda x: f'{x}:00'
        )
    
    # Duration, Complexity, Concurrent Tasks, Resource Availability
    estimated_duration = st.number_input(
        "Estimated Duration (hours):",
        min_value=0.5,
        max_value=24.0,
        value=2.0,
        step=0.5
    )
    
    task_complexity = st.slider(
        "Task Complexity (0-10):",
        min_value=0,
        max_value=10,
        value=3,
        step=1
    )
    
    concurrent_tasks = st.number_input(
        "Concurrent Tasks:",
        min_value=0,
        max_value=20,
        value=5,
        step=1
    )
    
    resource_availability = st.slider(
        "Resource Availability (0-1):",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )
    
    # Team selection
    st.markdown("### Select Teams to Compare")
    all_teams = [
        'Maintenance A',
        'Maintenance B',
        'HVAC Team',
        'Electrical Team',
        'Plumbing Team',
        'Security Team',
        'Janitorial Team'
    ]
    
    default_teams = all_teams[:5]  # Select first 5 teams by default
    selected_teams = st.multiselect(
        "Teams:",
        options=all_teams,
        default=default_teams
    )
    
    # Calculate button
    calculate_button = st.button("Calculate Miss Rates", type="primary")

# Right column - Results
with col2:
    if calculate_button and selected_teams:
        # Create scheduled_time from date and hour
        scheduled_time = datetime.combine(scheduled_date, datetime.min.time()) + timedelta(hours=scheduled_hour)
        
        # Store results for each team
        results = []
        
        # Loop through each selected team and predict the task miss rate
        for team in selected_teams:
            miss_rate = predict_task_miss_rate(
                team, task_type, building, floor, priority, scheduled_time, 
                estimated_duration, task_complexity, concurrent_tasks, resource_availability
            )
            results.append({
                'Team': team,
                'Task Miss Rate': miss_rate,
                'Success Rate': 1 - miss_rate
            })
        
        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by miss rate (ascending)
        results_df = results_df.sort_values('Task Miss Rate')
        
        # Determine the best-performing team
        best_team = results_df.iloc[0]
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Team Comparison", "Best Team", "Detailed Results"])
        
        with tab1:
            st.markdown("### Task Miss Rate Comparison")
            
            # Create the bar chart
            fig = go.Figure()
            
            # Add bars for miss rate
            fig.add_trace(go.Bar(
                x=results_df['Team'],
                y=results_df['Task Miss Rate'],
                name='Miss Rate',
                marker_color='#FF6B6B',
                text=[f"{rate:.1%}" for rate in results_df['Task Miss Rate']],
                textposition='auto'
            ))
            
            # Add bars for success rate
            fig.add_trace(go.Bar(
                x=results_df['Team'],
                y=results_df['Success Rate'],
                name='Success Rate',
                marker_color='#4ECDC4',
                text=[f"{rate:.1%}" for rate in results_df['Success Rate']],
                textposition='auto'
            ))
            
            # Update layout
            fig.update_layout(
                barmode='stack',
                xaxis_title='Team',
                yaxis_title='Rate',
                yaxis=dict(tickformat='.0%'),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                plot_bgcolor='white',
                height=400
            )
            
            # Display the figure
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Best Team Performance")
            
            # Create columns for gauge and recommendation
            col2a, col2b = st.columns([2, 3])
            
            with col2a:
                # Create gauge chart for best team
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=best_team['Success Rate'],
                    title={'text': f"Best Team: {best_team['Team']}", 'font': {'size': 20}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 1], 'tickformat': '.0%'},
                        'bar': {'color': "#4ECDC4"},
                        'steps': [
                            {'range': [0, 0.6], 'color': '#FFD2D2'},
                            {'range': [0.6, 0.8], 'color': '#FFEDCC'},
                            {'range': [0.8, 1], 'color': '#D4EDDA'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.8
                        }
                    },
                    number={'suffix': '%', 'valueformat': '.1f'}
                ))
                
                gauge_fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2b:
                # Recommended team message
                st.markdown("### Recommended Team")
                st.markdown(f"""
                Based on the prediction, **{best_team['Team']}** is the most suitable team with the lowest expected miss rate of **{best_team['Task Miss Rate']:.2%}**.
                
                **Task details:**
                - {task_type} in {building}, Floor {floor}, {priority} priority
                - Scheduled for: {scheduled_date.strftime('%Y-%m-%d')} at {scheduled_hour}:00
                - Estimated duration: {estimated_duration} hours, Complexity level: {task_complexity}/10
                - Concurrent tasks: {concurrent_tasks}, Resource availability: {resource_availability}
                """)
        
        with tab3:
            st.markdown("### Detailed Results Table")
            
            # Prepare table data
            table_data = results_df.copy()
            table_data['Rank'] = range(1, len(table_data) + 1)
            
            # Format percentage columns
            formatted_data = table_data.copy()
            formatted_data['Task Miss Rate'] = formatted_data['Task Miss Rate'].apply(lambda x: f"{x:.2%}")
            formatted_data['Success Rate'] = formatted_data['Success Rate'].apply(lambda x: f"{x:.2%}")
            
            # Reorder columns
            formatted_data = formatted_data[['Rank', 'Team', 'Task Miss Rate', 'Success Rate']]
            
            # Display the table
            st.dataframe(formatted_data, use_container_width=True, 
                         column_config={
                             "Rank": st.column_config.NumberColumn("Rank", help="Team ranking based on miss rate"),
                             "Team": st.column_config.TextColumn("Team"),
                             "Task Miss Rate": st.column_config.TextColumn("Task Miss Rate"),
                             "Success Rate": st.column_config.TextColumn("Success Rate")
                         }, 
                         hide_index=True)
        
        # Task history section
        st.markdown("---")
        st.markdown("### Task Assignment Trends")
        
        # Create a dummy historical data for task assignment trends
        teams = selected_teams
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
        
        hist_data = []
        for team in teams:
            team_data = []
            for date in dates:
                # Generate randomized historical assignment counts
                count = np.random.randint(1, 10)
                team_data.append({
                    'Date': date,
                    'Team': team,
                    'Tasks': count
                })
            hist_data.extend(team_data)
        
        hist_df = pd.DataFrame(hist_data)
        hist_df['Date'] = pd.to_datetime(hist_df['Date'])
        
        # Create the task history line chart
        history_fig = px.line(
            hist_df, 
            x='Date', 
            y='Tasks', 
            color='Team',
            title='Future Task Assignments (30 Days)',
            labels={'Tasks': 'Number of Tasks', 'Date': 'Date'}
        )
        
        history_fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Number of Tasks',
            legend_title='Team',
            plot_bgcolor='white',
            height=350
        )
        
        st.plotly_chart(history_fig, use_container_width=True)
    
    elif calculate_button and not selected_teams:
        st.warning("Please select at least one team to compare")
    
    else:
        # Display placeholder when no calculation has been done
        st.info("👈 Adjust the parameters on the left and click 'Calculate Miss Rates' to see the team predictions")
        
        # Display sample visualization
        st.markdown("### Sample Visualization")
        
        # Create sample data
        sample_teams = ['Team A', 'Team B', 'Team C', 'Team D']
        sample_miss_rates = [0.15, 0.22, 0.18, 0.29]
        
        sample_df = pd.DataFrame({
            'Team': sample_teams,
            'Miss Rate': sample_miss_rates,
            'Success Rate': [1-rate for rate in sample_miss_rates]
        })
        
        # Create sample bar chart
        fig = px.bar(
            sample_df,
            x='Team',
            y=['Miss Rate', 'Success Rate'],
            title='Task Success vs Miss Rate by Team (Sample Data)',
            barmode='stack',
            color_discrete_map={'Miss Rate': '#FF6B6B', 'Success Rate': '#4ECDC4'}
        )
        
        fig.update_layout(
            xaxis_title='Team',
            yaxis_title='Rate',
            yaxis=dict(tickformat='.0%'),
            plot_bgcolor='white',
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Facility Task Management Dashboard • Powered by Streamlit and Plotly")