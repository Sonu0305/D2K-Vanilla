import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os
from dash.exceptions import PreventUpdate

# Initialize the Dash app
app = dash.Dash(__name__, title="Facility Task Management Dashboard")
server = app.server

# Define styles
colors = {
    'background': '#F9F9F9',
    'text': '#333333',
    'accent': '#007BFF',
    'secondary': '#6c757d'
}

# Load the trained model and preprocessing components
model_path = r'output\kaggle\working\facility_task_prediction_model.pkl'
model_data = joblib.load(model_path)
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

# Define app layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'padding': '20px'}, children=[
    # Header
    html.Div([
        html.H1("Facility Task Management Dashboard", 
                style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '20px'}),
        html.H3("Task Miss Rate Prediction & Team Comparison", 
                style={'textAlign': 'center', 'color': colors['secondary'], 'marginBottom': '30px'})
    ]),
    
    # Main content - two columns
    html.Div([
        # Left column - Input parameters
        html.Div([
            html.H4("Task Parameters", style={'color': colors['accent'], 'marginBottom': '15px'}),
            
            # Task type
            html.Label("Task Type:"),
            dcc.Dropdown(
                id='task-type-dropdown',
                options=[
                    {'label': 'Corrective Maintenance', 'value': 'Corrective Maintenance'},
                    {'label': 'Preventive Maintenance', 'value': 'Preventive Maintenance'},
                    {'label': 'Emergency Repair', 'value': 'Emergency Repair'},
                    {'label': 'Installation', 'value': 'Installation'},
                    {'label': 'Inspection', 'value': 'Inspection'}
                ],
                value='Corrective Maintenance',
                clearable=False
            ),
            
            # Building
            html.Label("Building:", style={'marginTop': '10px'}),
            dcc.Dropdown(
                id='building-dropdown',
                options=[
                    {'label': f'Building_{i}', 'value': f'Building_{i}'} for i in range(1, 11)
                ],
                value='Building_4',
                clearable=False
            ),
            
            # Floor
            html.Label("Floor:", style={'marginTop': '10px'}),
            dcc.Dropdown(
                id='floor-dropdown',
                options=[{'label': str(i), 'value': i} for i in range(1, 21)],
                value=9,
                clearable=False
            ),
            
            # Priority
            html.Label("Priority:", style={'marginTop': '10px'}),
            dcc.Dropdown(
                id='priority-dropdown',
                options=[
                    {'label': 'Low', 'value': 'Low'},
                    {'label': 'Medium', 'value': 'Medium'},
                    {'label': 'High', 'value': 'High'},
                    {'label': 'Critical', 'value': 'Critical'}
                ],
                value='Low',
                clearable=False
            ),
            
            # Scheduled Time
            html.Label("Scheduled Date & Time:", style={'marginTop': '10px'}),
            dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed=datetime.now().date(),
                max_date_allowed=datetime.now().date() + timedelta(days=30),
                initial_visible_month=datetime.now().date(),
                date=datetime.now().date()
            ),
            dcc.Dropdown(
                id='time-dropdown',
                options=[{'label': f'{i}:00', 'value': i} for i in range(24)],
                value=9,
                clearable=False,
                style={'marginTop': '5px'}
            ),
            
            # Estimated Duration
            html.Label("Estimated Duration (hours):", style={'marginTop': '10px'}),
            dcc.Input(
                id='duration-input',
                type='number',
                min=0.5,
                max=24,
                step=0.5,
                value=2.0
            ),
            
            # Task Complexity
            html.Label("Task Complexity (0-10):", style={'marginTop': '10px'}),
            dcc.Slider(
                id='complexity-slider',
                min=0,
                max=10,
                step=1,
                value=3,
                marks={i: str(i) for i in range(11)},
            ),
            
            # Concurrent Tasks
            html.Label("Concurrent Tasks:", style={'marginTop': '10px'}),
            dcc.Input(
                id='concurrent-input',
                type='number',
                min=0,
                max=20,
                step=1,
                value=5
            ),
            
            # Resource Availability
            html.Label("Resource Availability (0-1):", style={'marginTop': '10px'}),
            dcc.Slider(
                id='resource-slider',
                min=0,
                max=1,
                step=0.1,
                value=0.7,
                marks={i/10: str(i/10) for i in range(11)},
            ),
            
            # Submit button
            html.Button('Calculate Miss Rates', 
                      id='submit-button', 
                      style={'marginTop': '20px', 
                             'backgroundColor': colors['accent'],
                             'color': 'white',
                             'padding': '10px 15px',
                             'border': 'none',
                             'borderRadius': '5px',
                             'cursor': 'pointer',
                             'fontSize': '16px',
                             'width': '100%'})
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)'}),
        
        # Right column - Results
        html.Div([
            # Team selection checkboxes
            html.H4("Select Teams to Compare", style={'color': colors['accent'], 'marginBottom': '15px'}),
            dcc.Checklist(
                id='team-checklist',
                options=[
                    {'label': ' Maintenance A', 'value': 'Maintenance A'},
                    {'label': ' Maintenance B', 'value': 'Maintenance B'},
                    {'label': ' HVAC Team', 'value': 'HVAC Team'},
                    {'label': ' Electrical Team', 'value': 'Electrical Team'},
                    {'label': ' Plumbing Team', 'value': 'Plumbing Team'},
                    {'label': ' Security Team', 'value': 'Security Team'},
                    {'label': ' Janitorial Team', 'value': 'Janitorial Team'}
                ],
                value=['Maintenance A', 'Maintenance B', 'HVAC Team', 'Electrical Team', 'Plumbing Team'],
                inline=False,
                labelStyle={'display': 'block', 'marginBottom': '5px'}
            ),
            
            # Results visualization
            html.Div([
                html.H4("Task Miss Rate Comparison", style={'color': colors['accent'], 'marginTop': '20px', 'marginBottom': '15px'}),
                dcc.Graph(id='miss-rate-chart'),
                
                # Gauge chart for best team
                html.Div([
                    html.H5("Best Team Performance", style={'color': colors['accent'], 'textAlign': 'center', 'marginBottom': '10px'}),
                    dcc.Graph(id='gauge-chart')
                ]),
                
                html.Div(id='recommended-team', style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#e8f4f8', 'borderRadius': '5px'}),
                
                html.H4("Detailed Results Table", style={'color': colors['accent'], 'marginTop': '20px', 'marginBottom': '15px'}),
                dash_table.DataTable(
                    id='results-table',
                    style_header={
                        'backgroundColor': colors['accent'],
                        'color': 'white',
                        'fontWeight': 'bold'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'padding': '10px'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#f9f9f9'
                        },
                        {
                            'if': {'filter_query': '{Rank} = 1'},
                            'backgroundColor': '#e6f7ff',
                            'fontWeight': 'bold'
                        }
                    ]
                )
            ], id='results-container', style={'display': 'none'})
        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '20px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)'})
    ]),
    
    # Task history section
    html.Div([
        html.H4("Task Assignment Trends", style={'color': colors['accent'], 'marginTop': '30px', 'marginBottom': '15px'}),
        dcc.Graph(id='task-history')
    ], style={'marginTop': '20px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)'}),
    
    # Footer
    html.Div([
        html.Hr(style={'marginTop': '30px', 'marginBottom': '15px'}),
        html.P("Facility Task Management Dashboard • Powered by Dash and Plotly", 
              style={'textAlign': 'center', 'color': colors['secondary'], 'fontSize': '14px'})
    ])
])

# Callback to update results based on input parameters
@app.callback(
    [Output('miss-rate-chart', 'figure'),
     Output('gauge-chart', 'figure'),
     Output('results-table', 'data'),
     Output('results-table', 'columns'),
     Output('recommended-team', 'children'),
     Output('results-container', 'style'),
     Output('task-history', 'figure')],
    Input('submit-button', 'n_clicks'),
    [State('task-type-dropdown', 'value'),
     State('building-dropdown', 'value'),
     State('floor-dropdown', 'value'),
     State('priority-dropdown', 'value'),
     State('date-picker', 'date'),
     State('time-dropdown', 'value'),
     State('duration-input', 'value'),
     State('complexity-slider', 'value'),
     State('concurrent-input', 'value'),
     State('resource-slider', 'value'),
     State('team-checklist', 'value')]
)
def update_results(n_clicks, task_type, building, floor, priority, date_value, 
                  time_value, duration, complexity, concurrent, resource, selected_teams):
    if n_clicks is None:
        raise PreventUpdate
    
    # Construct scheduled time
    scheduled_date = datetime.strptime(date_value, '%Y-%m-%d')
    scheduled_time = scheduled_date.replace(hour=time_value)
    
    # Store results for each team
    results = []
    
    # Loop through each selected team and predict the task miss rate
    for team in selected_teams:
        miss_rate = predict_task_miss_rate(
            team, task_type, building, floor, priority, scheduled_time, 
            duration, complexity, concurrent, resource
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
        title='Team Performance Comparison',
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
    
    # Create gauge chart for best team
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=best_team['Success Rate'],
        title={'text': f"Best Team: {best_team['Team']}", 'font': {'size': 24}},
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
        height=250,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    # Prepare table data and columns
    table_data = results_df.copy()
    table_data['Task Miss Rate'] = table_data['Task Miss Rate'].apply(lambda x: f"{x:.2%}")
    table_data['Success Rate'] = table_data['Success Rate'].apply(lambda x: f"{x:.2%}")
    table_data['Rank'] = range(1, len(table_data) + 1)
    
    # Reorder columns
    table_data = table_data[['Rank', 'Team', 'Task Miss Rate', 'Success Rate']]
    
    # Define table columns
    columns = [{'name': col, 'id': col} for col in table_data.columns]
    
    # Recommended team message
    recommended_team_msg = html.Div([
        html.H5("Recommended Team", style={'marginBottom': '10px', 'color': colors['accent']}),
        html.P([
            f"Based on the prediction, ",
            html.Strong(f"{best_team['Team']}"),
            f" is the most suitable team with the lowest expected miss rate of ",
            html.Strong(f"{best_team['Task Miss Rate']:.2%}"),
            "."
        ]),
        html.P([
            f"Task details: {task_type} in {building}, Floor {floor}, {priority} priority",
            html.Br(),
            f"Scheduled for: {scheduled_date.strftime('%Y-%m-%d')} at {time_value}:00",
            html.Br(),
            f"Estimated duration: {duration} hours, Complexity level: {complexity}/10"
        ], style={'fontSize': '14px', 'marginTop': '10px', 'color': colors['secondary']})
    ])
    
    # Display the results container
    results_style = {'display': 'block'}
    
    # Create a dummy historical data for task assignment trends
    # In a real app, this would come from a database
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
        title='Historical Task Assignments (Last 30 Days)',
        labels={'Tasks': 'Number of Tasks', 'Date': 'Date'}
    )
    
    history_fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Tasks',
        legend_title='Team',
        plot_bgcolor='white',
        height=350
    )
    
    return fig, gauge_fig, table_data.to_dict('records'), columns, recommended_team_msg, results_style, history_fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port='8051')