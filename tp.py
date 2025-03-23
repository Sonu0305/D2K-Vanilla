import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = 'AIzaSyDGGDccSDqbR5VWfa31GCqhf3fQrHKZSO4'
genai.configure(api_key=GOOGLE_API_KEY)

# Set page config
st.set_page_config(
    page_title="Facility Management Team Recommender",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("🏢 Facility Management Team Recommender")
st.markdown("""
This application helps you find the best team for your facility management tasks
by predicting task miss rates for different teams based on task parameters.
""")

# Function to load the model
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load(r'output\kaggle\working\facility_task_prediction_model.pkl')
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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
    input_df['scheduled_date'] = input_df['scheduled_time'].dt.date
    input_df['scheduled_hour'] = input_df['scheduled_time'].dt.hour
    input_df['scheduled_day'] = input_df['scheduled_time'].dt.day
    input_df['scheduled_month'] = input_df['scheduled_time'].dt.month
    input_df['scheduled_year'] = input_df['scheduled_time'].dt.year
    input_df['day_of_week'] = input_df['scheduled_time'].dt.dayofweek
    input_df['is_weekend'] = input_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Feature engineering (simplified)
    input_df['daily_type_count'] = 1
    input_df['team_daily_workload'] = 1
    input_df['building_daily_workload'] = 1
    input_df['rolling_miss_rate'] = 0.1
    
    # Priority encoding
    priority_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    input_df['priority_level'] = input_df['priority'].map(priority_map)
    
    # Time since last maintenance
    input_df['days_since_last_maintenance'] = 30
    
    return input_df

# Function to predict task miss rate
def predict_task_miss_rate(model_data, team_name, task_type, building, floor, priority, 
                          scheduled_time, estimated_duration, task_complexity, 
                          concurrent_tasks, resource_availability):
    
    model = model_data['model']
    scaler = model_data['scaler']
    encoder = model_data['encoder']
    numerical_features = model_data['numerical_features']
    categorical_features = model_data['categorical_features']
    
    # Prepare input data
    input_df = prepare_input_data(team_name, task_type, building, floor, priority, 
                                 scheduled_time, estimated_duration, task_complexity, 
                                 concurrent_tasks, resource_availability)
    
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

# Initialize Gemini chatbot
def get_gemini_response(prompt, conversation_history=None):
    if conversation_history is None:
        conversation_history = []
    
    # Add system prompt for context
    system_prompt = """
    You are an AI assistant for a facility management team recommendation system. 
    You can help users understand:
    1. How the prediction model works
    2. What factors influence task success rates
    3. How to interpret the prediction results
    4. Best practices for facility management task allocation
    
    Keep your answers focused on facility management and the prediction model.
    When asked about technical details, explain in a way that's easy to understand.
    """
    
    # Create the model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        system_instruction=system_prompt
    )
    
    # Convert conversation history to format expected by Gemini if needed
    gemini_history = []
    for msg in conversation_history:
        if isinstance(msg, dict) and "role" in msg:
            role = "user" if msg["role"] == "user" else "model"
            if "content" in msg:
                gemini_history.append({"role": role, "parts": [msg["content"]]})
            elif "parts" in msg and isinstance(msg["parts"], list) and len(msg["parts"]) > 0:
                gemini_history.append({"role": role, "parts": msg["parts"]})
    
    # Create or continue the chat
    if not gemini_history:
        chat = model.start_chat(history=[])
    else:
        chat = model.start_chat(history=gemini_history)
    
    # Generate response
    response = chat.send_message(prompt)
    
    # Return the text response only
    return response.text, conversation_history

# Sidebar for model info and inputs
st.sidebar.header("Task Parameters")

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load model data
model_data = load_model()

if model_data:
    # Input form for task parameters
    with st.sidebar.form("task_form"):
        # Task parameters
        task_type = st.selectbox(
            "Task Type",
            options=[
                'Preventive Maintenance', 'Corrective Maintenance', 'Safety Inspection',
                'Cleaning', 'Filter Replacement', 'Equipment Calibration', 'Security Check',
                'HVAC Service', 'Plumbing Repair', 'Electrical Inspection', 'Pest Control',
                'Landscape Maintenance', 'Fire System Check', 'Elevator Maintenance'
            ],
            index=0
        )
        
        building = st.selectbox(
            "Building",
            options=[f'Building_{i}' for i in range(1, 6)],
            index=3
        )
        
        floor = st.number_input("Floor", min_value=1, max_value=10, value=9)
        
        priority = st.selectbox(
            "Priority",
            options=['Low', 'Medium', 'High', 'Critical'],
            index=0
        )
        
        today = datetime.now()
        scheduled_date = st.date_input("Scheduled Date", today + timedelta(days=1))
        scheduled_hour = st.slider("Scheduled Hour", 7, 19, 9)
        scheduled_time = datetime.combine(scheduled_date, datetime.min.time()) + timedelta(hours=scheduled_hour)
        
        estimated_duration = st.slider("Estimated Duration (hours)", 0.5, 5.0, 2.0, 0.5)
        
        task_complexity = st.slider("Task Complexity (1-10)", 1, 10, 2)
        
        concurrent_tasks = st.slider("Concurrent Tasks", 0, 10, 5)
        
        resource_availability = st.slider("Resource Availability (0-1)", 0.5, 1.0, 0.9, 0.1)
        
        submitted = st.form_submit_button("Predict Best Team")

    # Main content area with tabs
    tab1, tab2 = st.tabs(["Team Recommendations", "AI Assistant"])
    
    with tab1:
        if submitted:
            st.subheader("Team Performance Prediction")
            
            # Define the list of teams to compare
            teams = ['Maintenance A', 'Maintenance B', 'HVAC Team', 'Electrical Team', 
                    'Plumbing Team', 'Security Team', 'Janitorial Team']
            
            # Store results for each team
            results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            
            # Loop through each team and predict the task miss rate
            for i, team in enumerate(teams):
                miss_rate = predict_task_miss_rate(
                    model_data, team, task_type, building, floor, priority, 
                    scheduled_time, estimated_duration, task_complexity, 
                    concurrent_tasks, resource_availability
                )
                
                results.append({
                    'Team': team,
                    'Task Miss Rate': miss_rate,
                    'Success Rate': 1 - miss_rate
                })
                
                # Update progress
                progress_bar.progress((i + 1) / len(teams))
            
            # Convert results to a DataFrame for better visualization
            results_df = pd.DataFrame(results)
            
            # Sort by task miss rate
            results_df = results_df.sort_values('Task Miss Rate')
            
            # Determine the best-performing team (lowest task miss rate)
            best_team = results_df.iloc[0]
            
            # Create columns for visualization
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Best Team for Task")
                st.markdown(f"""
                #### 🏆 {best_team['Team']}
                - **Success Rate:** {(1 - best_team['Task Miss Rate']):.2%}
                - **Miss Rate:** {best_team['Task Miss Rate']:.2%}
                """)
                
                st.markdown("### Task Details")
                st.markdown(f"""
                - **Task Type:** {task_type}
                - **Location:** {building}, Floor {floor}
                - **Priority:** {priority}
                - **Scheduled:** {scheduled_date.strftime('%Y-%m-%d')} at {scheduled_hour}:00
                - **Complexity:** {task_complexity}/10
                """)
            
            with col2:
                # Create bar chart of miss rates
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Plot success rates instead of miss rates
                success_rates = results_df['Success Rate'].values
                teams = results_df['Team'].values
                
                bars = ax.barh(teams, success_rates, color='skyblue')
                
                # Add data labels
                for i, v in enumerate(success_rates):
                    ax.text(v + 0.01, i, f"{v:.2%}", va='center')
                
                # Add a reference line at 80% success rate
                ax.axvline(0.8, color='red', linestyle='--', alpha=0.7, label='80% Success Target')
                
                ax.set_title('Team Success Rates for Task')
                ax.set_xlabel('Success Rate')
                ax.set_xlim(0, 1)
                ax.legend()
                ax.grid(axis='x', alpha=0.3)
                
                st.pyplot(fig)
                
                # Display detailed results table
                st.subheader("All Teams Comparison")
                
                # Format the dataframe for display
                display_df = results_df.copy()
                display_df['Success Rate'] = display_df['Success Rate'].apply(lambda x: f"{x:.2%}")
                display_df['Task Miss Rate'] = display_df['Task Miss Rate'].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(display_df, hide_index=True)
        
        else:
            st.info("👈 Enter task parameters in the sidebar and click 'Predict Best Team' to get recommendations.")
            
            # Show sample visualization
            st.subheader("Sample Visualization")
            
            # Create sample data
            sample_teams = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']
            sample_rates = [0.92, 0.87, 0.83, 0.78, 0.71]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.barh(sample_teams, sample_rates, color='lightblue')
            ax.set_title('Sample Team Success Rates')
            ax.set_xlabel('Success Rate')
            ax.set_xlim(0, 1)
            ax.axvline(0.8, color='red', linestyle='--', alpha=0.7, label='80% Success Target')
            ax.grid(axis='x', alpha=0.3)
            ax.legend()
            
            # Add data labels
            for i, v in enumerate(sample_rates):
                ax.text(v + 0.01, i, f"{v:.2%}", va='center')
                
            st.pyplot(fig)
    
    with tab2:
        st.subheader("AI Assistant")
        st.markdown("""
        Ask questions about the facility management prediction model, team performance, 
        or best practices for task allocation. Our AI assistant can help you understand 
        factors that impact task completion rates.
        """)
        
        # Chat interface
        user_input = st.text_input("Your question:", key="user_input")
        
        if st.button("Ask"):
            if user_input:
                with st.spinner("Thinking..."):
                    # Prepare context about the current prediction if available
                    if 'results_df' in locals():
                        context = f"""
                        The user has just predicted team performance for the following task:
                        - Task Type: {task_type}
                        - Building: {building}, Floor: {floor}
                        - Priority: {priority}
                        - Complexity: {task_complexity}/10
                        - Best Team: {best_team['Team']} with Success Rate: {(1-best_team['Task Miss Rate']):.2%}
                        
                        Please consider this context when answering their question: "{user_input}"
                        """
                    else:
                        context = f"Question from user: {user_input}"
                    
                    # Get response from Gemini
                    response, _ = get_gemini_response(context, st.session_state.chat_history)
                    
                    # Add to chat history display with standardized format
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display chat history with proper formatting
        for message in st.session_state.chat_history:
            try:
                if isinstance(message, dict) and "role" in message:
                    if message["role"] == "user" and "content" in message:
                        st.markdown(f"**You:** {message['content']}")
                    elif message["role"] == "assistant" and "content" in message:
                        # Format the response - replace \n with proper markdown line breaks
                        formatted_content = message['content'].replace("\n", "<br>")
                        st.markdown(f"**Assistant:** {formatted_content}", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying message: {str(e)}")
        
        # Tips for using the AI assistant
        with st.expander("Example Questions"):
            st.markdown("""
            - How does the prediction model work?
            - What factors have the biggest impact on task success rates?
            - Why might the HVAC team have a higher miss rate for this task?
            - What can I do to improve task completion rates?
            - How should I prioritize tasks between different teams?
            - What does task complexity measure in the model?
            """)
else:
    st.error("Failed to load the prediction model. Please check if the model file exists and is accessible.")
    
# Footer
st.markdown("---")
st.markdown("© 2025 Facility Management Team Recommender | Built with Streamlit")