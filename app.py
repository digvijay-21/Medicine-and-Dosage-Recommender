import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, LGBMRegressor

# Load dataset
df = pd.read_csv('expanded_ehr_dataset_india_english.csv')

# Feature engineering
df['Systolic BP'] = df['Blood Pressure'].apply(lambda x: int(x.split('/')[0]))
df['Diastolic BP'] = df['Blood Pressure'].apply(lambda x: int(x.split('/')[1]))
df.drop(['Blood Pressure', 'Patient ID', 'Name', 'Address', 'Phone Number', 'Email', 'Doctor', 'Visit Date'], axis=1,
        inplace=True)

# Define features and target
categorical_features = ['Gender', 'Diagnosis', 'Procedure', 'Insurance', 'Smoking Status', 'Alcohol Consumption',
                        'Physical Activity Level', 'Allergies', 'Chronic Conditions', 'Family History',
                        'Recent Surgeries', 'Mental Health', 'Vision Test', 'Hearing Test', 'Vaccination Status']
numerical_features = ['Age', 'BMI', 'Heart Rate', 'Cholesterol', 'Glucose Level', 'Systolic BP', 'Diastolic BP', 'WBC',
                      'RBC', 'Platelets']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Prepare features (X) and target (y)
X = df.drop(['Medication', 'Dose Quantity'], axis=1)
y_medication = df['Medication']
y_dose = df['Dose Quantity']

# Encode medication target
medication_encoder = LabelEncoder()
y_medication_encoded = medication_encoder.fit_transform(y_medication)

# Split data
X_train, X_test, y_med_train, y_med_test, y_dose_train, y_dose_test = train_test_split(X, y_medication_encoded, y_dose,
                                                                                       test_size=0.2, random_state=42)

# Medication model: Classification
medication_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=42, verbose=-1, log_level='error'))  # Suppress logs
])

# Dose model: Regression
dose_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(random_state=42, verbose=-1, log_level='error'))  # Suppress logs
])

# Train the medication model
medication_pipeline.fit(X_train, y_med_train)

# Train the dose model
dose_pipeline.fit(X_train, y_dose_train)

# Function to suggest medication and dose for a new patient
def suggest_medication(patient_details):
    # Convert patient details to a DataFrame
    patient_df = pd.DataFrame([patient_details])

    # Reorder columns to match the original DataFrame's order
    patient_df = patient_df[X.columns]  # Align columns

    # Predict medication
    med_pred = medication_pipeline.predict(patient_df)
    suggested_medication = medication_encoder.inverse_transform(med_pred)[0]

    # Predict dose
    dose_pred = dose_pipeline.predict(patient_df)[0]

    return suggested_medication, dose_pred

# Function to calculate the dosage schedule
def calculate_dosage_schedule(start_time, num_doses, interval_hours):
    schedule = []
    for i in range(num_doses):
        dose_time = start_time + timedelta(hours=interval_hours * i)
        schedule.append(dose_time.strftime("%d-%b %I:%M %p"))
    return schedule

# Streamlit frontend
st.title("Medication and Dosage Suggestion System")

st.sidebar.header("Enter Patient Details")

# User inputs
age = st.sidebar.slider('Age', 18, 90, 45)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'])
diagnosis = st.sidebar.selectbox('Diagnosis', ['Hypertension', 'Diabetes', 'Asthma', 'Chronic Pain', 'Hyperlipidemia',
                                               'Anxiety', 'Depression', 'Allergy', 'Cardiovascular Disease',
                                               'COPD', 'Osteoarthritis'])
procedure = st.sidebar.selectbox('Procedure', ['Blood Test', 'MRI', 'CT Scan', 'X-ray', 'Ultrasound', 'Biopsy',
                                               'EKG', 'Colonoscopy', 'Endoscopy', 'Physical Therapy'])
insurance = st.sidebar.selectbox('Insurance', ['Yes', 'No'])
bmi = st.sidebar.number_input('BMI', min_value=18.5, max_value=40.0, value=28.4, step=0.1)
heart_rate = st.sidebar.slider('Heart Rate', 60, 100, 78)
cholesterol = st.sidebar.number_input('Cholesterol (mg/dL)', min_value=150.0, max_value=280.0, value=220.0, step=0.1)
glucose_level = st.sidebar.number_input('Glucose Level (mg/dL)', min_value=70.0, max_value=180.0, value=110.0, step=0.1)
systolic_bp = st.sidebar.slider('Systolic Blood Pressure', 90, 160, 140)
diastolic_bp = st.sidebar.slider('Diastolic Blood Pressure', 60, 100, 90)
wbc = st.sidebar.number_input('WBC', min_value=3.0, max_value=12.0, value=6.5, step=0.1)
rbc = st.sidebar.number_input('RBC', min_value=3.5, max_value=5.5, value=4.8, step=0.1)
platelets = st.sidebar.number_input('Platelets', min_value=150.0, max_value=450.0, value=300.0, step=1.0)
smoking_status = st.sidebar.selectbox('Smoking Status', ['Never', 'Former', 'Current'])
alcohol_consumption = st.sidebar.selectbox('Alcohol Consumption', ['None', 'Occasional', 'Regular'])
physical_activity = st.sidebar.selectbox('Physical Activity Level', ['Sedentary', 'Moderate', 'Active'])
allergies = st.sidebar.selectbox('Allergies', ['Penicillin', 'Peanuts', 'Shellfish', 'Latex', 'Pollen', 'None'])
chronic_conditions = st.sidebar.selectbox('Chronic Conditions', ['Hypertension', 'Diabetes', 'Asthma',
                                                                  'Chronic Kidney Disease', 'Heart Failure',
                                                                  'COPD', 'None'])
family_history = st.sidebar.selectbox('Family History', ['Heart Disease', 'Diabetes', 'Cancer', 'None'])
recent_surgeries = st.sidebar.selectbox('Recent Surgeries', ['Appendectomy', 'Gallbladder Removal',
                                                             'Knee Replacement', 'None'])
mental_health = st.sidebar.selectbox('Mental Health', ['Stable', 'Anxiety', 'Depression', 'Bipolar Disorder'])
vision_test = st.sidebar.selectbox('Vision Test', ['Normal', 'Near-sighted', 'Far-sighted', 'Glaucoma'])
hearing_test = st.sidebar.selectbox('Hearing Test', ['Normal', 'Mild Loss', 'Moderate Loss', 'Severe Loss'])
vaccination_status = st.sidebar.selectbox('Vaccination Status', ['Up-to-date', 'Partial', 'None'])

# Dosage scheduling inputs
num_doses = st.sidebar.number_input('Number of Doses', min_value=1, max_value=12, value=3, step=1)
interval_hours = st.sidebar.number_input('Interval between Doses (hours)', min_value=1.0, max_value=24.0, value=6.0, step=0.5)
start_time_input = st.sidebar.time_input('Start Time', value=datetime.now().time())

# Convert start time to datetime
start_time = datetime.combine(datetime.now().date(), start_time_input)

# Create dictionary for patient details
patient_details = {
    'Age': age,
    'Gender': gender,
    'Diagnosis': diagnosis,
    'Procedure': procedure,
    'Insurance': insurance,
    'BMI': bmi,
    'Heart Rate': heart_rate,
    'Cholesterol': cholesterol,
    'Glucose Level': glucose_level,
    'Systolic BP': systolic_bp,
    'Diastolic BP': diastolic_bp,
    'WBC': wbc,
    'RBC': rbc,
    'Platelets': platelets,
    'Smoking Status': smoking_status,
    'Alcohol Consumption': alcohol_consumption,
    'Physical Activity Level': physical_activity,
    'Allergies': allergies,
    'Chronic Conditions': chronic_conditions,
    'Family History': family_history,
    'Recent Surgeries': recent_surgeries,
    'Mental Health': mental_health,
    'Vision Test': vision_test,
    'Hearing Test': hearing_test,
    'Vaccination Status': vaccination_status
}

# Get suggestions
if st.sidebar.button('Get Suggestions'):
    suggested_med, suggested_dose = suggest_medication(patient_details)
    st.markdown(f"<h3 style='font-size: 24px;'>Suggested Medication: {suggested_med}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-size: 24px;'>Suggested Dose: {suggested_dose:.2f}</h3>", unsafe_allow_html=True)

    # Calculate the dosage schedule
    schedule = calculate_dosage_schedule(start_time, num_doses, interval_hours)

    # Display the schedule in a table format
    st.write("### Dosage Schedule")
    schedule_df = pd.DataFrame({"Dose Number": range(1, num_doses + 1), "Scheduled Time": schedule})
    st.table(schedule_df)
