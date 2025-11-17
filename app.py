import streamlit as st
import joblib
import pandas as pd
import time
import googlemaps
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase

# --- 1. Page Configuration (MUST be the first command) ---
st.set_page_config(
    page_title="Health Navigator",
    page_icon="🩺",
    layout="wide"
)

# --- 2. Initialize APIs ---

# Initialize Google Maps Client
try:
    gmaps = googlemaps.Client(key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Could not initialize Google Maps: {e}. Check your API key.")
    gmaps = None

# Initialize Firebase Admin (for database)
try:
    # Check if app is already initialized
    if not firebase_admin._apps:
        # Use the secrets to create a credentials dictionary
        cred_dict = dict(st.secrets["firebase_service_account"])
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    print("Firestore client initialized.")
except Exception as e:
    st.error(f"Could not initialize Firestore Admin: {e}. Check your service_account keys.")
    db = None

# Initialize Pyrebase (for auth)
try:
    firebase_config = dict(st.secrets["firebase_config"])
    firebase = pyrebase.initialize_app(firebase_config)
    auth = firebase.auth()
    print("Firebase Auth initialized.")
except Exception as e:
    st.error(f"Could not initialize Firebase Auth: {e}. Check your firebase_config keys.")
    auth = None


# --- 3. Caching ---
@st.cache_resource
def load_model():
    """Loads the saved machine learning model."""
    try:
        model = joblib.load('model/disease_predictor_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Error: Model file not found. Make sure 'disease_predictor_model.pkl' is in the 'model' folder.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None


@st.cache_data
def get_symptoms_list():
    """Loads the list of symptoms from the Training.csv file."""
    try:
        data = pd.read_csv('Training.csv')
        all_cols = data.columns.tolist()
        symptoms = [col for col in all_cols if col != 'prognosis' and not col.startswith('Unnamed:')]
        return symptoms
    except FileNotFoundError:
        st.error("Error: 'Training.csv' not found. Make sure it's in the main project folder.")
        return []
    except Exception as e:
        st.error(f"An error occurred while loading symptoms: {e}")
        return []


# --- 4. Load resources ---
model = load_model()
symptoms = get_symptoms_list()

# --- 5. Session State Initialization ---
# This holds our user's login state
if 'user' not in st.session_state:
    st.session_state.user = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'doctor_profile' not in st.session_state:
    st.session_state.doctor_profile = None

# App-specific state
if "booking_confirmation" not in st.session_state:
    st.session_state.booking_confirmation = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "specialist" not in st.session_state:
    st.session_state.specialist = None
if "doctors_list" not in st.session_state:
    st.session_state.doctors_list = []
if "map_data_list" not in st.session_state:
    st.session_state.map_data_list = []


# --- 6. Helper Function to Check User Role ---
def check_user_role(user_id):
    """Checks the 'doctors' collection in Firestore to determine user role."""
    if db:
        doc_ref = db.collection("doctors").document(user_id)
        profile = doc_ref.get()
        if profile.exists:
            st.session_state.doctor_profile = profile.to_dict()
            return "doctor"
    return "patient"


# --- 7. Main App Logic ---

# Check if all services are initialized
services_ready = model and symptoms and gmaps and db and auth

# If user is not logged in, show login/signup page
if st.session_state.user is None:

    st.title("Welcome to Health Navigator 🩺")
    st.markdown("Please log in or sign up to continue.")

    if not services_ready:
        st.error("A core service (Model, Auth, or Database) failed to load. Please check server logs.")
    else:
        login_tab, patient_signup_tab, doctor_signup_tab = st.tabs(["Login", "Patient Signup", "Doctor Signup"])

        # --- Login Tab ---
        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                login_button = st.form_submit_button("Login")

                if login_button:
                    try:
                        user = auth.sign_in_with_email_and_password(email, password)
                        st.session_state.user = user  # Save user info in state
                        st.session_state.user_role = check_user_role(user['localId'])  # Check and save role
                        st.rerun()  # Rerun the app
                    except Exception as e:
                        st.error(f"Login failed: Invalid email or password.")

        # --- Patient Sign Up Tab ---
        with patient_signup_tab:
            with st.form("patient_signup_form"):
                st.markdown("Create a new patient account.")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                signup_button = st.form_submit_button("Create Patient Account")

                if signup_button:
                    if not email or not password or not confirm_password:
                        st.warning("Please fill out all fields.")
                    elif password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            user = auth.create_user_with_email_and_password(email, password)
                            st.success("Patient account created successfully! Please log in.")
                        except Exception as e:
                            st.error(f"Sign up failed. This email may already be in use.")

        # --- Doctor Sign Up Tab ---
        with doctor_signup_tab:
            with st.form("doctor_signup_form"):
                st.markdown("Create a new doctor profile.")
                email = st.text_input("Your Email")
                password = st.text_input("Password", type="password")
                st.divider()
                st.markdown("Enter your clinic's **exact** name and location to link your profile.")
                clinic_name = st.text_input("Your Clinic's Name (e.g., 'Medanta, Lucknow')")
                clinic_location = st.text_input("City/Area (e.g., 'Lucknow')")
                signup_button = st.form_submit_button("Create Doctor Account")

                if signup_button:
                    if not email or not password or not clinic_name or not clinic_location:
                        st.warning("Please fill out all fields.")
                    else:
                        try:
                            with st.spinner("Creating your user account..."):
                                user = auth.create_user_with_email_and_password(email, password)
                                user_id = user['localId']

                            with st.spinner(f"Finding {clinic_name} in {clinic_location}..."):
                                query = f"{clinic_name} {clinic_location}"
                                places_result = gmaps.places(query=query)
                                if not places_result.get('results'):
                                    st.error(f"Error: Could not find a clinic matching '{query}'.")
                                    auth.delete_user_account(user['idToken'])
                                    st.stop()

                                top_result = places_result['results'][0]
                                found_name = top_result['name']
                                found_address = top_result['formatted_address']
                                found_place_id = top_result['place_id']

                            with st.spinner(f"Saving your profile..."):
                                profile_data = {
                                    "email": email, "clinic_name": found_name,
                                    "address": found_address, "place_id": found_place_id
                                }
                                db.collection("doctors").document(user_id).set(profile_data)
                            st.success(f"Success! Your profile for '{found_name}' is created. Please log in.")

                        except Exception as e:
                            st.error(f"Sign up failed: {e}")

# If user IS logged in, show the correct dashboard
else:
    # --- Sidebar ---
    with st.sidebar:
        user_email = st.session_state.user['email']
        st.markdown(f"Logged in as: **{user_email}**")

        if st.button("Logout"):
            # Clear all session state on logout
            for key in st.session_state.keys():
                if key != 'db' and key != 'gmaps':  # Keep services initialized
                    st.session_state[key] = None
            st.rerun()

        st.divider()
        st.title("About")
        st.info("This app predicts conditions and recommends local specialists.")
        st.divider()
        st.title("ℹ️ Disclaimer")
        st.warning("This tool is for informational purposes only. Always consult a qualified healthcare provider.")

    # --- ROLE-BASED ROUTING ---

    # --- A. DOCTOR DASHBOARD ---
    if st.session_state.user_role == "doctor":
        st.title("🧑‍⚕️ Doctor's Appointment Dashboard")

        if not db:
            st.error("Database client is not initialized.")
        else:
            try:
                # Load profile if it's not in state
                if st.session_state.doctor_profile is None:
                    st.session_state.user_role = check_user_role(st.session_state.user['localId'])

                profile = st.session_state.doctor_profile
                st.subheader(f"Managing Appointments for: **{profile['clinic_name']}**")
                st.caption(f"📍 {profile['address']}")

                place_id = profile['place_id']

                appointments_ref = db.collection("appointments") \
                    .where("doctor_place_id", "==", place_id) \
                    .stream()

                appointments = list(appointments_ref)

                if not appointments:
                    st.warning("You have no pending appointments.")
                else:
                    st.divider()
                    st.subheader("Manage Appointments")

                    col1, col2, col3, col4, col5 = st.columns([2, 2, 3, 2, 3])
                    col1.write("**Date**")
                    col2.write("**Time**")
                    col3.write("**Patient Name**")
                    col4.write("**Status**")
                    col5.write("**Actions**")
                    st.divider()

                    for appt in appointments:
                        appt_id = appt.id
                        appt_data = appt.to_dict()

                        col1, col2, col3, col4, col5 = st.columns([2, 2, 3, 2, 3])

                        col1.write(appt_data.get("appointment_date"))
                        col2.write(appt_data.get("appointment_time"))
                        col3.write(appt_data.get("patient_name"))

                        status = appt_data.get("status", "Pending")
                        if status == "Pending":
                            col4.warning(status)
                        elif status == "Accepted":
                            col4.success(status)
                        elif status == "Declined":
                            col4.error(status)

                        btn_col1, btn_col2 = col5.columns(2)

                        if status == "Pending":  # Only show buttons if pending
                            if btn_col1.button("Accept", key=f"accept_{appt_id}", use_container_width=True):
                                db.collection("appointments").document(appt_id).update({"status": "Accepted"})
                                st.rerun()

                            if btn_col2.button("Decline", key=f"decline_{appt_id}", use_container_width=True):
                                db.collection("appointments").document(appt_id).update({"status": "Declined"})
                                st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {e}")

    # --- B. PATIENT APP ---
    elif st.session_state.user_role == "patient":
        st.title("Patient Dashboard 🩺")

        tab1, tab2 = st.tabs(["Find a Doctor 🔍", "My Appointments 🗓️"])

        # --- TAB 1: FIND A DOCTOR ---
        with tab1:
            if not services_ready:
                st.error("A core service (Maps, Database, or Model) could not be initialized.")
            else:
                col1, col2 = st.columns([1, 1.2])

                with col1:
                    st.subheader("🧑‍⚕️ Your Information")
                    user_symptoms = st.multiselect(
                        label="What symptoms are you experiencing?",
                        options=symptoms,
                        placeholder="Type and select your symptoms"
                    )
                    user_location = st.text_input(
                        label="Enter your City or Zip Code:",
                        placeholder="e.g., 'New Delhi' or '221002'"
                    )
                    find_doctor_button = st.button("Find a Doctor", type="primary", use_container_width=True)

                with col2:
                    st.subheader("✨ Recommendations")
                    confirmation_placeholder = st.empty()
                    if st.session_state.booking_confirmation:
                        confirmation_placeholder.success(st.session_state.booking_confirmation)
                        st.session_state.booking_confirmation = None

                    with st.container(height=800):

                        if find_doctor_button:
                            if not user_symptoms or not user_location:
                                st.warning("Please select at least one symptom or location.")
                            else:
                                st.session_state.prediction = None
                                st.session_state.specialist = None
                                st.session_state.doctors_list = []
                                st.session_state.map_data_list = []

                                with st.status("Finding recommendations...", expanded=True) as status:
                                    try:
                                        st.write("Analyzing your symptoms...")
                                        time.sleep(1)
                                        model_input = {symptom: 0 for symptom in symptoms}
                                        for symptom in user_symptoms:
                                            if symptom in model_input:
                                                model_input[symptom] = 1
                                        input_df = pd.DataFrame([model_input])
                                        st.session_state.prediction = model.predict(input_df)[0]

                                        st.write("Identifying the right specialist...")
                                        specialty_map = {
                                            'Fungal infection': 'Dermatologist', 'Allergy': 'Allergist',
                                            'GERD': 'Gastroenterologist',
                                            'Acne': 'Dermatologist', 'Pneumonia': 'Pulmonologist',
                                            'Jaundice': 'Gastroenterologist',
                                            'Migraine': 'Neurologist', 'Hypertension ': 'Cardiologist',
                                            'Heart attack': 'Cardiologist',
                                            'Paralysis (brain hemorrhage)': 'Neurologist',
                                            'Chicken pox': 'Dermatologist',
                                            'Malaria': 'General Practitioner', 'Dengue': 'General Practitioner',
                                            'Typhoid': 'General Practitioner'
                                        }
                                        st.session_state.specialist = specialty_map.get(st.session_state.prediction,
                                                                                        'General Practitioner')

                                        st.write(
                                            f"Searching for {st.session_state.specialist}s near {user_location}...")
                                        query = f"{st.session_state.specialist} in {user_location}"
                                        places_result = gmaps.places(query=query, type='doctor')
                                        st.session_state.doctors_list = places_result.get('results', [])[:5]

                                        temp_map_data = []
                                        for doctor in st.session_state.doctors_list:
                                            place_id = doctor['place_id']
                                            fields = ['name', 'formatted_address', 'international_phone_number',
                                                      'website', 'rating', 'opening_hours', 'geometry']
                                            place_details = gmaps.place(place_id=place_id, fields=fields)
                                            details = place_details.get('result', {})
                                            doctor['details'] = details
                                            if 'geometry' in details:
                                                location = details['geometry']['location']
                                                temp_map_data.append(
                                                    {'name': details.get('name', 'N/A'), 'lat': location['lat'],
                                                     'lon': location['lng']})
                                        st.session_state.map_data_list = temp_map_data
                                        status.update(label="Analysis Complete!", state="complete", expanded=False)

                                    except Exception as e:
                                        status.update(label="Error processing request", state="error")
                                        st.error(f"An error occurred: {e}")

                        if st.session_state.prediction:
                            st.success(f"**Predicted Condition:** {st.session_state.prediction}")
                            if find_doctor_button: st.balloons()
                            st.markdown(f"### Recommended Specialist: **{st.session_state.specialist}**")
                            st.divider()
                            st.subheader(f"Top 5 {st.session_state.specialist}s near you:")

                            if not st.session_state.doctors_list:
                                st.warning("No doctors found matching your criteria.")
                            else:
                                for doctor in st.session_state.doctors_list:
                                    details = doctor.get('details', {})
                                    place_id = doctor['place_id']
                                    name = details.get('name', 'N/A')
                                    address = details.get('formatted_address', 'Address not available')
                                    phone = details.get('international_phone_number', 'Phone not available')
                                    website = details.get('website', None)
                                    rating = details.get('rating', 'N/A')

                                    open_now = "Status unknown"
                                    if 'opening_hours' in details:
                                        open_now = "🟢 Open now" if details['opening_hours'].get('open_now',
                                                                                                False) else "🔴 Closed"

                                    with st.container(border=True):
                                        st.markdown(f"#### {name}")
                                        st.write(f"**{rating}** ⭐ | {open_now}")
                                        st.write(f"📍 **Address:** {address}")
                                        st.write(f"📞 **Phone:** {phone}")

                                        col_btn1, col_btn2, col_btn3 = st.columns(3)

                                        with col_btn1:
                                            if website:
                                                st.link_button("Visit Website 🌐", url=website, use_container_width=True)
                                            else:
                                                st.button("Website N/A", disabled=True, use_container_width=True,
                                                          key=f"web_na_{place_id}")
                                        with col_btn2:
                                            gmaps_url = f"https://www.google.com/maps/search/?api=1&query={name.replace(' ', '+')}&query_place_id={place_id}"
                                            st.link_button("View on Map 🗺️", url=gmaps_url, use_container_width=True)

                                        with col_btn3:
                                            with st.popover("Book Appointment", use_container_width=True):
                                                with st.form(key=f"book_form_{place_id}", clear_on_submit=True):
                                                    st.markdown(f"**Book with {name}**")
                                                    patient_name_input = st.text_input("Your Full Name",
                                                                                       key=f"name_{place_id}")
                                                    appt_date = st.date_input("Preferred Date",
                                                                              min_value=datetime.date.today(),
                                                                              key=f"date_{place_id}")
                                                    appt_time = st.time_input("Preferred Time", key=f"time_{place_id}")

                                                    submitted = st.form_submit_button("Request Appointment")

                                                    if submitted:
                                                        if not patient_name_input:
                                                            st.warning("Please enter your name.")
                                                        else:
                                                            try:
                                                                doc_ref = db.collection("appointments").document()
                                                                doc_ref.set({
                                                                    "patient_email": st.session_state.user['email'],
                                                                    "patient_id": st.session_state.user['localId'],
                                                                    "patient_name": patient_name_input,
                                                                    "doctor_name": name,
                                                                    "doctor_place_id": place_id,
                                                                    "appointment_date": str(appt_date),
                                                                    "appointment_time": str(appt_time),
                                                                    "status": "Pending"
                                                                })
                                                                st.session_state.booking_confirmation = \
                                                                    f"✅ Success! Your appointment request for {name} has been sent."
                                                            except Exception as e:
                                                                st.session_state.booking_confirmation = \
                                                                    f"❌ Error: Could not book appointment. {e}"
                                                            st.rerun()

                                    st.write("")

                            if st.session_state.map_data_list:
                                st.divider()
                                st.subheader("Doctor Locations:")
                                map_df = pd.DataFrame(st.session_state.map_data_list)
                                st.map(map_df, latitude='lat', longitude='lon', size=10, zoom=12)

                        elif not st.session_state.prediction:
                            st.info("Your results will appear here.")

        # --- TAB 2: MY APPOINTMENTS ---
        with tab2:
            st.subheader(f"Your Appointment Status")

            if not db:
                st.error("Database client could not be initialized. Cannot check appointments.")
            else:
                try:
                    user_id = st.session_state.user['localId']
                    appointments_ref = db.collection("appointments") \
                        .where("patient_id", "==", user_id) \
                        .stream()

                    appointments = list(appointments_ref)

                    if not appointments:
                        st.warning(f"You have no appointments. Find a doctor to get started!")
                    else:
                        st.success(f"Found {len(appointments)} appointments:")

                        col1, col2, col3, col4 = st.columns([3, 2, 2, 1.5])
                        col1.write("**Doctor**")
                        col2.write("**Date**")
                        col3.write("**Time**")
                        col4.write("**Status**")
                        st.divider()

                        for appt in appointments:
                            appt_data = appt.to_dict()
                            col1, col2, col3, col4 = st.columns([3, 2, 2, 1.5])

                            col1.write(appt_data.get("doctor_name"))
                            col2.write(appt_data.get("appointment_date"))
                            col3.write(appt_data.get("appointment_time"))

                            status = appt_data.get("status", "Pending")
                            if status == "Pending":
                                col4.warning(status)
                            elif status == "Accepted":
                                col4.success(status)
                            elif status == "Declined":
                                col4.error(status)

                except Exception as e:
                    st.error(f"An error occurred while fetching appointments: {e}")