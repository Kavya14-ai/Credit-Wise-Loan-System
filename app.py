
import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score


# PAGE SETTINGS
st.set_page_config(
    page_title="Credit Wise Loan System",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Credit Wise Loan System")
st.write(
    "Enter applicant details to get a machine-learning prediction. "
    "This is an educational prediction tool, not a bank's final loan decision."
)

DATA_FILE = "loan_approval_data.csv"


# FEATURE ENGINEERING
def add_engineered_features(data):
    data = data.copy()

    data["Education_Level"] = data["Education_Level"].map(
        {"Graduate": 0, "Not Graduate": 1}
    )

    data["DTI_Ratio_sq"] = data["DTI_Ratio"] ** 2
    data["Credit_Score_sq"] = data["Credit_Score"] ** 2

    data = data.drop(columns=["DTI_Ratio", "Credit_Score"])

    return data


# MODEL TRAINING
@st.cache_resource
def train_model():

    data = pd.read_csv(DATA_FILE)

    data = data.dropna(subset=["Loan_Approved"]).copy()

    data = data.drop(columns=["Applicant_ID"], errors="ignore")

    y = data["Loan_Approved"].map({"No": 0, "Yes": 1})

    X = data.drop(columns=["Loan_Approved"])

    X = add_engineered_features(X)

    numeric_features = [
        "Applicant_Income",
        "Coapplicant_Income",
        "Age",
        "Dependents",
        "Existing_Loans",
        "Savings",
        "Collateral_Value",
        "Loan_Amount",
        "Loan_Term",
        "DTI_Ratio_sq",
        "Credit_Score_sq",
        "Education_Level"
    ]

    categorical_features = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category"
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ])

    # SPLIT DATASET
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # TRAIN MODEL
    model.fit(X_train, y_train)

    # MODEL ACCURACY
    accuracy = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    return model, accuracy, len(data)


# LOAD AND TRAIN MODEL
try:

    model, accuracy, training_rows = train_model()

except FileNotFoundError:

    st.error(
        "Dataset not found. Keep app.py and "
        "loan_approval_data.csv in the same project folder."
    )

    st.stop()

except Exception as error:

    st.error(f"Could not train the model: {error}")

    st.stop()


# SIDEBAR
with st.sidebar:

    st.header("Model Information")

    st.metric(
        "Test Accuracy",
        f"{accuracy:.1%}"
    )

    st.write(
        f"Training records with known outcomes: {training_rows}"
    )

    st.caption(
        "The app trains the Naive Bayes model from the CSV when it starts."
    )


# APPLICANT FORM
st.subheader("Applicant Information")

with st.form("loan_application"):

    col1, col2, col3 = st.columns(3)

    with col1:

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=25000.0,
            step=1000.0
        )

        coapplicant_income = st.number_input(
            "Co-applicant Income",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=20,
            value=0
        )

        employment_status = st.selectbox(
            "Employment Status",
            ["Salaried", "Self-employed", "Contract", "Unemployed"]
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married"]
        )

    with col2:

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=700
        )

        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            max_value=30,
            value=0
        )

        dti_ratio = st.number_input(
            "DTI Ratio",
            min_value=0.0,
            max_value=10.0,
            value=0.30,
            step=0.01,
            format="%.2f"
        )

        savings = st.number_input(
            "Savings",
            min_value=0.0,
            value=20000.0,
            step=1000.0
        )

        collateral_value = st.number_input(
            "Collateral Value",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        education_level = st.selectbox(
            "Education Level",
            ["Graduate", "Not Graduate"]
        )

    with col3:

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=30000.0,
            step=1000.0
        )

        loan_term = st.number_input(
            "Loan Term (Months)",
            min_value=1,
            max_value=600,
            value=60
        )

        loan_purpose = st.selectbox(
            "Loan Purpose",
            ["Personal", "Home", "Car", "Education", "Business"]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        employer_category = st.selectbox(
            "Employer Category",
            ["Private", "Government", "MNC", "Business", "Unemployed"]
        )

    submitted = st.form_submit_button(
        "Predict Loan Outcome",
        type="primary"
    )


# PREDICTION
if submitted:

    applicant = pd.DataFrame([{

        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Employment_Status": employment_status,
        "Marital_Status": marital_status,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category

    }])

    processed_applicant = add_engineered_features(applicant)

    prediction = int(
        model.predict(processed_applicant)[0]
    )

    probabilities = model.predict_proba(
        processed_applicant
    )[0]

    approval_probability = float(probabilities[1])

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            "The model predicts: Loan likely to be approved"
        )

    else:

        st.warning(
            "The model predicts: Loan likely to be rejected"
        )

    st.metric(
        "Estimated Approval Probability",
        f"{approval_probability:.1%}"
    )

    st.caption(
        "This is an educational prediction from the provided dataset, "
        "not a bank's final loan decision."
    )
