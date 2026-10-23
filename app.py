
import os
import joblib
import pandas as pd
import streamlit as st

import shap
import matplotlib.pyplot as plt

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="FairCredit AI",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "models/faircredit_model.joblib"

# =====================================================
# 2. CUSTOM STYLING
# =====================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1120, #101b2d);
        color: #e5edf7;
    }

    [data-testid="stSidebar"] {
        background: #101827;
        border-right: 1px solid #29394e;
    }

    .hero {
        padding: 28px;
        border: 1px solid #2b4056;
        border-radius: 20px;
        background: linear-gradient(120deg, #17283b, #123b43);
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 36px;
        font-weight: 800;
        color: #f1f7ff;
    }

    .hero-subtitle {
        color: #a9c1d5;
        font-size: 14px;
        letter-spacing: 1px;
    }

    div[data-testid="stMetric"] {
        background: #152438;
        border: 1px solid #2a4056;
        padding: 18px;
        border-radius: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a9c1d5;
    }

    div[data-testid="stMetricValue"] {
        color: #f1f7ff;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: linear-gradient(90deg, #14b8a6, #0d9488);
        color: white;
        border: none;
        border-radius: 10px;
        min-height: 45px;
        font-weight: 700;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #0f766e;
        color: white;
        border: 1px solid #5eead4;
    }

    .result-panel {
        background: #152438;
        padding: 20px;
        border: 1px solid #2a4056;
        border-radius: 16px;
    }

    .small-note {
        color: #9eb0c4;
        font-size: 12px;
    }

    hr {
        border-color: #29394e;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD THE SAVED MODEL
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found. Run "
        "`python src/train.py` from the project root."
    )
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# =====================================================
# 4. HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">💠 FairCredit AI</div>
    <div class="hero-subtitle">
        CREDIT RISK INTELLIGENCE · PREDICTIVE ANALYTICS
    </div>
    <br>
    Estimate credit default probability with machine learning.
</div>
""", unsafe_allow_html=True)


# =====================================================
# 5. SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## 💠 FairCredit AI")
    st.caption("RISK INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Risk Assessment", "About the Model"]
    )

    st.divider()
    st.markdown("### Model Information")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Test ROC-AUC:** 0.758")
    st.write("**Training records:** 24,000")
    st.write("**Test records:** 6,000")

    st.divider()
    st.caption("Predictive Analytics Project")


# =====================================================
# 6. ABOUT PAGE
# =====================================================
if page == "About the Model":
    st.title("About FairCredit AI")

    st.write("""
    FairCredit AI estimates the probability of credit card payment
    default using a trained Logistic Regression model.
    """)

    st.subheader("Model workflow")

    st.markdown("""
    1. Collect customer account information.
    2. Encode categorical features.
    3. Standardize numerical features.
    4. Estimate the probability of default.
    5. Display the prediction and probability.
    """)

    st.subheader("Evaluation")
    st.metric("Test ROC-AUC", "0.758")

    st.info(
        "This is an educational prototype. Model estimates can be "
        "wrong and should not be used alone to make lending decisions."
    )

    st.stop()


# =====================================================
# 7. CUSTOMER INPUT FORM
# =====================================================
st.subheader("01 | Customer Profile")
st.caption(
    "Enter the account information and repayment history below."
)

repayment_options = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

with st.form("risk_form"):

    st.markdown("#### Personal and credit details")

    col1, col2, col3 = st.columns(3)

    with col1:
        limit_bal = st.number_input(
            "Credit limit",
            min_value=0,
            max_value=10_000_000,
            value=50_000,
            step=10_000
        )

    with col2:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    with col3:
        sex = st.selectbox("SEX code", [1, 2])

    col4, col5 = st.columns(2)

    with col4:
        education = st.selectbox(
            "EDUCATION code",
            [1, 2, 3, 4, 5, 6, 0]
        )

    with col5:
        marriage = st.selectbox(
            "MARRIAGE code",
            [1, 2, 3, 0]
        )

    st.divider()
    st.markdown("#### Repayment history")

    st.caption(
        "Use the original repayment status codes from the dataset."
    )

    pay_col1, pay_col2, pay_col3 = st.columns(3)

    with pay_col1:
        pay_0 = st.selectbox(
            "PAY_0 · Most recent",
            repayment_options,
            index=2
        )
        pay_2 = st.selectbox(
            "PAY_2",
            repayment_options,
            index=2
        )

    with pay_col2:
        pay_3 = st.selectbox(
            "PAY_3",
            repayment_options,
            index=2
        )
        pay_4 = st.selectbox(
            "PAY_4",
            repayment_options,
            index=2
        )

    with pay_col3:
        pay_5 = st.selectbox(
            "PAY_5",
            repayment_options,
            index=2
        )
        pay_6 = st.selectbox(
            "PAY_6",
            repayment_options,
            index=2
        )

    st.divider()
    st.markdown("#### Bill amounts")

    bill_col1, bill_col2, bill_col3 = st.columns(3)

    with bill_col1:
        bill_1 = st.number_input("BILL_AMT1", value=20_000)
        bill_2 = st.number_input("BILL_AMT2", value=18_000)

    with bill_col2:
        bill_3 = st.number_input("BILL_AMT3", value=15_000)
        bill_4 = st.number_input("BILL_AMT4", value=12_000)

    with bill_col3:
        bill_5 = st.number_input("BILL_AMT5", value=10_000)
        bill_6 = st.number_input("BILL_AMT6", value=8_000)

    st.divider()
    st.markdown("#### Previous payment amounts")

    payment_col1, payment_col2, payment_col3 = st.columns(3)

    with payment_col1:
        payment_1 = st.number_input(
            "PAY_AMT1", min_value=0, value=2_000
        )
        payment_2 = st.number_input(
            "PAY_AMT2", min_value=0, value=2_000
        )

    with payment_col2:
        payment_3 = st.number_input(
            "PAY_AMT3", min_value=0, value=1_500
        )
        payment_4 = st.number_input(
            "PAY_AMT4", min_value=0, value=1_500
        )

    with payment_col3:
        payment_5 = st.number_input(
            "PAY_AMT5", min_value=0, value=1_000
        )
        payment_6 = st.number_input(
            "PAY_AMT6", min_value=0, value=1_000
        )

    submitted = st.form_submit_button(
        "✦ Calculate Default Risk",
        use_container_width=True
    )


# =====================================================
# 8. BUILD CUSTOMER INPUT
# =====================================================
def create_customer(
    limit_bal, sex, education, marriage, age,
    pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
    bill_1, bill_2, bill_3, bill_4, bill_5, bill_6,
    payment_1, payment_2, payment_3,
    payment_4, payment_5, payment_6
):
    return pd.DataFrame([{
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,
        "BILL_AMT1": bill_1,
        "BILL_AMT2": bill_2,
        "BILL_AMT3": bill_3,
        "BILL_AMT4": bill_4,
        "BILL_AMT5": bill_5,
        "BILL_AMT6": bill_6,
        "PAY_AMT1": payment_1,
        "PAY_AMT2": payment_2,
        "PAY_AMT3": payment_3,
        "PAY_AMT4": payment_4,
        "PAY_AMT5": payment_5,
        "PAY_AMT6": payment_6
    }])


# =====================================================
# 9. PREDICTION RESULTS
# =====================================================
if submitted:

    customer = create_customer(
        limit_bal, sex, education, marriage, age,
        pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
        bill_1, bill_2, bill_3, bill_4, bill_5, bill_6,
        payment_1, payment_2, payment_3,
        payment_4, payment_5, payment_6
    )

    # Keep the submitted customer available when Streamlit reruns
    # after the SHAP button is clicked.
    st.session_state["shap_customer"] = customer.copy()

    try:
        probability = float(
            model.predict_proba(customer)[0, 1]
        )
        predicted_class = int(model.predict(customer)[0])

        st.divider()
        st.subheader("02 | Risk Assessment")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Estimated default probability",
                f"{probability:.1%}"
            )

        with metric2:
            st.metric(
                "Model classification",
                "Default" if predicted_class == 1 else "No default"
            )

        with metric3:
            st.metric(
                "Classification threshold",
                "50%"
            )

        st.markdown("#### Probability scale")
        st.progress(probability)

        st.caption(
            "The probability is model-estimated. The 50% threshold "
            "is a classification setting, not a recommended lending threshold."
        )

        if predicted_class == 1:
            st.warning(
                "The model assigns this input to the default class."
            )
        else:
            st.success(
                "The model assigns this input to the non-default class."
            )

        # ---------------------------------------------
        # 10. FIXED CUSTOMER SUMMARY CARDS
        # ---------------------------------------------
        st.divider()
        st.subheader("Customer Input Summary")

        summary1, summary2, summary3 = st.columns(3)

        with summary1:
            st.metric(
                label="Credit Limit",
                value=f"{limit_bal:,.0f}"
            )

        with summary2:
            st.metric(
                label="Age",
                value=f"{age} years"
            )

        with summary3:
            st.metric(
                label="Recent Repayment Status",
                value=str(pay_0)
            )

        # ---------------------------------------------
        # 11. INTERACTIVE SENSITIVITY CHECK
        # ---------------------------------------------
        st.divider()
        st.subheader("03 | Interactive Sensitivity Check")

        st.write(
            "Change one input to see how the estimated probability "
            "changes while the other values remain constant."
        )

        feature_choice = st.selectbox(
            "Choose a feature to adjust",
            [
                "Credit limit",
                "Most recent repayment status",
                "Age"
            ]
        )

        changed_customer = customer.copy()

        if feature_choice == "Credit limit":
            changed_value = st.number_input(
                "Alternative credit limit",
                min_value=0,
                max_value=10_000_000,
                value=min(limit_bal + 10_000, 10_000_000),
                step=10_000,
                key="alternative_limit"
            )
            changed_customer.loc[0, "LIMIT_BAL"] = changed_value

        elif feature_choice == "Most recent repayment status":
            changed_value = st.selectbox(
                "Alternative PAY_0",
                repayment_options,
                index=repayment_options.index(pay_0),
                key="alternative_pay0"
            )
            changed_customer.loc[0, "PAY_0"] = changed_value

        else:
            changed_value = st.slider(
                "Alternative age",
                min_value=18,
                max_value=100,
                value=int(age),
                key="alternative_age"
            )
            changed_customer.loc[0, "AGE"] = changed_value

        changed_probability = float(
            model.predict_proba(changed_customer)[0, 1]
        )

        difference = changed_probability - probability

        comparison1, comparison2, comparison3 = st.columns(3)

        with comparison1:
            st.metric(
                "Original probability",
                f"{probability:.1%}"
            )

        with comparison2:
            st.metric(
                "Adjusted probability",
                f"{changed_probability:.1%}"
            )

        with comparison3:
            st.metric(
                "Probability change",
                f"{difference:+.1%}"
            )

        st.caption(
            "This is a one-input sensitivity check, not a causal "
            "effect or a formal feature-importance explanation."
        )

        with st.expander("View all model input values"):
            st.dataframe(
                customer.T.rename(columns={0: "Value"}),
                use_container_width=True
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that the saved model and input feature names match."
        )


# =====================================================
# 12. SHAP EXPLANATION
# =====================================================
st.divider()
st.subheader("04 | Why did the model make this prediction?")

st.write(
    "SHAP estimates how individual features contribute to the "
    "model's prediction relative to a background dataset."
)

saved_customer = st.session_state.get("shap_customer")

if saved_customer is None:
    st.info("First enter the customer details and click 'Calculate Default Risk'. Then generate the SHAP explanation.")

if st.button("Generate SHAP Explanation", disabled=(saved_customer is None)):
    with st.spinner("Calculating feature contributions..."):
        try:
            # Use the customer saved in session state because Streamlit
            # reruns the script when this button is clicked.
            customer = saved_customer.copy()

            # Transform the customer using the fitted preprocessor
            fitted_preprocessor = model.named_steps["preprocessing"]
            fitted_classifier = model.named_steps["model"]

            customer_processed = fitted_preprocessor.transform(customer)

            # Load a small sample of training data as the background
            training_df = pd.read_excel(
                "data/default of credit card clients.xls",
                header=1
            )

            training_df = training_df.rename(
                columns={"default payment next month": "DEFAULT"}
            )

            X_background = training_df.drop(
                columns=["DEFAULT", "ID"]
            ).sample(n=100, random_state=42)

            background_processed = fitted_preprocessor.transform(
                X_background
            )

            # Convert sparse matrices to dense arrays for SHAP
            if hasattr(background_processed, "toarray"):
                background_processed = background_processed.toarray()

            if hasattr(customer_processed, "toarray"):
                customer_processed = customer_processed.toarray()

            # Explain the classifier's log-odds output
            explainer = shap.LinearExplainer(
                fitted_classifier,
                background_processed
            )

            # Use the current SHAP explanation API
            shap_explanation = explainer(customer_processed)
            shap_values = shap_explanation.values

            # Get transformed feature names
            feature_names = fitted_preprocessor.get_feature_names_out()

            # Logistic Regression binary output is normally (rows, features).
            # Handle a possible extra class dimension defensively.
            if shap_values.ndim == 3:
                values = shap_values[0, :, 1]
            else:
                values = shap_values[0]

            explanation_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP value": values
            })

            explanation_df["Absolute impact"] = (
                explanation_df["SHAP value"].abs()
            )

            explanation_df = explanation_df.sort_values(
                "Absolute impact",
                ascending=False
            ).head(10)

            
            # Display the most influential features
            st.markdown("#### Top 10 contributing features")

            # Convert technical column names into readable labels
            def make_readable_feature(name):
                name = name.replace("numerical__", "").replace("categorical__", "")
                friendly = {
                    "LIMIT_BAL": "Credit limit",
                    "AGE": "Age",
                    "SEX": "Sex category",
                    "EDUCATION": "Education category",
                    "MARRIAGE": "Marriage category",
                    "PAY_0": "Most recent repayment status",
                    "PAY_2": "Repayment status (2 months ago)",
                    "PAY_3": "Repayment status (3 months ago)",
                    "PAY_4": "Repayment status (4 months ago)",
                    "PAY_5": "Repayment status (5 months ago)",
                    "PAY_6": "Repayment status (6 months ago)",
                    "BILL_AMT1": "Most recent bill amount",
                    "BILL_AMT2": "Bill amount (2 months ago)",
                    "BILL_AMT3": "Bill amount (3 months ago)",
                    "BILL_AMT4": "Bill amount (4 months ago)",
                    "BILL_AMT5": "Bill amount (5 months ago)",
                    "BILL_AMT6": "Bill amount (6 months ago)",
                    "PAY_AMT1": "Most recent payment amount",
                    "PAY_AMT2": "Payment amount (2 months ago)",
                    "PAY_AMT3": "Payment amount (3 months ago)",
                    "PAY_AMT4": "Payment amount (4 months ago)",
                    "PAY_AMT5": "Payment amount (5 months ago)",
                    "PAY_AMT6": "Payment amount (6 months ago)",
                }

                # One-hot encoded columns end with the category code.
                for original, label in friendly.items():
                    if name == original:
                        return label
                    if name.startswith(original + "_"):
                        category = name[len(original) + 1:]
                        return f"{label} = {category}"

                return name.replace("_", " ")

            explanation_df["Readable feature"] = explanation_df["Feature"].apply(
                make_readable_feature
            )

            # Plot the largest contributions as a horizontal bar chart.
            # Positive bars push toward default; negative bars push away from default.
            plot_df = explanation_df.sort_values(
                "Absolute impact", ascending=True
            ).copy()

            fig_height = max(5.5, 0.55 * len(plot_df) + 1.5)
            fig, ax = plt.subplots(figsize=(12, fig_height))

            bar_colors = [
                "#f97373" if value > 0 else "#38bdf8"
                for value in plot_df["SHAP value"]
            ]

            ax.barh(
                plot_df["Readable feature"],
                plot_df["SHAP value"],
                color=bar_colors,
                edgecolor="none",
                height=0.68,
            )
            ax.axvline(0, color="#cbd5e1", linewidth=1)
            ax.set_title(
                "How each feature shifts the model output",
                fontsize=15,
                fontweight="bold",
                pad=14,
            )
            ax.set_xlabel("SHAP value (log-odds contribution)", fontsize=11)
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelsize=10, pad=8)
            ax.tick_params(axis="x", labelsize=10)
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            ax.set_axisbelow(True)
            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            
            st.markdown("""
            🔴 **Positive SHAP value:** pushes the model output toward default

            🔵 **Negative SHAP value:** pushes the model output toward non-default
            """)

            st.caption(
                "SHAP values explain contributions to the model's log-odds, "
                "not direct percentage-point changes in default probability."
            )


            with st.expander("View detailed SHAP values"):
                st.dataframe(
                    plot_df[
                        ["Readable feature", "SHAP value", "Absolute impact"]
                    ].rename(columns={
                        "Readable feature": "Feature"
                    }).sort_values(
                        "Absolute impact", ascending=False
                    ).reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as error:
            st.error(f"Could not generate SHAP explanation: {error}")
