import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ Mall Customer Segmentation")

st.write(
    "Enter a customer's annual income and spending score "
    "to predict their customer segment using the trained K-Means model."
)

# --------------------------------------------------
# 2. Load the saved model
# --------------------------------------------------
MODEL_PATH = Path(__file__).parent / "customer_segmentation_model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    saved_object = joblib.load(MODEL_PATH)

    if not isinstance(saved_object, dict):
        raise ValueError(
            "The saved model must be a dictionary containing "
            "the pipeline and model metadata."
        )

    required_keys = [
        "pipeline",
        "features",
        "segment_names",
        "training_ranges"
    ]

    missing_keys = [
        key for key in required_keys
        if key not in saved_object
    ]

    if missing_keys:
        raise ValueError(
            f"Model file is missing required keys: {missing_keys}"
        )

    return saved_object


try:
    model_data = load_model()

    pipeline = model_data["pipeline"]
    features = model_data["features"]
    segment_names = model_data["segment_names"]
    training_ranges = model_data["training_ranges"]

except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.info(
        "Make sure customer_segmentation_model.pkl is in the "
        "same folder as app.py and was saved by your Week 7 notebook."
    )
    st.stop()


# --------------------------------------------------
# 3. Validate model metadata
# --------------------------------------------------
income_feature = "Annual Income (k$)"
spending_feature = "Spending Score (1-100)"

if income_feature not in features or spending_feature not in features:
    st.error(
        "The model features do not match the expected input fields. "
        f"Saved model features: {features}"
    )
    st.stop()


def get_range(feature_name):
    """Read a feature's minimum and maximum training values."""

    feature_range = training_ranges[feature_name]

    if isinstance(feature_range, dict):
        min_value = feature_range.get(
            "min", feature_range.get("minimum")
        )
        max_value = feature_range.get(
            "max", feature_range.get("maximum")
        )

        if min_value is None or max_value is None:
            raise ValueError(
                f"Cannot identify minimum and maximum for "
                f"{feature_name}: {feature_range}"
            )

    elif isinstance(feature_range, (list, tuple)) and len(feature_range) >= 2:
        min_value, max_value = feature_range[0], feature_range[1]

    else:
        raise ValueError(
            f"Unsupported training range for {feature_name}: "
            f"{feature_range}"
        )

    return float(min_value), float(max_value)


try:
    income_min, income_max = get_range(income_feature)
    spending_min, spending_max = get_range(spending_feature)

except Exception as error:
    st.error(f"Could not read training ranges: {error}")
    st.stop()


# --------------------------------------------------
# 4. Customer input form
# --------------------------------------------------
st.subheader("Customer Information")

with st.form("customer_form"):

    annual_income = st.number_input(
        "Annual Income (in thousands of dollars)",
        min_value=0.0,
        value=50.0,
        step=1.0,
        help="For example, enter 50 for an annual income of $50,000."
    )

    spending_score = st.number_input(
        "Spending Score (1–100)",
        min_value=1,
        max_value=100,
        value=50,
        step=1
    )

    submitted = st.form_submit_button(
        "Predict Customer Segment",
        type="primary"
    )

# --------------------------------------------------
# 5. Predict the customer segment
# --------------------------------------------------
if submitted:

    # Validate the inputs against the training ranges.
    if not income_min <= annual_income <= income_max:
        st.warning(
            f"The training income range was "
            f"{income_min:.2f} to {income_max:.2f} "
            "(in thousands of dollars). "
            "The prediction may be less reliable outside this range."
        )

    if not spending_min <= spending_score <= spending_max:
        st.warning(
            f"The training spending score range was "
            f"{spending_min:.2f} to {spending_max:.2f}. "
            "The prediction may be less reliable outside this range."
        )

    # Prepare customer data in the correct feature order.
    customer = pd.DataFrame(
        [{
            income_feature: annual_income,
            spending_feature: spending_score
        }],
        columns=features
    )

    try:
        # Predict the customer cluster.
        prediction = pipeline.predict(customer)
        cluster_id = int(prediction[0])

        # Get the corresponding customer segment name.
        segment_name = segment_names.get(
            cluster_id,
            segment_names.get(
                str(cluster_id),
                f"Cluster {cluster_id}"
            )
        )

        # Display a clean prediction result.
        st.success("Prediction completed successfully!")

        st.divider()

        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Cluster ID",
            value=cluster_id
        )

        st.markdown(f"### 🛍️ {segment_name}")

        st.write("**Customer Information**")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Annual Income",
                value=f"${annual_income:,.2f}k"
            )

        with col2:
            st.metric(
                label="Spending Score",
                value=f"{spending_score}/100"
            )

        st.caption(
            "K-Means groups customers based on similarity in "
            "annual income and spending score. Predictions do not "
            "guarantee future customer behavior."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# --------------------------------------------------
# 6. Model information
# --------------------------------------------------
with st.expander("About this application"):

    st.write(
        "This application uses a saved preprocessing and K-Means "
        "pipeline trained on annual income and spending score."
    )

    st.write(f"Model features: {features}")

    st.write(
        f"Number of clusters: "
        f"{model_data.get('n_clusters', 'Not recorded')}"
    )
