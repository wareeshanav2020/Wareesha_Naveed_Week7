# Mall Customer Segmentation Using K-Means

## 1. Project Overview

This project implements a machine learning application that groups mall customers into different segments based on their annual income and spending score.

The project uses the K-Means clustering algorithm to identify groups of customers with similar characteristics. A Streamlit web application allows users to enter customer information and obtain a predicted cluster and its corresponding customer segment.

## 2. Project Objectives

- Understand and explore the customer dataset.
- Prepare the data for machine learning.
- Apply K-Means clustering to identify customer segments.
- Evaluate the clustering results using suitable evaluation techniques.
- Save the trained machine learning pipeline.
- Develop a Streamlit application for interactive customer segmentation.
- Test the application using multiple customer inputs.

## 3. Input Features

The application uses two input features:

1. **Annual Income (k$):** The customer's annual income in thousands of dollars. For example, an input of 50 represents $50,000.
2. **Spending Score (1–100):** A numerical score representing the customer's spending behaviour.

## 4. Machine Learning Model

The project uses K-Means clustering with five clusters.

The saved pipeline shown during application testing contains:

- `StandardScaler` for feature scaling.
- `KMeans` with `n_clusters=5`, `n_init=10`, and `random_state=42`.

The saved model package contains the pipeline and metadata used by the application, including feature names, segment names, and training ranges.

## 5. Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Excel for application testing documentation

## 6. Project Files

- `Mall_Customer_Segmentation.ipynb` — Executed Jupyter Notebook containing the machine learning workflow.
- `customer_segmentation_model.pkl` — Saved trained model package.
- `app.py` — Streamlit application source code.
- `README.md` — Project documentation.
- `Technical_Report.pdf` — Short technical report.
- `Application_Testing.xlsx` — Application testing table containing test inputs and prediction results.
- `screenshots/` — Folder containing screenshots of the application and its test results.

Include the processed dataset and cluster-labelled dataset if they are required by the assignment instructions.

## 7. Installation Requirements

Install Python and the required packages. Open a terminal in the project folder and run:

```bash
pip install streamlit pandas scikit-learn joblib openpyxl
```

The installed scikit-learn version should be compatible with the version used to save the model.

## 8. How to Run the Application

1. Place `app.py` and `customer_segmentation_model.pkl` in the same folder.
2. Open a terminal in that folder.
3. Run the following command:

```bash
streamlit run app.py
```

4. Open the local URL displayed in the terminal.
5. Enter the customer's annual income in thousands of dollars.
6. Enter the spending score between 1 and 100.
7. Click **Predict Customer Segment**.
8. Review the predicted cluster ID and customer segment.

## 9. Application Testing

Three sample inputs were previously tested:

| Test Case | Annual Income (k$) | Spending Score | Previously Observed Cluster |
|---|---:|---:|---:|
| Test 1 | 15 | 20 | 4 |
| Test 2 | 50 | 50 | 0 |
| Test 3 | 100 | 80 | 1 |

These are previously observed results. Run all three test cases again after the final application changes and record the actual outputs in `Application_Testing.xlsx`.

## 10. Limitations

- K-Means assigns cluster IDs that do not inherently indicate customer value.
- Customer segment names must accurately reflect the characteristics of each cluster.
- Predictions outside the training data ranges may be less reliable.
- The model identifies similarities in customer data and does not guarantee future customer behaviour.

## 11. Conclusion

This project demonstrates an end-to-end machine learning workflow, from data preparation and clustering to saving the model and deploying an interactive Streamlit application.

The application enables users to explore customer segments using annual income and spending score. The final model and segment-name mapping should be verified before submission.