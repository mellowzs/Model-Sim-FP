import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📈",
    layout="wide")

def preprocess_data(data):
    data = pd.get_dummies(data, columns=['Day'], drop_first=True)  # Keep day-related information
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[['Discount (%)', 'Marketing Spend ($)', 'Competitor Price ($)']])
    scaled_df = pd.DataFrame(scaled_features, columns=['Discount (%)', 'Marketing Spend ($)', 'Competitor Price ($)'])
    data[['Discount (%)', 'Marketing Spend ($)', 'Competitor Price ($)']] = scaled_df
    st.write("**Scaled Data:**")
    st.write(data.head())
    return data

def generate_synthetic_data(n_samples, avg_marketing_spend, std_marketing_spend, min_discount, max_discount, competitor_price_min, competitor_price_max):
    np.random.seed(42)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day = np.random.choice(days, n_samples)
    discount = np.random.uniform(min_discount, max_discount, n_samples)
    marketing_spend = np.random.normal(avg_marketing_spend, std_marketing_spend, n_samples).clip(min=0)
    competitor_price = np.random.uniform(competitor_price_min, competitor_price_max, n_samples)

    #sales count generation logic
    base_sales = 50 + 0.02 * marketing_spend - 0.5 * discount - 0.01 * competitor_price
    sales_count = (base_sales + np.random.normal(0, 5, n_samples)).clip(min=0).astype(int)

    return pd.DataFrame({
        'Day': day,
        'Discount (%)': discount,
        'Marketing Spend ($)': marketing_spend,
        'Competitor Price ($)': competitor_price,
        'Sales Count': sales_count
    })

def tune_model(X, y, model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

# Evaluation Metrics
def display_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    
    st.write(f"**{model_name} Evaluation Metrics**")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    
    st.write(f"""
    **Implications for {model_name}:**
    - **MSE**: Lower values indicate better performance.
    - **R²**: Higher values (close to 1) indicate the model explains more variance in the data.
    - **MAE**: Lower values indicate less average error between actual and predicted values.
    """)
    return mse, r2, mae

def perform_eda(data):
    colu1, colu2 = st.columns(2, gap='medium', border=True)
    with colu1:
      # Sales Count Distribution
      st.subheader("Sales Count Distribution")
      fig, ax = plt.subplots()
      sns.histplot(data['Sales Count'], bins=20, kde=True, ax=ax)
      ax.set_title("Sales Count Distribution")
      ax.set_xlabel("Sales Count")
      ax.set_ylabel("Frequency")
      st.pyplot(fig)
    with colu1:
      st.write("""
      
      - **X-axis (`Sales Count`)**: Represents the number of items sold in a given scenario.
      - **Y-axis (`Frequency`)**: Shows how often a particular sales count appears in the dataset.
      - The histogram, combined with the KDE curve, indicates that the distribution of Sales Count is approximately normal, with the highest frequency around a sales count of 55. This suggests that most sales fall within this range, indicating a central tendency. 
      - **Peaks in the KDE Curve:** Represent the most common sales counts. Flat regions on either side of the peak indicate fewer occurrences of very low or very high sales counts.
      """)

      with colu2:
         # Scatter Plot: Sales Count vs Marketing Spend
         st.subheader("Sales Count vs Marketing Spend")
         fig, ax = plt.subplots()
         sns.scatterplot(x=data['Marketing Spend ($)'], y=data['Sales Count'], hue=data['Day'], palette="viridis", ax=ax)
         ax.set_title("Sales Count vs Marketing Spend")
         ax.set_xlabel("Marketing Spend ($)")
         ax.set_ylabel("Sales Count")
         st.pyplot(fig)
      with colu2:
         st.write("""
         
         - **X-axis (`Marketing Spend ($)`)**: Represents the amount spent on marketing in dollars, ranging from 2000 to $9000.
         - **Y-axis (`Sales Count`)**: Represents the number of items sold.
         - **Legend (`Day`)**: Indicates the day of the week corresponding to the data points. Each color represents a specific day.
         - The scatter plot shows the relationship between `Marketing Spend` and `Sales Count`. A visible positive trend suggests that higher marketing spend tends to result in more sales.
         - The color-coded points allow us to observe if certain days have stronger relationships between marketing spend and sales. For example, weekends might show higher sales with lower marketing spend.
         """)
    with st.container(border=True):
      map1, map2 = st.columns(2)
      # Correlation Heatmap
      with map1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = data.select_dtypes(include=[np.number]).corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
            ax.set_title("Feature Correlation Heatmap")
            st.pyplot(fig)
      with map2:
            st.write("""
            - **X-axis & Y-axis**: Represent features of the dataset (e.g., `Marketing Spend`, `Discount (%)`, etc.).
            - **Cell Values**: Show the correlation coefficient between two features.
               - Values range from -1 to 1:
               - `1`: Perfect positive correlation (as one feature increases, the other increases).
               - `0`: No correlation.
               - `-1`: Perfect negative correlation (as one feature increases, the other decreases).
               - Highly correlated features (e.g., `Marketing Spend` and `Sales Count`) suggest strong relationships, which could guide business decisions. For instance, focusing on `Marketing Spend` could optimize sales outcomes.
               """)

# Streamlit App Setup
st.title("📊 Sales Analytics & Prediction Dashboard")
st.sidebar.header("Input Parameters")
n_samples = st.sidebar.number_input("Number of Samples", min_value=100, value=10000)
avg_marketing_spend = st.sidebar.number_input("Average Marketing Spend ($)", value=10000)
std_marketing_spend = st.sidebar.number_input("Standard Deviation of Marketing Spend ($)", value=4000)
# Add warning if std is too high relative to mean
if std_marketing_spend > avg_marketing_spend/2:
    st.sidebar.warning("⚠️ High standard deviation may generate unrealistic values. Consider reducing it below 50% of the average.")
min_discount = st.sidebar.slider("Minimum Discount (%)", min_value=0, max_value=30, value=3)
max_discount = st.sidebar.slider("Maximum Discount (%)", min_value=10, max_value=50, value=10)
competitor_price_min = st.sidebar.number_input("Competitor Price Min ($)", value=1000)  
competitor_price_max = st.sidebar.number_input("Competitor Price Max ($)", value=5000)
model_choice = st.sidebar.selectbox("Select Model", ["SVR", "Decision Tree", "KNN", "Gradient Boosting", "XGBoost"])

# Generate synthetic data
data = generate_synthetic_data(n_samples, avg_marketing_spend, std_marketing_spend, min_discount, max_discount, competitor_price_min, competitor_price_max)
st.write("Generated Data:")
st.write(data.head())

# Perform EDA
st.header("Exploratory Data Analysis")
perform_eda(data)

# Preprocess the data
data = preprocess_data(data)
X = data.drop(columns=['Sales Count'])
y = data['Sales Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and tuning
if model_choice == "SVR":
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['rbf']}
    model = SVR()
    best_model, best_params = tune_model(X_train, y_train, model, param_grid)
elif model_choice == "Decision Tree":
    param_grid = {'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10]}
    model = DecisionTreeRegressor(random_state=42)
    best_model, best_params = tune_model(X_train, y_train, model, param_grid)
elif model_choice == "KNN":
    param_grid = {'n_neighbors': [3, 5, 10]}
    model = KNeighborsRegressor()
    best_model, best_params = tune_model(X_train, y_train, model, param_grid)
elif model_choice == "Gradient Boosting":
    param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300]}
    model = GradientBoostingRegressor(random_state=42)
    best_model, best_params = tune_model(X_train, y_train, model, param_grid)
elif model_choice == "XGBoost":
    param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 6, 10], 'n_estimators': [100, 200, 300]}
    model = XGBRegressor(random_state=42)
    best_model, best_params = tune_model(X_train, y_train, model, param_grid)

with st.container(border=True):
    st.header("Model Training and Evaluation")

    # Train the best model
    y_pred = best_model.predict(X_test)

    # Display metrics and interpretations
    col1, col2 = st.columns(2)
    with col2:
        st.write(f"### {model_choice} Metrics")
        mse, r2, mae = display_metrics(y_test, y_pred, model_choice)
    
    with col1:
        # Scatter Plot: Actual vs Predicted Sales
        st.subheader("Actual vs Predicted Sales")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, label=model_choice, color='blue')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal Prediction")
        ax.set_title("Actual vs Predicted Sales")
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        ax.legend()
        st.pyplot(fig)
    
    # Model-specific interpretations
        if model_choice == "SVR":
            st.write("""
            #### Interpretation:
            - **Scattered points**: Show how well SVR captures non-linear relationships
            - **Red line**: Perfect prediction reference
            - **Spread**: Wider spread indicates lower prediction accuracy
            - **Clusters**: Dense areas show common sales patterns
            
            #### Business Implications:
            - Best for: Complex, non-linear sales patterns
            - Consider kernel adjustments if predictions are too rigid/flexible
            """)
        
        elif model_choice == "Decision Tree":
            st.write("""
            #### Interpretation:
            - **Step-like patterns**: Show discrete prediction levels
            - **Vertical groups**: Indicate similar predictions for different actuals
            - **Gaps**: Represent decision boundaries in the tree
            
            #### Business Implications:
            - Best for: Clear decision rules and segment-based predictions
            - Useful for identifying distinct sales categories
            """)
            
        elif model_choice == "KNN":
            st.write("""
            #### Interpretation:
            - **Point clusters**: Show similar sales patterns
            - **Outliers**: Points far from red line need attention
            - **Density**: Areas with many points show common scenarios
            
            #### Business Implications:
            - Best for: Sales predictions based on similar historical patterns
            - Adjust k-value if predictions are too general/specific
            """)
            
        elif model_choice == "Gradient Boosting":
            st.write("""
            #### Interpretation:
            - **Tight clustering**: Shows high prediction confidence
            - **Gradient pattern**: Progressive improvement in predictions
            - **Outlier handling**: Better with extreme values
            
            #### Business Implications:
            - Best for: Complex sales patterns with many features
            - Robust against outliers and noise
            """)
            
        elif model_choice == "XGBoost":
            st.write("""
            #### Interpretation:
            - **Dense diagonal**: Indicates high accuracy
            - **Outlier predictions**: Usually more accurate than other models
            - **Consistent spread**: Shows reliable predictions across ranges
            
            #### Business Implications:
            - Best for: High-stakes sales forecasting
            - Superior handling of complex patterns
            """)

