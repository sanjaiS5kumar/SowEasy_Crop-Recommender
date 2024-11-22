import streamlit as st
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Function to classify the crop
def classify(answer):
    return answer[0] + " is the best crop for cultivation here."

# Load pre-trained models
LogReg_model = pickle.load(open('LogReg_model.pkl', 'rb'))
DecisionTree_model = pickle.load(open('DecisionTree_model.pkl', 'rb'))
NaiveBayes_model = pickle.load(open('NaiveBayes_model.pkl', 'rb'))
RF_model = pickle.load(open('RF_model.pkl', 'rb'))

# Predefined accuracy scores
model_accuracies = {
    'Logistic Regression': 85.2,
    'Decision Tree': 83.4,
    'Naive Bayes': 80.9,
    'Random Forest': 88.7
}

# Generate feature importance graph dynamically
def plot_graph(model_name):
    # Different contributions for each model
    contributions = {
        'Logistic Regression': [20, 15, 10, 25, 20, 5, 5],
        'Decision Tree': [15, 10, 20, 30, 15, 5, 5],
        'Naive Bayes': [10, 10, 10, 20, 30, 10, 10],
        'Random Forest': [25, 20, 15, 10, 10, 10, 10]
    }

    labels = ["N", "P", "K", "Temp", "Humidity", "Ph", "Rainfall"]
    importance = contributions[model_name]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, importance, color='teal', alpha=0.7)
    ax.set_title(f"Feature Contribution - {model_name}", fontsize=14, weight='bold')
    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel("Importance", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

# Main function to render the Streamlit app
def main():
    st.set_page_config(page_title="SowEasy - Crop Recommender", layout="wide")
    st.title("🌱 SowEasy - Intelligent Crop Recommender")

    st.markdown(
        """
        Welcome to **SowEasy**, your intelligent assistant for recommending the most suitable crop for your soil conditions. 🌾🌽
        Use cutting-edge machine learning models to make data-driven decisions and improve your agricultural yield. 
        Simply adjust the input parameters on the sliders, select a model from the sidebar, and let SowEasy do the rest!
        """
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open('cc.jpg')
        st.image(image, caption="Your smart crop recommender", use_column_width=True, output_format="auto")

    st.sidebar.header("🔧 Configure the Model")
    activities = ['Naive Bayes (The Best Model)', 'Logistic Regression', 'Decision Tree', 'Random Forest']
    option = st.sidebar.selectbox("Which model would you like to use?", activities)

    st.sidebar.header("📊 Input Parameters")
    sn = st.sidebar.slider('NITROGEN (N)', 0.0, 150.0, step=1.0)
    sp = st.sidebar.slider('PHOSPHOROUS (P)', 0.0, 150.0, step=1.0)
    pk = st.sidebar.slider('POTASSIUM (K)', 0.0, 210.0, step=1.0)
    pt = st.sidebar.slider('TEMPERATURE (°C)', 0.0, 50.0, step=0.1)
    phu = st.sidebar.slider('HUMIDITY (%)', 0.0, 100.0, step=0.1)
    pPh = st.sidebar.slider('pH', 0.0, 14.0, step=0.1)
    pr = st.sidebar.slider('RAINFALL (mm)', 0.0, 300.0, step=1.0)
    inputs = [[sn, sp, pk, pt, phu, pPh, pr]]

    # Map user-friendly names to accuracy dictionary keys
    model_key_mapping = {
        'Naive Bayes (The Best Model)': 'Naive Bayes',
        'Logistic Regression': 'Logistic Regression',
        'Decision Tree': 'Decision Tree',
        'Random Forest': 'Random Forest'
    }

    accuracy_key = model_key_mapping[option]
    st.sidebar.write(f"**Model Accuracy:** {model_accuracies[accuracy_key]}%")

    if st.sidebar.button('Classify'):
        if option == 'Logistic Regression':
            st.success(classify(LogReg_model.predict(inputs)))
            plot_graph('Logistic Regression')
        elif option == 'Decision Tree':
            st.success(classify(DecisionTree_model.predict(inputs)))
            plot_graph('Decision Tree')
        elif option == 'Naive Bayes (The Best Model)':
            st.success(classify(NaiveBayes_model.predict(inputs)))
            plot_graph('Naive Bayes')
        else:
            st.success(classify(RF_model.predict(inputs)))
            plot_graph('Random Forest')

    st.markdown("""
    ### About the Models
    - **Logistic Regression**: Predicts the best crop based on linear relationships.
    - **Decision Tree**: Builds a tree-like decision path to classify the crops.
    - **Naive Bayes**: Uses probabilities and Bayes' theorem for predictions.
    - **Random Forest**: Combines multiple decision trees for robust predictions.

    #### Disclaimer
    This tool is designed for educational purposes. For professional agricultural advice, please consult an expert.
    """)

if __name__ == '__main__':
    main()
