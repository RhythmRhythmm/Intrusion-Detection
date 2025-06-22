from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import os

# Suppress all warnings
warnings.filterwarnings("ignore")

# Get the absolute path of the directory containing app.py
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_FOLDER_PATH = os.path.join(BASE_DIR, 'static')
DATA_FOLDER_PATH = os.path.join(BASE_DIR, 'data')


# Initialize Flask app with explicit static and template folder paths
app = Flask(
    __name__,
    static_folder=STATIC_FOLDER_PATH,
    template_folder=STATIC_FOLDER_PATH # index.html is in static
)

# --- Global Variables for Data and Models ---
df = None # DataFrame for EDA (may be sampled)
rf_model = None
lr_model = None
X_test_global = None # X_test for evaluation
y_test_global = None # y_test for evaluation
feature_names = None
label_encoder = None

# Define a maximum number of rows to load to prevent MemoryError
# The original dataset has ~2.8 million rows. Loading ~200k-500k should be sufficient for demonstration.
MAX_ROWS_TO_LOAD = 200000 # Adjust this value based on available RAM if needed

def load_and_preprocess_data():
    """
    Loads a sample of the CICIDS2017 dataset, preprocesses it,
    and returns training features/labels and global test features/labels.
    """
    global df, X_test_global, y_test_global, feature_names, label_encoder

    try:
        csv_file_path = os.path.join(DATA_FOLDER_PATH, 'MachineLearningCSV.csv')
        
        # Load a limited number of rows from the dataset to manage memory
        df_raw = pd.read_csv(csv_file_path, encoding='latin1', nrows=MAX_ROWS_TO_LOAD)
        df = df_raw.copy() # Use df for EDA plots later

        # Clean column names: remove spaces, special chars, ensure string type
        df.columns = [str(col).strip().replace(' ', '_').replace('/', '_').replace('-', '_').replace('.', '_') for col in df.columns]

        # Handle 'Label' column specific character issue (if present)
        if 'Label' in df.columns:
            df['Label'] = df['Label'].astype(str).str.replace('\x96', '-', regex=False).str.strip()

        # Identify all columns that contain string/object data type, excluding 'Label'
        object_columns = df.select_dtypes(include='object').columns.tolist()
        if 'Label' in object_columns:
            object_columns.remove('Label')

        # Drop these identified object columns that are not 'Label'
        if object_columns:
            df.drop(columns=object_columns, inplace=True)
            print(f"Dropped all non-numeric feature columns: {object_columns}")

        # Convert all remaining columns (except 'Label') to numeric.
        # Any values that cannot be converted will become NaN.
        for col in df.columns:
            if col != 'Label':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Replace infinite values (which may appear after numeric conversion of large numbers) with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows with any NaN values (from inf replacement or numeric coercion)
        df.dropna(inplace=True)

        # Remove duplicate rows (after all cleaning and NaN handling)
        df.drop_duplicates(inplace=True)

        # Separate features (X) and target (y)
        X = df.drop('Label', axis=1)
        y = df['Label'] # y is now the cleaned Label series

        # Encode the target variable (y)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y) # y_encoded is now the numerical target

        # Scale features
        scaler = StandardScaler()
        X_scaled_columns = X.columns
        X[X_scaled_columns] = scaler.fit_transform(X[X_scaled_columns])

        feature_names = X.columns.tolist()

        # Perform the main train-test split for the loaded and preprocessed data
        # X_train_full and y_train_full will be passed to train_models
        # X_test_global and y_test_global will be stored globally for evaluation
        X_train_full, X_test_global, y_train_full, y_test_global = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )

        print("Data loaded and preprocessed successfully.")
        print(f"Dataset shape after preprocessing (sampled): {df.shape}")
        print(f"Number of features after preprocessing: {X.shape[1]}")
        print(f"Unique labels (encoded): {np.unique(y_test_global)}")
        print(f"Label mapping: {list(label_encoder.classes_)}")
        
        return X_train_full, y_train_full # Return training data for models

    except FileNotFoundError as fnf_e:
        print(f"Critical Error: {fnf_e}. Please ensure '{os.path.basename(csv_file_path)}' is in the '{os.path.dirname(csv_file_path)}' directory.")
        raise RuntimeError(f"Data file not found: {fnf_e}")
    except MemoryError as mem_e:
        print(f"Memory Error: {mem_e}. The dataset might still be too large for your RAM even with nrows={MAX_ROWS_TO_LOAD}.")
        print("Consider reducing MAX_ROWS_TO_LOAD or using a more memory-efficient approach if this persists.")
        raise RuntimeError(f"Memory allocation failed during data loading: {mem_e}")
    except Exception as e:
        print(f"An unexpected error occurred during data loading and preprocessing: {e}")
        raise RuntimeError(f"Data preprocessing failed: {e}")

def plot_to_base64(plt_figure):
    """Converts a matplotlib figure to a base64 encoded PNG image."""
    buf = io.BytesIO()
    plt_figure.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(plt_figure)
    return img_base64

def setup():
    """Runs once to load data and train models."""
    global rf_model, lr_model
    # Get training data from preprocessing step
    X_train_for_models, y_train_for_models = load_and_preprocess_data()
    print("Training models...")
    train_models(X_train_for_models, y_train_for_models)
    print("Setup complete.")

def train_models(X_train_data, y_train_data):
    """Trains RandomForest and LogisticRegression models using provided training data."""
    global rf_model, lr_model

    if len(X_train_data) == 0 or len(y_train_data) == 0:
        print("Warning: Training data is empty. Models will not be trained.")
        return

    # RandomForest Classifier
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_data, y_train_data)
    print("RandomForest model trained.")

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
    lr_model.fit(X_train_data, y_train_data)
    print("LogisticRegression model trained.")


@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/eda_plots')
def get_eda_plots():
    """Generates and returns EDA plots as base64 images."""
    if df is None:
        return jsonify({"error": "Data not loaded. Please restart the server."}), 500

    plots = {}

    # 1. Distribution of 'Label' (Attack types)
    fig_label, ax_label = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, y='Label', ax=ax_label, palette='viridis')
    ax_label.set_title('Distribution of Attack Types (Labels)')
    ax_label.set_xlabel('Count')
    ax_label.set_ylabel('Attack Type')
    plots['label_distribution'] = plot_to_base64(fig_label)

    # 2. Correlation Heatmap (Top 10 most correlated features with 'Label')
    numeric_df = df.select_dtypes(include=np.number)
    if 'Label' in df.columns:
        if df['Label'].dtype == 'object':
            temp_df = numeric_df.copy()
            if label_encoder is None:
                raise RuntimeError("LabelEncoder not initialized. Data preprocessing failed.")
            temp_df['Label_encoded'] = label_encoder.transform(df['Label'])
            correlations = temp_df.corr()['Label_encoded'].abs().sort_values(ascending=False)
            top_features = [col for col in correlations.index if col != 'Label_encoded'][:10] # Top 10 excluding Label itself
            if 'Label_encoded' not in top_features: # Ensure Label_encoded is included for heatmap row/col if relevant
                top_features.append('Label_encoded')
            corr_matrix = temp_df[top_features].corr()
        else:
            correlations = numeric_df.corr()['Label'].abs().sort_values(ascending=False)
            top_features = [col for col in correlations.index if col != 'Label'][:10]
            if 'Label' not in top_features:
                top_features.append('Label')
            corr_matrix = numeric_df[top_features].corr()
    else:
        corr_matrix = numeric_df.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
        top_features = corr_matrix.index[:10].get_level_values(0).tolist()
        corr_matrix = numeric_df[top_features].corr()


    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
    ax_corr.set_title('Correlation Heatmap (Top 10 Features with Label)')
    plots['correlation_heatmap'] = plot_to_base64(fig_corr)

    # 3. Histograms for a few important features
    selected_features_for_hist = ['Flow_Duration', 'Total_Length_of_Fwd_Packets', 'Subflow_Fwd_Bytes']
    for feature in selected_features_for_hist:
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
            fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
            sns.histplot(df[feature], bins=50, kde=True, ax=ax_hist, color='skyblue')
            ax_hist.set_title(f'Distribution of {feature}')
            ax_hist.set_xlabel(feature)
            ax_hist.set_ylabel('Frequency')
            plots[f'hist_{feature}'] = plot_to_base64(fig_hist)
        else:
            print(f"Warning: Feature '{feature}' not found or not numeric for histogram plotting.")

    return jsonify(plots)

@app.route('/model_results')
def get_model_results():
    """Calculates and returns evaluation metrics and confusion matrices for trained models."""
    global rf_model, lr_model, X_test_global, y_test_global, label_encoder

    if rf_model is None or lr_model is None or X_test_global is None or y_test_global is None:
        return jsonify({"error": "Models not trained or data not loaded. Please restart the server."}), 500

    results = {}
    models = {
        "RandomForest": rf_model,
        "LogisticRegression": lr_model
    }

    for name, model in models.items():
        y_pred = model.predict(X_test_global)

        accuracy = accuracy_score(y_test_global, y_pred)
        precision = precision_score(y_test_global, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_global, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_global, y_pred, average='weighted', zero_division=0)

        # Generate Confusion Matrix
        cm = confusion_matrix(y_test_global, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        
        unique_labels = np.unique(np.concatenate((y_test_global, y_pred)))
        if label_encoder is not None:
            # Get all original classes, then filter to only those present in unique_labels
            target_names = [label_encoder.inverse_transform([lbl])[0] for lbl in sorted(unique_labels)]
        else:
            target_names = [f'Class {i}' for i in sorted(unique_labels)]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=target_names, yticklabels=target_names)
        ax_cm.set_title(f'Confusion Matrix - {name}')
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        cm_base64 = plot_to_base64(fig_cm)

        results[name] = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix_plot": cm_base64
        }
    return jsonify(results)

if __name__ == '__main__':
    try:
        setup() # Call the setup function directly here
        port = int(os.environ.get('PORT', 5000)) # For deployment flexibility
        app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False) # <--- ADDED use_reloader=False
    except RuntimeError as e:
        print(f"Application failed to start due to setup error: {e}")
