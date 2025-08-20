from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import pickle
import os
import json
from werkzeug.utils import secure_filename
import warnings
import traceback
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("=== UPLOAD FUNCTION STARTED ===")
    print(f"Request method: {request.method}")
    print(f"Request files: {request.files}")
    print(f"Request form: {request.form}")
    
    if 'file' not in request.files:
        print("ERROR: No file in request.files")
        return jsonify({'error': 'No file selected'})
    
    file = request.files['file']
    dataset_type = request.form.get('dataset_type')
    
    print(f"File object: {file}")
    print(f"File filename: {file.filename}")
    print(f"Dataset type: {dataset_type}")
    
    if file.filename == '':
        print("ERROR: Empty filename")
        return jsonify({'error': 'No file selected'})
    
    print(f"File extension check: {file.filename.endswith('.csv')} | {file.filename.endswith('.xlsx')} | {file.filename.endswith('.xls')}")
    
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        print(f"Secure filename: {filename}")
        print(f"Full filepath: {filepath}")
        print(f"Upload folder exists: {os.path.exists(app.config['UPLOAD_FOLDER'])}")
        
        try:
            file.save(filepath)
            print(f"File saved successfully to: {filepath}")
            print(f"File exists after save: {os.path.exists(filepath)}")
            print(f"File size: {os.path.getsize(filepath)} bytes")
        except Exception as save_error:
            print(f"ERROR saving file: {str(save_error)}")
            print(f"Save error traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Error saving file: {str(save_error)}'})
        
        # Load and analyze the dataset
        try:
            print("=== STARTING FILE READING ===")
            
            # Handle both CSV and Excel files
            if filename.endswith('.csv'):
                print("Reading as CSV file...")
                df = pd.read_csv(filepath)
                print("CSV file loaded successfully")
            elif filename.endswith('.xlsx'):
                print("Reading as XLSX file...")
                df = pd.read_excel(filepath, engine='openpyxl')
                print("XLSX file loaded successfully")
            elif filename.endswith('.xls'):
                print("Reading as XLS file...")
                df = pd.read_excel(filepath, engine='xlrd')
                print("XLS file loaded successfully")
            
            print(f"Dataset shape: {df.shape}")
            print(f"Dataset columns: {df.columns.tolist()}")
            print(f"Dataset dtypes:\n{df.dtypes}")
            print(f"First row data:\n{df.head(1)}")
            print(f"Dataset info:")
            df.info()
            
            # Store dataset info in session
            print("=== STORING SESSION DATA ===")
            session[f'{dataset_type}_data'] = filepath
            session[f'{dataset_type}_columns'] = df.columns.tolist()
            print(f"Session data stored for {dataset_type}")
            
            # Get basic info
            print("=== PREPARING RESPONSE DATA ===")
            # Replace NaN values with None (which becomes null in JSON)
            df_head = df.head().fillna('')  # Replace NaN with empty string for display
            head_data = df_head.to_dict('records')
            print(f"Head data prepared: {len(head_data)} records")
            print(f"Sample head data: {head_data[0] if head_data else 'No data'}")
            
            # Analyze columns
            categorical_cols = []
            numerical_cols = []
            
            for col in df.columns:
                print(f"Column '{col}' has dtype: {df[col].dtype}")
                if df[col].dtype in ['object', 'category']:
                    categorical_cols.append(col)
                    print(f"  -> Added to categorical")
                else:
                    numerical_cols.append(col)
                    print(f"  -> Added to numerical")
            
            print(f"Categorical columns: {categorical_cols}")
            print(f"Numerical columns: {numerical_cols}")
            
            # Check for missing values
            print("=== CHECKING MISSING VALUES ===")
            missing_info = df.isnull().sum().to_dict()
            # Convert numpy int64 to regular int to avoid JSON serialization issues
            missing_cols = {k: int(v) for k, v in missing_info.items() if v > 0}
            print(f"Missing values info: {missing_cols}")
            
            response_data = {
                'success': True,
                'head_data': head_data,
                'columns': df.columns.tolist(),
                'categorical_cols': categorical_cols,
                'numerical_cols': numerical_cols,
                'missing_values': missing_cols,
                'total_rows': int(len(df)),  # Ensure it's a regular int
                'dataset_type': dataset_type
            }
            
            print("=== RESPONSE DATA PREPARED ===")
            print(f"Response keys: {response_data.keys()}")
            print(f"Total rows: {response_data['total_rows']}")
            
            # Test JSON serialization before returning
            try:
                import json
                json_test = json.dumps(response_data)
                print("JSON serialization test: PASSED")
            except Exception as json_error:
                print(f"JSON serialization test: FAILED - {json_error}")
                # If JSON serialization fails, try to fix the data
                print("Attempting to fix JSON serialization issues...")
                
                # Convert any remaining problematic values
                def clean_for_json(obj):
                    if isinstance(obj, dict):
                        return {k: clean_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [clean_for_json(v) for v in obj]
                    elif pd.isna(obj):
                        return None
                    elif isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                    else:
                        return obj
                
                response_data = clean_for_json(response_data)
                print("Data cleaned for JSON serialization")
            
            print("=== RETURNING SUCCESS RESPONSE ===")
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"=== ERROR PROCESSING FILE ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            return jsonify({'error': f'Error processing file: {str(e)}. Please check if the file is not corrupted and contains valid data.'})
    
    print("ERROR: Invalid file format")
    return jsonify({'error': 'Invalid file format. Please upload a CSV (.csv) or Excel (.xlsx, .xls) file.'})

@app.route('/fix_missing', methods=['POST'])
def fix_missing_values():
    print("=== FIX MISSING VALUES STARTED ===")
    dataset_type = request.json.get('dataset_type')
    filepath = session.get(f'{dataset_type}_data')
    
    print(f"Dataset type: {dataset_type}")
    print(f"Filepath: {filepath}")
    
    if not filepath or not os.path.exists(filepath):
        print("ERROR: Dataset file not found")
        return jsonify({'error': 'Dataset not found'})
    
    try:
        print("Loading dataset...")
        # Handle both CSV and Excel files based on file extension
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            print("CSV file loaded for fixing missing values")
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath, engine='openpyxl')
            print("XLSX file loaded for fixing missing values")
        elif filepath.endswith('.xls'):
            df = pd.read_excel(filepath, engine='xlrd')
            print("XLS file loaded for fixing missing values")
        else:
            return jsonify({'error': 'Unsupported file format'})
        
        print(f"Dataset loaded. Shape: {df.shape}")
        original_missing = df.isnull().sum().to_dict()
        fixes_applied = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if df[col].dtype in ['object', 'category']:
                    # Fill categorical with mode
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
                    fixes_applied[col] = f"Filled {missing_count} missing values with mode: '{mode_val}'"
                else:
                    # Fill numerical with median
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    fixes_applied[col] = f"Filled {missing_count} missing values with median: {median_val:.2f}"
        
        # Save cleaned dataset
        print("Saving cleaned dataset...")
        if filepath.endswith('.csv'):
            cleaned_filepath = filepath.replace('.csv', '_cleaned.csv')
            df.to_csv(cleaned_filepath, index=False)
        elif filepath.endswith('.xlsx'):
            cleaned_filepath = filepath.replace('.xlsx', '_cleaned.xlsx')
            df.to_excel(cleaned_filepath, index=False, engine='openpyxl')
        elif filepath.endswith('.xls'):
            # Convert .xls to .xlsx for saving (since xlrd doesn't support writing)
            cleaned_filepath = filepath.replace('.xls', '_cleaned.xlsx')
            df.to_excel(cleaned_filepath, index=False, engine='openpyxl')
        
        session[f'{dataset_type}_cleaned_data'] = cleaned_filepath
        print(f"Cleaned dataset saved to: {cleaned_filepath}")
        
        return jsonify({
            'success': True,
            'fixes_applied': fixes_applied,
            'message': f'Fixed missing values in {len(fixes_applied)} columns'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error fixing missing values: {str(e)}'})

@app.route('/train_models', methods=['POST'])
def train_models():
    print("=== TRAIN MODELS STARTED ===")
    dataset_type = request.json.get('dataset_type')
    cleaned_filepath = session.get(f'{dataset_type}_cleaned_data')
    
    print(f"Dataset type: {dataset_type}")
    print(f"Cleaned filepath: {cleaned_filepath}")
    
    if not cleaned_filepath or not os.path.exists(cleaned_filepath):
        print("ERROR: Cleaned dataset not found")
        return jsonify({'error': 'Cleaned dataset not found'})
    
    try:
        print("Loading cleaned dataset...")
        # Handle both CSV and Excel files based on file extension
        if cleaned_filepath.endswith('.csv'):
            df = pd.read_csv(cleaned_filepath)
            print("CSV file loaded for training")
        elif cleaned_filepath.endswith('.xlsx'):
            df = pd.read_excel(cleaned_filepath, engine='openpyxl')
            print("XLSX file loaded for training")
        else:
            return jsonify({'error': 'Unsupported cleaned file format'})
        
        print(f"Dataset loaded for training. Shape: {df.shape}")
        print(f"Available columns: {df.columns.tolist()}")
        
        # Check if RESULT column exists (case-insensitive)
        result_column = None
        possible_result_columns = ['RESULT', 'Result', 'result']
        
        for col in df.columns:
            if col in possible_result_columns:
                result_column = col
                print(f"Found target column: {result_column}")
                break
            # Also check for columns containing 'result' (case-insensitive)
            elif 'result' in col.lower():
                result_column = col
                print(f"Found target column containing 'result': {result_column}")
                break
        
        if result_column is None:
            available_cols = ', '.join(df.columns.tolist())
            return jsonify({
                'error': f'Target column not found. Looking for columns like RESULT, Result, result, TARGET, etc. Available columns: {available_cols}'
            })
        
        print(f"Using '{result_column}' as target column")
        print(f"Target column unique values: {df[result_column].unique()}")
        print(f"Target column value counts:\n{df[result_column].value_counts()}")
        
        # Prepare features and target
        X = df.drop([result_column], axis=1)
        y = df[result_column]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature columns: {X.columns.tolist()}")
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        print(f"Categorical columns to encode: {categorical_cols.tolist()}")
        
        for col in categorical_cols:
            print(f"Encoding column '{col}' with unique values: {X[col].unique()}")
            X[col] = pd.Categorical(X[col]).codes
            print(f"After encoding, '{col}' has values: {X[col].unique()}")
        
        # Check for any remaining non-numeric columns
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            print(f"Warning: Non-numeric columns still present: {non_numeric_cols.tolist()}")
            # Try to convert them to numeric
            for col in non_numeric_cols:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    print(f"Converted '{col}' to numeric")
                except:
                    print(f"Could not convert '{col}' to numeric, dropping it")
                    X = X.drop([col], axis=1)
        
        print(f"Final feature columns: {X.columns.tolist()}")
        print(f"Final features shape: {X.shape}")
        
        # Check for NaN values in features and handle them
        print("=== CHECKING FOR NaN VALUES IN FEATURES ===")
        nan_counts = X.isnull().sum()
        nan_columns = nan_counts[nan_counts > 0]
        
        if len(nan_columns) > 0:
            print(f"Found NaN values in features:")
            for col, count in nan_columns.items():
                print(f"  {col}: {count} NaN values")
            
            print("Filling NaN values in features...")
            # Fill NaN values in numeric columns with median
            numeric_cols = X.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if X[col].isnull().any():
                    median_val = X[col].median()
                    if pd.isna(median_val):  # If median is also NaN, use 0
                        median_val = 0
                    X[col].fillna(median_val, inplace=True)
                    print(f"  Filled NaN in '{col}' with median: {median_val}")
            
            # Double-check for any remaining NaN values
            remaining_nans = X.isnull().sum().sum()
            if remaining_nans > 0:
                print(f"Warning: {remaining_nans} NaN values still remain. Filling with 0.")
                X.fillna(0, inplace=True)
            
            print("All NaN values in features have been handled.")
        else:
            print("No NaN values found in features.")
        
        # Final validation - ensure no NaN values remain
        final_nan_check = X.isnull().sum().sum()
        if final_nan_check > 0:
            print(f"ERROR: Still have {final_nan_check} NaN values after cleaning!")
            return jsonify({'error': f'Could not clean all NaN values from features. {final_nan_check} NaN values remain.'})
        
        print("✓ Features are clean and ready for training")
        
        # Additional validation - check for infinite values
        inf_check = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        if inf_check > 0:
            print(f"Found {inf_check} infinite values, replacing with 0")
            X.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Check data types
        print(f"Feature data types:\n{X.dtypes}")
        
        # Ensure all features are numeric
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                print(f"Warning: Column '{col}' is not numeric: {X[col].dtype}")
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col].fillna(0, inplace=True)
                    print(f"Converted '{col}' to numeric")
                except:
                    print(f"Could not convert '{col}' to numeric, dropping it")
                    X = X.drop([col], axis=1)
        
        # Check if we have any features left
        if X.shape[1] == 0:
            return jsonify({'error': 'No valid features found after preprocessing'})
        
        # Check if target has valid values
        if y.isnull().all():
            return jsonify({'error': f'Target column "{result_column}" contains only null values'})
        
        # Remove rows with null target values
        valid_indices = ~y.isnull()
        if not valid_indices.all():
            print(f"Removing {(~valid_indices).sum()} rows with null target values")
            X = X[valid_indices]
            y = y[valid_indices]
            print(f"After removing null targets - Features: {X.shape}, Target: {y.shape}")
        
        # Check if we have enough data
        if len(y) < 10:
            return jsonify({'error': f'Not enough data for training. Only {len(y)} valid samples found.'})
        
        print(f"Target value distribution:\n{y.value_counts()}")
        
        # Final data summary before training
        print("=== FINAL DATA SUMMARY BEFORE TRAINING ===")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Features data types: {X.dtypes.value_counts()}")
        print(f"Any NaN in features: {X.isnull().any().any()}")
        print(f"Any NaN in target: {y.isnull().any()}")
        print(f"Any infinite values in features: {np.isinf(X.select_dtypes(include=[np.number])).any().any()}")
        print(f"Feature value ranges:")
        for col in X.columns[:5]:  # Show first 5 columns
            print(f"  {col}: min={X[col].min():.3f}, max={X[col].max():.3f}, mean={X[col].mean():.3f}")
        if X.shape[1] > 5:
            print(f"  ... and {X.shape[1] - 5} more columns")
        
        # Split the data
        try:
            # Try stratified split first
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            print("Used stratified split")
        except ValueError as e:
            print(f"Stratified split failed: {e}")
            print("Using regular split without stratification")
            # If stratified split fails (e.g., due to class imbalance), use regular split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training set - Features: {X_train.shape}, Target: {y_train.shape}")
        print(f"Test set - Features: {X_test.shape}, Target: {y_test.shape}")
        print(f"Training target distribution:\n{y_train.value_counts()}")
        print(f"Test target distribution:\n{y_test.value_counts()}")
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        results = {}
        trained_models = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # ROC AUC (for binary classification)
            try:
                if len(np.unique(y)) == 2 and y_pred_proba is not None:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
                else:
                    roc_auc = None
                    roc_data = None
            except:
                roc_auc = None
                roc_data = None
            
            results[name] = {
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'roc_data': roc_data
            }
            
            trained_models[name] = model
        
        # Save models and feature columns
        models_dir = f'models_{dataset_type}'
        os.makedirs(models_dir, exist_ok=True)
        
        for name, model in trained_models.items():
            model_filename = os.path.join(models_dir, f'{name.lower().replace(" ", "_")}.pkl')
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
        
        # Save feature columns for prediction
        feature_columns_file = os.path.join(models_dir, 'feature_columns.json')
        with open(feature_columns_file, 'w') as f:
            json.dump(X.columns.tolist(), f)
        
        session[f'{dataset_type}_models_dir'] = models_dir
        session[f'{dataset_type}_feature_columns'] = X.columns.tolist()
        
        return jsonify({
            'success': True,
            'results': results,
            'feature_columns': X.columns.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': f'Error training models: {str(e)}'})

@app.route('/predict', methods=['POST'])
def predict():
    dataset_type = request.json.get('dataset_type')
    model_name = request.json.get('model_name')
    input_data = request.json.get('input_data')
    
    models_dir = session.get(f'{dataset_type}_models_dir')
    feature_columns = session.get(f'{dataset_type}_feature_columns')
    
    if not models_dir or not feature_columns:
        return jsonify({'error': 'Models not trained yet'})
    
    try:
        # Load the model
        model_filename = os.path.join(models_dir, f'{model_name.lower().replace(" ", "_")}.pkl')
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
        
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        
        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value for missing columns
        
        # Reorder columns to match training data
        input_df = input_df[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'prediction_probability': prediction_proba.tolist() if prediction_proba is not None else None
        })
        
    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'})

if __name__ == '__main__':
    print("=== FLASK APP STARTING ===")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Upload folder exists: {os.path.exists(app.config['UPLOAD_FOLDER'])}")
    print(f"Max content length: {app.config['MAX_CONTENT_LENGTH']} bytes")
    print(f"Secret key set: {'secret_key' in app.config}")
    print("=== STARTING FLASK SERVER ===")
    app.run(debug=True)