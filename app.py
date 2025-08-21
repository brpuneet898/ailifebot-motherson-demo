from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)
import pickle
import os
import re
import json
from werkzeug.utils import secure_filename
import warnings
import traceback
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.xlsx'):
        return pd.read_excel(path, engine='openpyxl')
    elif path.endswith('.xls'):
        return pd.read_excel(path, engine='xlrd')
    else:
        raise ValueError('Unsupported file format')

def save_dataframe_csv(df: pd.DataFrame, base_path: str, suffix: str) -> str:
    root, _ = os.path.splitext(base_path)
    out_path = f"{root}{suffix}.csv"
    df.to_csv(out_path, index=False)
    return out_path

ID_NAME_REGEX = re.compile(
    r"(?:^|[_-])(id|uuid|guid|serial|serialno|partserial|barcode|qrcode|scan|scancode|"
    r"code$|jobcode|stationcode|cellcode|tenantcode|baseorgcode|fgqr|hash|"
    r"created|createdon|updated|timestamp|time|date|eoldate)(?:$|[_-])",
    re.IGNORECASE
)

def suggest_drop_columns(df: pd.DataFrame):
    """Heuristics to suggest ID-like / non-informative columns."""
    n = len(df)
    suggestions = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        nunique = s.nunique(dropna=True)
        unique_ratio = (nunique / max(1, n))
        name_match = bool(ID_NAME_REGEX.search(col))
        avg_len = float(s.astype(str).str.len().mean()) if s.dtype == 'object' else None
        looks_long_token = (avg_len is not None and avg_len >= 12)

        reason_parts = []
        score = 0.0

        if name_match:
            reason_parts.append("name looks like ID/code/time")
            score += 2.0
        if unique_ratio >= 0.95:
            reason_parts.append(f"very high uniqueness ({unique_ratio:.2f})")
            score += 1.5
        if dtype == 'object' and looks_long_token:
            reason_parts.append(f"long string tokens (avg len ~{avg_len:.0f})")
            score += 0.7
        # Monotonic increasing ints often are row IDs
        if pd.api.types.is_integer_dtype(s) and n > 3:
            try:
                if s.is_monotonic_increasing and unique_ratio > 0.9:
                    reason_parts.append("monotonic increasing index-like")
                    score += 0.5
            except Exception:
                pass

        if score >= 1.5:
            suggestions.append({
                "column": col,
                "dtype": dtype,
                "unique_ratio": round(unique_ratio, 4),
                "reason": ", ".join(reason_parts),
                "score": round(score, 3)
            })

    # Sort highest score first
    suggestions.sort(key=lambda x: x["score"], reverse=True)
    return suggestions

# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})

    file = request.files['file']
    dataset_type = request.form.get('dataset_type')

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        return jsonify({'error': 'Invalid file format. Please upload CSV/XLSX/XLS'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
    except Exception as save_error:
        return jsonify({'error': f'Error saving file: {str(save_error)}'})

    try:
        df = load_dataframe(filepath)

        # Reset/clear any prior state for this dataset
        session[f'{dataset_type}_data'] = filepath
        session[f'{dataset_type}_dropped_data'] = None
        session[f'{dataset_type}_dropped_columns'] = []
        session[f'{dataset_type}_drop_confirmed'] = False
        session[f'{dataset_type}_cleaned_data'] = None
        session[f'{dataset_type}_models_dir'] = None
        session[f'{dataset_type}_feature_columns'] = None
        session[f'{dataset_type}_categorical_cols'] = None

        # Prepare response
        df_head = df.head()
        head_data = json.loads(df_head.to_json(orient='records', date_format='iso'))

        categorical_cols, numerical_cols = [], []
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)

        missing_info = df.isnull().sum().to_dict()
        missing_cols = {k: int(v) for k, v in missing_info.items() if v > 0}

        # NEW: drop suggestions
        drop_suggestions = suggest_drop_columns(df)

        # Make JSON-able
        response_data = {
            'success': True,
            'head_data': head_data,
            'columns': df.columns.tolist(),
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols,
            'missing_values': missing_cols,
            'total_rows': int(len(df)),
            'dataset_type': dataset_type,
            'drop_suggestions': drop_suggestions  # <--- NEW
        }
        json.dumps(response_data)  # sanity
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}. Please check the file.'})

@app.route('/drop_columns', methods=['POST'])
def drop_columns():
    """
    Body:
      {
        "dataset_type": "lever-force" | "eol-machine",
        "columns": ["Id", "ScanCode", ...]   // may be empty to confirm NONE
      }
    """
    data = request.get_json(silent=True) or {}
    dataset_type = data.get('dataset_type')
    cols_to_drop = data.get('columns', [])

    original_path = session.get(f'{dataset_type}_data')
    if not original_path or not os.path.exists(original_path):
        return jsonify({'error': 'Dataset not found. Please upload again.'})

    try:
        df = load_dataframe(original_path)
        existing = [c for c in cols_to_drop if c in df.columns]
        missing = [c for c in cols_to_drop if c not in df.columns]

        if existing:
            df = df.drop(columns=existing)

        # Always produce a working dropped file (even if no drops), so downstream is consistent
        dropped_path = save_dataframe_csv(df, original_path, '_dropped')
        session[f'{dataset_type}_dropped_data'] = dropped_path
        session[f'{dataset_type}_dropped_columns'] = existing
        session[f'{dataset_type}_drop_confirmed'] = True
        # Reset any cleaned/model state after a new drop action
        session[f'{dataset_type}_cleaned_data'] = None
        session[f'{dataset_type}_models_dir'] = None
        session[f'{dataset_type}_feature_columns'] = None
        session[f'{dataset_type}_categorical_cols'] = None

        return jsonify({
            'success': True,
            'message': 'Columns dropped and working file prepared' if existing else 'No columns dropped (confirmed)',
            'dropped_columns': existing,
            'not_found': missing,
            'working_file': os.path.basename(dropped_path),
            'remaining_columns': df.columns.tolist(),
            'shape': [int(df.shape[0]), int(df.shape[1])]
        })
    except Exception as e:
        return jsonify({'error': f'Error dropping columns: {str(e)}'})

@app.route('/fix_missing', methods=['POST'])
def fix_missing_values():
    dataset_type = request.json.get('dataset_type')

    # Gate: must confirm /drop_columns first (even with empty list)
    drop_confirmed = session.get(f'{dataset_type}_drop_confirmed', False)
    if not drop_confirmed:
        return jsonify({'error': 'Please review/drop columns first. Call /drop_columns with your list (or [] to confirm none).'})
    
    # Use the dropped working file if available; else fallback to original (shouldn't happen due to gate)
    filepath = session.get(f'{dataset_type}_dropped_data') or session.get(f'{dataset_type}_data')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Working dataset not found. Please upload and drop columns again.'})

    try:
        df = load_dataframe(filepath)

        original_missing = df.isnull().sum().to_dict()
        fixes_applied = {}

        for col in df.columns:
            miss = df[col].isnull().sum()
            if miss > 0:
                if df[col].dtype in ['object', 'category']:
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
                    fixes_applied[col] = f"Filled {miss} missing with mode: '{mode_val}'"
                else:
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df[col].fillna(median_val, inplace=True)
                    fixes_applied[col] = f"Filled {miss} missing with median: {median_val:.4f}"

        cleaned_filepath = save_dataframe_csv(df, filepath, '_cleaned')
        session[f'{dataset_type}_cleaned_data'] = cleaned_filepath
        return jsonify({
            'success': True,
            'fixes_applied': fixes_applied,
            'message': f"Fixed missing values in {len(fixes_applied)} columns"
        })
    except Exception as e:
        return jsonify({'error': f'Error fixing missing values: {str(e)}'})

@app.route('/train_models', methods=['POST'])
def train_models():
    dataset_type = request.json.get('dataset_type')
    cleaned_filepath = session.get(f'{dataset_type}_cleaned_data')

    if not cleaned_filepath or not os.path.exists(cleaned_filepath):
        return jsonify({'error': 'Cleaned dataset not found. Please fix missing values first.'})

    try:
        df = load_dataframe(cleaned_filepath)
        print("CSV file loaded for training" if cleaned_filepath.endswith('.csv') else "XLSX file loaded for training")
        print(f"Dataset loaded for training. Shape: {df.shape}")
        print(f"Available columns: {df.columns.tolist()}")

        # Identify target column (RESULT-ish)
        result_column = None
        possible = ['RESULT', 'Result', 'result']
        for col in df.columns:
            if col in possible or 'result' in col.lower():
                result_column = col
                break
        if result_column is None:
            available_cols = ', '.join(df.columns.tolist())
            return jsonify({'error': f'Target column not found. Available columns: {available_cols}'})

        X = df.drop([result_column], axis=1)
        y = df[result_column]

        # Encode categoricals
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        print(f"Categorical columns to encode: {categorical_cols.tolist()}")

        encoders = {}
        feature_metadata = {}
        for col in categorical_cols:
            cat = pd.Categorical(X[col])
            X[col] = cat.codes
            categories = list(cat.categories)
            encoders[col] = {
                "categories": categories,
                "value_to_code": {v: int(i) for i, v in enumerate(categories)}
            }

        # Build feature metadata for UI
        for col in X.columns:
            if col in encoders:
                uniques = encoders[col]["categories"]
                feature_metadata[col] = {
                    "type": "categorical",
                    "uniques": uniques,
                    "default": uniques[0] if len(uniques) > 0 else None
                }
            else:
                col_series = df[col] if col in df.columns else X[col]
                try:
                    default_val = float(pd.to_numeric(col_series, errors='coerce').median())
                    if pd.isna(default_val):
                        default_val = 0.0
                except Exception:
                    default_val = 0.0
                feature_metadata[col] = {"type": "numeric", "default": default_val}

        # Convert any lingering non-numeric to numeric or drop
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            for col in non_numeric_cols:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    X = X.drop([col], axis=1)

        # Handle NaNs
        if X.isnull().values.any():
            numeric_cols = X.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if X[col].isnull().any():
                    med = X[col].median()
                    if pd.isna(med):
                        med = 0
                    X[col].fillna(med, inplace=True)
            if X.isnull().values.any():
                X.fillna(0, inplace=True)

        # Remove rows with null target
        valid_idx = ~y.isnull()
        if not valid_idx.all():
            X = X[valid_idx]
            y = y[valid_idx]

        if len(y) < 10:
            return jsonify({'error': f'Not enough data for training. Only {len(y)} valid samples found.'})

        # Train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_split = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': make_pipeline(
                StandardScaler(),
                LogisticRegression(random_state=42, max_iter=2000)
            )
        }
        results = {}
        trained_models = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

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

        # Persist models + metadata
        models_dir = f'models_{dataset_type}'
        os.makedirs(models_dir, exist_ok=True)
        for name, model in trained_models.items():
            model_filename = os.path.join(models_dir, f'{name.lower().replace(" ", "_")}.pkl')
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)

        with open(os.path.join(models_dir, 'feature_columns.json'), 'w') as f:
            json.dump(X.columns.tolist(), f)
        with open(os.path.join(models_dir, 'encoders.json'), 'w') as f:
            json.dump(encoders, f)
        with open(os.path.join(models_dir, 'feature_metadata.json'), 'w') as f:
            json.dump(feature_metadata, f)

        session[f'{dataset_type}_models_dir'] = models_dir
        session[f'{dataset_type}_feature_columns'] = X.columns.tolist()
        session[f'{dataset_type}_categorical_cols'] = list(encoders.keys())

        # Top-5 feature importance
        top_importance = {}
        try:
            rf = trained_models.get('Random Forest')
            if rf is not None and hasattr(rf, 'feature_importances_'):
                importances = rf.feature_importances_
                pairs = sorted(zip(X.columns.tolist(), importances), key=lambda t: t[1], reverse=True)[:5]
                top_importance['Random Forest'] = [{"feature": f, "importance": float(s)} for f, s in pairs]
        except Exception as e:
            print(f"RF importance error: {e}")

        try:
            lr = trained_models.get('Logistic Regression')
            if lr is not None:
                from sklearn.pipeline import Pipeline
                lr_est = lr.named_steps.get('logisticregression') if isinstance(lr, Pipeline) else lr
                if hasattr(lr_est, 'coef_'):
                    coefs = abs(lr_est.coef_[0]) if lr_est.coef_.ndim == 2 else abs(lr_est.coef_)
                    pairs = sorted(zip(X.columns.tolist(), coefs), key=lambda t: t[1], reverse=True)[:5]
                    top_importance['Logistic Regression'] = [{"feature": f, "importance": float(s)} for f, s in pairs]
        except Exception as e:
            print(f"LR coef importance error: {e}")

        return jsonify({
            'success': True,
            'results': results,
            'feature_columns': X.columns.tolist(),
            'feature_metadata': feature_metadata,
            'top_feature_importance': top_importance
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
    categorical_cols = session.get(f'{dataset_type}_categorical_cols', [])

    if not models_dir or not feature_columns:
        return jsonify({'error': 'Models not trained yet'})

    try:
        model_filename = os.path.join(models_dir, f'{model_name.lower().replace(" ", "_")}.pkl')
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)

        encoders_path = os.path.join(models_dir, 'encoders.json')
        metadata_path = os.path.join(models_dir, 'feature_metadata.json')
        encoders = {}
        feature_metadata = {}

        if os.path.exists(encoders_path):
            with open(encoders_path, 'r') as f:
                encoders = json.load(f)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                feature_metadata = json.load(f)

        input_df = pd.DataFrame([input_data])

        # Ensure all expected columns exist
        for col in feature_columns:
            if col not in input_df.columns:
                if col in feature_metadata and feature_metadata[col].get("type") == "categorical":
                    default_cat = feature_metadata[col].get("default")
                    input_df[col] = default_cat if default_cat is not None else ""
                else:
                    input_df[col] = 0

        # Encode categoricals
        for col in categorical_cols:
            if col in input_df.columns and col in encoders:
                v2c = encoders[col]["value_to_code"]
                input_df[col] = input_df[col].map(lambda v: v2c.get(str(v), v2c.get(v, -1)))

        # Coerce numerics
        for col in input_df.columns:
            if col not in categorical_cols:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        input_df = input_df[feature_columns]

        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None

        pred_val = int(prediction) if isinstance(prediction, (np.integer, int, np.int64)) else prediction
        proba_list = prediction_proba.tolist() if prediction_proba is not None else None

        return jsonify({
            'success': True,
            'prediction': pred_val,
            'prediction_probability': proba_list
        })
    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
