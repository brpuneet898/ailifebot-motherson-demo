from flask import Flask, render_template, request, jsonify, session, send_from_directory
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import pickle
import os
import time
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

def read_any(filepath):
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        return pd.read_excel(filepath, engine='openpyxl')
    elif filepath.endswith('.xls'):
        return pd.read_excel(filepath, engine='xlrd')
    else:
        raise ValueError('Unsupported file format')

def suggest_drops(df):
    """Heuristic: ID-like or high-uniqueness columns are suggested to drop."""
    suggestions = []
    n = len(df)
    low = [c.lower() for c in df.columns]
    for col, lcol in zip(df.columns, low):
        try:
            uniq = df[col].nunique(dropna=True)
            ratio = round(float(uniq) / max(1, n), 4)
        except Exception:
            uniq, ratio = 0, 0.0

        reason = None
        # name-based hints
        id_like = any(k in lcol for k in [
            'id', 'serial', 'scan', 'code', 'fgqr', 'qr', 'uuid',
            'createdon', 'created_at', 'timestamp', 'eoldate', 'date', 'time',
            'partserial', 'part_no'
        ])
        if id_like:
            reason = 'ID-like column'
        # uniqueness-based hint
        if ratio >= 0.9:
            reason = 'High-uniqueness (likely identifier)'

        if reason:
            suggestions.append({'column': col, 'reason': reason, 'unique_ratio': ratio})
    return suggestions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'})

        file = request.files['file']
        dataset_type = request.form.get('dataset_type')
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'})

        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
            return jsonify({'error': 'Invalid file format. Please upload CSV/XLSX/XLS.'})

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and analyze
        df = read_any(filepath)

        # Persist original upload path and default "working file" (before drops)
        session[f'{dataset_type}_data'] = filepath
        session[f'{dataset_type}_working_file'] = filepath  # will switch after /drop_columns
        session[f'{dataset_type}_dropped_cols'] = []        # reset drops on new upload

        # Build preview safely (timestamps -> ISO)
        df_head = df.head()
        head_data = json.loads(df_head.to_json(orient='records', date_format='iso'))

        # Column types
        categorical_cols = []
        numerical_cols = []
        for col in df.columns:
            if str(df[col].dtype) in ['object', 'category']:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)

        # Missing
        missing_info = df.isnull().sum().to_dict()
        missing_cols = {k: int(v) for k, v in missing_info.items() if v > 0}

        # Drop suggestions
        drop_suggestions = suggest_drops(df)

        response_data = {
            'success': True,
            'head_data': head_data,
            'columns': df.columns.tolist(),
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols,
            'missing_values': missing_cols,
            'total_rows': int(len(df)),
            'dataset_type': dataset_type,
            'drop_suggestions': drop_suggestions
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}. Please check the file.'})

@app.route('/drop_columns', methods=['POST'])
def drop_columns():
    try:
        data = request.get_json(force=True)
        dataset_type = data.get('dataset_type')
        cols = data.get('columns', []) or []

        base_path = session.get(f'{dataset_type}_data')
        if not base_path or not os.path.exists(base_path):
            return jsonify({'error': 'Dataset not found. Please upload first.'})

        df = read_any(base_path)
        not_found = [c for c in cols if c not in df.columns]
        drop_these = [c for c in cols if c in df.columns]
        if drop_these:
            df = df.drop(columns=drop_these)

        # Save as working file
        root, ext = os.path.splitext(base_path)
        working_file = root + '_dropped.csv'
        df.to_csv(working_file, index=False)

        # Persist
        session[f'{dataset_type}_working_file'] = working_file
        session[f'{dataset_type}_dropped_cols'] = drop_these

        return jsonify({
            'success': True,
            'message': 'Drop step confirmed.',
            'dropped_columns': drop_these,
            'not_found': not_found,
            'working_file': working_file,
            'remaining_columns': df.columns.tolist(),
            'shape': [int(df.shape[0]), int(df.shape[1])]
        })
    except Exception as e:
        return jsonify({'error': f'Error dropping columns: {str(e)}'})

@app.route('/fix_missing', methods=['POST'])
def fix_missing_values():
    try:
        dataset_type = request.json.get('dataset_type')
        # prefer working file (after drops). Fallback to raw.
        filepath = session.get(f'{dataset_type}_working_file') or session.get(f'{dataset_type}_data')
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'Dataset not found'})

        df = read_any(filepath)

        original_missing = df.isnull().sum().to_dict()
        fixes_applied = {}

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if str(df[col].dtype) in ['object', 'category']:
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
                    fixes_applied[col] = f"Filled {missing_count} missing values with mode: '{mode_val}'"
                else:
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    df[col] = df[col].fillna(median_val)
                    fixes_applied[col] = f"Filled {missing_count} missing values with median: {median_val:.2f}"

        # Save cleaned dataset (CSV)
        root, _ = os.path.splitext(filepath)
        cleaned_filepath = root + '_cleaned.csv'
        df.to_csv(cleaned_filepath, index=False)
        session[f'{dataset_type}_cleaned_data'] = cleaned_filepath

        return jsonify({
            'success': True,
            'fixes_applied': fixes_applied,
            'message': f'Fixed missing values in {len(fixes_applied)} columns'
        })
    except Exception as e:
        return jsonify({'error': f'Error fixing missing values: {str(e)}'})

@app.route('/train_models', methods=['POST'])
def train_models():
    try:
        dataset_type = request.json.get('dataset_type')
        cleaned_filepath = session.get(f'{dataset_type}_cleaned_data')

        if not cleaned_filepath or not os.path.exists(cleaned_filepath):
            return jsonify({'error': 'Cleaned dataset not found. Please run Fix Missing Values first.'})

        df = read_any(cleaned_filepath)

        # Find target column (Result-ish)
        result_column = None
        for col in df.columns:
            if col.lower() == 'result' or 'result' in col.lower():
                result_column = col
                break
        if result_column is None:
            return jsonify({'error': f'Target column not found. Looking for *result*-like columns. Available: {", ".join(df.columns)}'})

        X = df.drop([result_column], axis=1)
        y = df[result_column]

        # Encode categoricals
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        encoders = {}
        feature_metadata = {}

        for col in categorical_cols:
            cat = pd.Categorical(X[col])
            X[col] = cat.codes
            categories = list(cat.categories)
            value_to_code = {v: int(i) for i, v in enumerate(categories)}
            encoders[col] = {"categories": categories, "value_to_code": value_to_code}

        # Build metadata
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

        # Convert any lingering non-numerics
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns
        for col in non_numeric_cols:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except Exception:
                X = X.drop(columns=[col])

        # Fill remaining NaNs in numeric features with medians (and record them for batch use)
        numeric_cols = X.select_dtypes(include=['number']).columns
        numeric_medians = {}
        for col in numeric_cols:
            med = X[col].median()
            if pd.isna(med):
                med = 0.0
            X[col] = X[col].fillna(med)
            numeric_medians[col] = float(med)

        # Remove rows with null target
        valid_idx = ~y.isnull()
        X, y = X[valid_idx], y[valid_idx]
        if len(y) < 10:
            return jsonify({'error': f'Not enough data for training. Only {len(y)} valid samples found.'})

        # Train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
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
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            roc_auc, roc_data = None, None
            try:
                if len(np.unique(y)) == 2 and y_pred_proba is not None:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                    roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            except Exception:
                roc_auc, roc_data = None, None

            results[name] = {
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'roc_data': roc_data
            }
            trained_models[name] = model

        # Save models + artifacts
        models_dir = f'models_{dataset_type}'
        os.makedirs(models_dir, exist_ok=True)
        for name, model in trained_models.items():
            with open(os.path.join(models_dir, f'{name.lower().replace(" ", "_")}.pkl'), 'wb') as f:
                pickle.dump(model, f)

        with open(os.path.join(models_dir, 'feature_columns.json'), 'w') as f:
            json.dump(X.columns.tolist(), f)
        with open(os.path.join(models_dir, 'encoders.json'), 'w') as f:
            json.dump(encoders, f)
        with open(os.path.join(models_dir, 'feature_metadata.json'), 'w') as f:
            json.dump(feature_metadata, f)
        with open(os.path.join(models_dir, 'numeric_medians.json'), 'w') as f:
            json.dump(numeric_medians, f)

        session[f'{dataset_type}_models_dir'] = models_dir
        session[f'{dataset_type}_feature_columns'] = X.columns.tolist()
        session[f'{dataset_type}_categorical_cols'] = list(encoders.keys())

        # Top 5 importances
        top_importance = {}
        try:
            rf = trained_models.get('Random Forest')
            if rf is not None and hasattr(rf, 'feature_importances_'):
                pairs = sorted(
                    zip(X.columns.tolist(), rf.feature_importances_), key=lambda t: t[1], reverse=True
                )[:5]
                top_importance['Random Forest'] = [{"feature": f, "importance": float(s)} for f, s in pairs]
        except Exception as e:
            print(f"RF importance error: {e}")

        try:
            lr = trained_models.get('Logistic Regression')
            if lr is not None:
                lr_est = lr.named_steps.get('logisticregression') if isinstance(lr, Pipeline) else lr
                if hasattr(lr_est, 'coef_'):
                    coefs = np.abs(lr_est.coef_[0]) if lr_est.coef_.ndim == 2 else np.abs(lr_est.coef_)
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
    try:
        dataset_type = request.json.get('dataset_type')
        model_name = request.json.get('model_name')
        input_data = request.json.get('input_data')

        models_dir = session.get(f'{dataset_type}_models_dir')
        feature_columns = session.get(f'{dataset_type}_feature_columns')
        categorical_cols = session.get(f'{dataset_type}_categorical_cols', [])

        if not models_dir or not feature_columns:
            return jsonify({'error': 'Models not trained yet'})

        # Load model
        model_filename = os.path.join(models_dir, f'{model_name.lower().replace(" ", "_")}.pkl')
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)

        # Load encoders & metadata
        encoders = {}
        feature_metadata = {}
        encoders_path = os.path.join(models_dir, 'encoders.json')
        metadata_path = os.path.join(models_dir, 'feature_metadata.json')
        if os.path.exists(encoders_path):
            with open(encoders_path, 'r') as f:
                encoders = json.load(f)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                feature_metadata = json.load(f)

        # Single-row DataFrame
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

        # Predict
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None

        pred_val = int(pred) if isinstance(pred, (np.integer, int, np.int64)) else pred
        proba_list = proba.tolist() if proba is not None else None

        return jsonify({'success': True, 'prediction': pred_val, 'prediction_probability': proba_list})
    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Upload a CSV/XLSX with proper headers, apply same drop/encode/alignment,
    and return a preview with predictions.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'})

        file = request.files['file']
        dataset_type = request.form.get('dataset_type')
        model_name = request.form.get('model_name', 'Random Forest')

        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'})

        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
            return jsonify({'error': 'Invalid file format. Please upload CSV/XLSX/XLS.'})

        # Ensure models exist
        models_dir = session.get(f'{dataset_type}_models_dir')
        feature_columns = session.get(f'{dataset_type}_feature_columns')
        categorical_cols = session.get(f'{dataset_type}_categorical_cols', [])
        dropped_cols = session.get(f'{dataset_type}_dropped_cols', [])

        if not models_dir or not feature_columns:
            return jsonify({'error': 'Models not trained yet. Train before batch prediction.'})

        # Save temp upload
        filename = secure_filename(file.filename)
        tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'batch_{dataset_type}_{filename}')
        file.save(tmp_path)

        df_in = read_any(tmp_path)

        # Apply same drop step (if any)
        if dropped_cols:
            df_in = df_in.drop(columns=[c for c in dropped_cols if c in df_in.columns], errors='ignore')

        # Load artifacts
        with open(os.path.join(models_dir, f'{model_name.lower().replace(" ", "_")}.pkl'), 'rb') as f:
            model = pickle.load(f)
        encoders = {}
        feature_metadata = {}
        numeric_medians = {}
        p = os.path.join(models_dir, 'encoders.json')
        if os.path.exists(p):
            with open(p, 'r') as f:
                encoders = json.load(f)
        p = os.path.join(models_dir, 'feature_metadata.json')
        if os.path.exists(p):
            with open(p, 'r') as f:
                feature_metadata = json.load(f)
        p = os.path.join(models_dir, 'numeric_medians.json')
        if os.path.exists(p):
            with open(p, 'r') as f:
                numeric_medians = json.load(f)

        # Make a working copy
        Xb = df_in.copy()

        # Ensure all expected columns exist
        for col in feature_columns:
            if col not in Xb.columns:
                if col in feature_metadata and feature_metadata[col].get("type") == "categorical":
                    default_cat = feature_metadata[col].get("default")
                    Xb[col] = default_cat if default_cat is not None else ""
                else:
                    Xb[col] = np.nan

        # For categoricals: fill NA with default then encode
        for col in encoders.keys():
            if col in Xb.columns:
                default_cat = feature_metadata.get(col, {}).get('default')
                if default_cat is not None:
                    Xb[col] = Xb[col].fillna(default_cat)
                v2c = encoders[col]["value_to_code"]
                Xb[col] = Xb[col].map(lambda v: v2c.get(str(v), v2c.get(v, -1)))

        # Numeric coercion + fill with training medians (fallback 0)
        for col in feature_columns:
            if col not in encoders:  # numeric-ish
                Xb[col] = pd.to_numeric(Xb[col], errors='coerce')
                fill_val = numeric_medians.get(col, 0.0)
                Xb[col] = Xb[col].fillna(fill_val)

        # Reorder
        Xb = Xb[feature_columns]

        # Predict
        preds = model.predict(Xb)
        proba = model.predict_proba(Xb) if hasattr(model, 'predict_proba') else None

        # Build preview DataFrame with predictions
        preview = df_in.copy()
        preview['PREDICTION'] = preds
        if proba is not None:
            # confidence = prob of predicted class
            classes = list(model.classes_)
            conf = []
            for i, p in enumerate(preds):
                idx = classes.index(p)
                conf.append(float(proba[i, idx]))
            preview['CONFIDENCE'] = conf
        else:
            preview['CONFIDENCE'] = np.nan

        # Limit rows for UI
        limit = 200
        preview_rows = json.loads(preview.head(limit).to_json(orient='records', date_format='iso'))
        cols_out = list(preview.columns)
        # Save full predictions to uploads for download
        download_filename = f'predictions_{dataset_type}_{int(time.time())}.csv'
        download_path = os.path.join(app.config['UPLOAD_FOLDER'], download_filename)
        try:
            preview.to_csv(download_path, index=False)
            download_info = {'download_filename': download_filename, 'download_url': f'/download/{download_filename}'}
        except Exception as e:
            download_info = {'download_error': str(e)}

        return jsonify({
            'success': True,
            'total_rows': int(len(preview)),
            'columns': cols_out,
            'preview_rows': preview_rows,
            'highlight_columns': ['PREDICTION', 'CONFIDENCE'],
            'message': f'Predicted {len(preview)} rows (showing first {min(limit, len(preview))}).',
            **download_info
        })
    except Exception as e:
        return jsonify({'error': f'Error in batch prediction: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/download/<path:filename>')
def download_file(filename):
    # Serve files from uploads directory for download (as attachment)
    safe_name = secure_filename(filename)
    # If secure_filename changed the name, prefer original if it exists; otherwise use safe_name
    targets = [filename, safe_name]
    for t in targets:
        p = os.path.join(app.config['UPLOAD_FOLDER'], t)
        if os.path.exists(p):
            return send_from_directory(app.config['UPLOAD_FOLDER'], t, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404
