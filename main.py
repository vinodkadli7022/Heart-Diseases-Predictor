import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# =========================
# Constants and configuration
# =========================
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_FOLDS = 5
TOP_K_FEATURES = 10
DPI = 150
sns.set_theme(style='whitegrid', context='talk')

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'heart.csv'
EDA_PLOT_PATH = BASE_DIR / 'eda_plots.png'
FEATURE_IMPORTANCE_PATH = BASE_DIR / 'feature_importance.png'
MODEL_COMPARISON_PATH = BASE_DIR / 'model_comparison.png'
EVALUATION_PLOT_PATH = BASE_DIR / 'evaluation_plots.png'
ARTIFACT_PATH = BASE_DIR / 'heart_disease_artifacts.pkl'


# =========================
# Utility helpers
# =========================
def status(message: str) -> None:
    print(f'\n[STATUS] {message}')


def print_divider(title: str) -> None:
    print('\n' + '=' * 100)
    print(title)
    print('=' * 100)


def fit_label_encoders(df: pd.DataFrame):
    encoders = {}
    encoded_df = df.copy()
    object_columns = encoded_df.select_dtypes(include=['object']).columns.tolist()
    for column in object_columns:
        encoder = LabelEncoder()
        encoded_df[column] = encoder.fit_transform(encoded_df[column])
        encoders[column] = encoder
    return encoded_df, encoders


def transform_with_label_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    encoded_df = df.copy()
    for column, encoder in encoders.items():
        if column in encoded_df.columns:
            encoded_df[column] = encoder.transform(encoded_df[column])
    return encoded_df


def impute_cholesterol_by_sex(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working['Cholesterol'] = working['Cholesterol'].replace(0, np.nan)
    sex_medians = working.groupby('Sex')['Cholesterol'].transform('median')
    overall_median = working['Cholesterol'].median()
    working['Cholesterol'] = working['Cholesterol'].fillna(sex_medians)
    working['Cholesterol'] = working['Cholesterol'].fillna(overall_median)
    return working


def impute_cholesterol_with_medians(df: pd.DataFrame, sex_medians: dict, overall_median: float) -> pd.DataFrame:
    working = df.copy()
    working['Cholesterol'] = working['Cholesterol'].replace(0, np.nan)
    working['Cholesterol'] = working['Cholesterol'].fillna(working['Sex'].map(sex_medians))
    working['Cholesterol'] = working['Cholesterol'].fillna(overall_median)
    return working


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    # Cardiac efficiency relative to age.
    working['HRtoAge_Ratio'] = working['MaxHR'] / working['Age']

    # Interaction between exercise-induced angina and ST depression.
    working['STxAngina'] = working['Oldpeak'] * working['ExerciseAngina']

    # Ordinal age grouping used as a coarse risk proxy.
    working['AgeBin'] = pd.cut(
        working['Age'],
        bins=[-np.inf, 39, 55, 70, np.inf],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype(int)

    return working


def save_artifacts(artifacts: dict, output_path: Path) -> None:
    with open(output_path, 'wb') as file_handle:
        pickle.dump(artifacts, file_handle)
    print(f'[SAVED] Model artifacts -> {output_path}')


def load_artifacts(artifact_path: Path) -> dict:
    with open(artifact_path, 'rb') as file_handle:
        return pickle.load(file_handle)


def preprocess_new_data(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    working = df.copy()

    required_columns = [
        'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
        'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
    ]
    missing_columns = [column for column in required_columns if column not in working.columns]
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')

    working = impute_cholesterol_with_medians(
        working,
        artifacts['sex_medians'],
        artifacts['overall_cholesterol_median'],
    )
    working = transform_with_label_encoders(working, artifacts['encoders'])
    working = engineer_features(working)

    x_new = working[artifacts['selected_features']].copy()
    x_new_scaled = pd.DataFrame(
        artifacts['scaler'].transform(x_new),
        columns=artifacts['selected_features'],
        index=x_new.index,
    )
    return x_new_scaled


def predict_new_data(input_path: Path, artifact_path: Path = ARTIFACT_PATH) -> None:
    print_divider('NEW DATA PREDICTION MODE')
    artifacts = load_artifacts(artifact_path)
    new_df = pd.read_csv(input_path)
    x_new_scaled = preprocess_new_data(new_df, artifacts)
    probabilities = artifacts['model'].predict_proba(x_new_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    output_df = new_df.copy()
    output_df['Predicted_HeartDisease'] = predictions
    output_df['Predicted_Probability'] = probabilities

    output_path = input_path.with_name(f'{input_path.stem}_predictions.csv')
    output_df.to_csv(output_path, index=False)
    print(f'[SAVED] Predictions -> {output_path}')
    print('\nPreview:')
    print(output_df[['Predicted_HeartDisease', 'Predicted_Probability']].head().to_string(index=False))


def fit_select_k_best(x_train: pd.DataFrame, y_train: pd.Series, feature_names):
    selector = SelectKBest(score_func=f_classif, k=min(TOP_K_FEATURES, x_train.shape[1]))
    selector.fit(x_train, y_train)
    scores = pd.DataFrame({
        'Feature': feature_names,
        'F_Score': selector.scores_,
        'P_Value': selector.pvalues_,
    }).sort_values('F_Score', ascending=False).reset_index(drop=True)
    selected_mask = selector.get_support()
    selected_features = list(np.array(feature_names)[selected_mask])
    return selector, scores, selected_features


def plot_eda(df: pd.DataFrame, output_path: Path) -> None:
    print_divider('SECTION 1 - EXPLORATION VISUALS')
    fig = plt.figure(figsize=(22, 16), constrained_layout=True)
    gs = gridspec.GridSpec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    class_counts = df['HeartDisease'].value_counts().sort_index()
    class_labels = ['No Disease', 'Disease']
    sns.barplot(x=class_labels, y=class_counts.values, ax=ax1, palette='viridis')
    ax1.set_title('Class Distribution')
    ax1.set_xlabel('Target Class')
    ax1.set_ylabel('Count')

    ax2 = fig.add_subplot(gs[0, 1])
    for label, color in zip([0, 1], ['steelblue', 'darkorange']):
        sns.histplot(
            df.loc[df['HeartDisease'] == label, 'Age'],
            bins=20,
            kde=True,
            stat='density',
            element='step',
            fill=False,
            ax=ax2,
            label=f'HeartDisease={label}',
            color=color,
        )
    ax2.set_title('Age Distribution by HeartDisease')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Density')
    ax2.legend(title='Class')

    ax3 = fig.add_subplot(gs[1, 0])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('Correlation Heatmap of Numeric Features')

    ax4 = fig.add_subplot(gs[1, 1])
    long_df = df.melt(id_vars='HeartDisease', value_vars=['Oldpeak', 'MaxHR'], var_name='Feature', value_name='Value')
    sns.boxplot(data=long_df, x='Feature', y='Value', hue='HeartDisease', ax=ax4, palette='Set2')
    ax4.set_title('Oldpeak and MaxHR by HeartDisease')
    ax4.set_xlabel('Feature')
    ax4.set_ylabel('Value')
    ax4.legend(title='HeartDisease')

    ax5 = fig.add_subplot(gs[2, :])
    chest_counts = df.groupby(['ChestPainType', 'HeartDisease']).size().reset_index(name='Count')
    sns.countplot(data=df, x='ChestPainType', hue='HeartDisease', ax=ax5, palette='Set1')
    ax5.set_title('ChestPainType by HeartDisease')
    ax5.set_xlabel('ChestPainType')
    ax5.set_ylabel('Count')
    ax5.legend(title='HeartDisease')

    fig.suptitle('Heart Disease Dataset - EDA Dashboard', fontsize=20, y=1.02)
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'[SAVED] EDA dashboard -> {output_path}')


def plot_feature_selection(scores: pd.DataFrame, output_path: Path) -> None:
    print_divider('SECTION 2 - FEATURE SELECTION VISUAL')
    plot_df = scores.copy()
    plot_df['Significant'] = plot_df['P_Value'] < 0.05
    plot_df = plot_df.sort_values('F_Score', ascending=True)

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    ax = fig.add_subplot(111)
    colors = plot_df['Significant'].map({True: '#d95f02', False: '#1b9e77'})
    ax.barh(plot_df['Feature'], plot_df['F_Score'], color=colors)
    ax.set_title('SelectKBest F-Scores by Feature')
    ax.set_xlabel('F-Score')
    ax.set_ylabel('Feature')

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color='#d95f02', label='p < 0.05'),
        Patch(color='#1b9e77', label='p >= 0.05'),
    ]
    ax.legend(handles=legend_handles, title='Statistical Significance', loc='lower right')
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'[SAVED] Feature selection chart -> {output_path}')


def evaluate_model_cv(model, x: pd.DataFrame, y: pd.Series, use_scaling: bool = True):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = []
    acc_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y), start=1):
        x_fold_train = x.iloc[train_idx].copy()
        x_fold_val = x.iloc[val_idx].copy()
        y_fold_train = y.iloc[train_idx].copy()
        y_fold_val = y.iloc[val_idx].copy()

        if use_scaling:
            fold_scaler = StandardScaler()
            x_fold_train = pd.DataFrame(fold_scaler.fit_transform(x_fold_train), columns=x.columns, index=x_fold_train.index)
            x_fold_val = pd.DataFrame(fold_scaler.transform(x_fold_val), columns=x.columns, index=x_fold_val.index)

        fold_model = clone(model)
        fold_model.fit(x_fold_train, y_fold_train)
        fold_proba = fold_model.predict_proba(x_fold_val)[:, 1]
        fold_pred = (fold_proba >= 0.5).astype(int)

        auc_scores.append(roc_auc_score(y_fold_val, fold_proba))
        acc_scores.append(accuracy_score(y_fold_val, fold_pred))
        print(f'  Fold {fold_idx}: AUC={auc_scores[-1]:.4f}, Accuracy={acc_scores[-1]:.4f}')

    return np.array(auc_scores), np.array(acc_scores)


def plot_model_comparison(results: pd.DataFrame, output_path: Path) -> None:
    print_divider('SECTION 3 - MODEL COMPARISON VISUAL')
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    ax = fig.add_subplot(111)

    x = np.arange(len(results))
    width = 0.35

    ax.bar(x - width / 2, results['Mean AUC'], width, yerr=results['AUC Std'], capsize=5, label='Mean AUC', color='#4c78a8')
    ax.bar(x + width / 2, results['Mean Accuracy'], width, yerr=results['Acc Std'], capsize=5, label='Mean Accuracy', color='#f58518')

    ax.set_xticks(x)
    ax.set_xticklabels(results['Model'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('5-Fold Cross-Validation Model Comparison')
    ax.legend()
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'[SAVED] Model comparison chart -> {output_path}')


def tune_best_model(model_name: str, x_train: pd.DataFrame, y_train: pd.Series):
    if model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs'],
        }
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=RANDOM_STATE)
        param_grid = {
            'n_estimators': [200, 400],
            'max_depth': [None, 8, 12],
            'min_samples_split': [2, 5],
        }
    else:
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.03, 0.1],
            'max_depth': [2, 3],
            'subsample': [0.8, 1.0],
        }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit=True,
    )
    grid.fit(x_train, y_train)
    return grid


def plot_evaluation_dashboard(y_test, y_pred, y_proba, feature_names, trained_model, output_path: Path) -> None:
    print_divider('SECTION 4 - FINAL EVALUATION VISUALS')

    cm = confusion_matrix(y_test, y_pred)
    cm_row_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_curve, precision_curve)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
    }

    fig = plt.figure(figsize=(22, 16), constrained_layout=True)
    gs = gridspec.GridSpec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        cm,
        annot=np.array([[f'{cm[i, j]}\n({cm_row_pct[i, j]:.1%})' for j in range(cm.shape[1])] for i in range(cm.shape[0])]),
        fmt='',
        cmap='Blues',
        cbar=False,
        ax=ax1,
        xticklabels=['Pred 0', 'Pred 1'],
        yticklabels=['Actual 0', 'Actual 1'],
    )
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted Class')
    ax1.set_ylabel('Actual Class')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fpr, tpr, color='#4c78a8', linewidth=2, label=f'ROC AUC = {roc_auc:.3f}')
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    ax2.set_title('ROC Curve')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(loc='lower right')

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(recall_curve, precision_curve, color='#f58518', linewidth=2, label=f'PR AUC = {pr_auc:.3f}')
    ax3.set_title('Precision-Recall Curve')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.legend(loc='lower left')

    ax4 = fig.add_subplot(gs[1, 1])
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    ax4.bar(metric_names, metric_values, color=['#4c78a8', '#72b7b2', '#e45756', '#54a24b', '#b279a2'])
    ax4.set_ylim(0, 1.05)
    ax4.set_title('Final Model Metrics')
    ax4.set_ylabel('Score')
    ax4.tick_params(axis='x', rotation=20)
    for idx, value in enumerate(metric_values):
        ax4.text(idx, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontsize=11)

    ax5 = fig.add_subplot(gs[2, 0])
    if hasattr(trained_model, 'feature_importances_'):
        importances = pd.Series(trained_model.feature_importances_, index=feature_names).sort_values(ascending=True)
        ax5.barh(importances.index, importances.values, color='#59a14f')
        ax5.set_title('Feature Importances')
        ax5.set_xlabel('Importance')
        ax5.set_ylabel('Feature')
    else:
        ax5.text(0.5, 0.5, 'Feature importances not available for this model', ha='center', va='center', fontsize=14)
        ax5.set_axis_off()

    ax6 = fig.add_subplot(gs[2, 1])
    y_proba = np.asarray(y_proba)
    ax6.hist(y_proba[y_test == 0], bins=20, alpha=0.7, label='Actual 0', color='#4c78a8', density=True)
    ax6.hist(y_proba[y_test == 1], bins=20, alpha=0.7, label='Actual 1', color='#f58518', density=True)
    ax6.set_title('Predicted Probability Distribution')
    ax6.set_xlabel('Predicted Probability of Heart Disease')
    ax6.set_ylabel('Density')
    ax6.legend()

    fig.suptitle('Final Model Evaluation Dashboard', fontsize=20, y=1.02)
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'[SAVED] Evaluation dashboard -> {output_path}')


# =========================
# Main execution pipeline
# =========================
def main():
    parser = argparse.ArgumentParser(description='Heart disease classification pipeline')
    parser.add_argument('--predict', type=str, help='Path to a new CSV file for inference')
    args = parser.parse_args()

    if args.predict:
        predict_new_data(Path(args.predict))
        return

    print_divider('HEART DISEASE CLASSIFICATION PIPELINE')
    status(f'Loading dataset from {DATA_PATH}')
    df = pd.read_csv(DATA_PATH)

    print_divider('SECTION 1 - DATA EXPLORATION')
    print(f'Dataset shape: {df.shape}')
    print('\nData types:')
    print(df.dtypes)
    print('\nDescribe:')
    print(df.describe())
    print('\nInfo:')
    df.info()

    zero_cholesterol_count = int((df['Cholesterol'] == 0).sum())
    print(f"\nMissing values before preprocessing:\n{df.isna().sum()}")
    print(f'Cholesterol values equal to 0 (treated as missing): {zero_cholesterol_count}')

    duplicate_count = int(df.duplicated().sum())
    print(f'Duplicate rows found: {duplicate_count}')
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f'Duplicate rows removed. New shape: {df.shape}')

    class_distribution = df['HeartDisease'].value_counts().sort_index()
    class_ratio = class_distribution / class_distribution.sum()
    print('\nTarget class distribution:')
    for cls, count in class_distribution.items():
        print(f'  Class {cls}: {count} ({class_ratio[cls]:.1%})')
    balance_note = 'balanced' if class_ratio.min() >= 0.4 else 'imbalanced'
    print(f'Class balance assessment: the target is {balance_note} (minority class share = {class_ratio.min():.1%}).')

    cholesterol_reference = df['Cholesterol'].replace(0, np.nan)
    sex_medians = cholesterol_reference.groupby(df['Sex']).median().to_dict()
    overall_cholesterol_median = float(cholesterol_reference.median())

    plot_eda(df, EDA_PLOT_PATH)

    status('Imputing Cholesterol=0 values using sex-specific medians')
    df = impute_cholesterol_by_sex(df)

    status('Encoding object-type columns with LabelEncoder')
    df, encoders = fit_label_encoders(df)
    print(f'Encoded columns: {list(encoders.keys())}')

    status('Engineering medically motivated features')
    df = engineer_features(df)
    print('Engineered features added: HRtoAge_Ratio, STxAngina, AgeBin')

    print_divider('SECTION 2 - FEATURE ENGINEERING AND SELECTION')
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f'Train shape: {x_train_raw.shape}; Test shape: {x_test_raw.shape}')

    selector, feature_scores, selected_features = fit_select_k_best(x_train_raw, y_train, X.columns.tolist())
    print('\nRanked feature scores (training set only):')
    print(feature_scores.to_string(index=False))
    print(f'\nTop {TOP_K_FEATURES} selected features: {selected_features}')

    plot_feature_selection(feature_scores, FEATURE_IMPORTANCE_PATH)

    x_train_selected = x_train_raw[selected_features].copy()
    x_test_selected = x_test_raw[selected_features].copy()

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train_selected), columns=selected_features, index=x_train_selected.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test_selected), columns=selected_features, index=x_test_selected.index)
    print('\nStandardScaler fitted on training data and applied to train/test splits.')

    print_divider('SECTION 3 - MODEL SELECTION AND JUSTIFICATION')
    model_configs = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
        ('Random Forest', RandomForestClassifier(random_state=RANDOM_STATE), False),
        ('Gradient Boosting', GradientBoostingClassifier(random_state=RANDOM_STATE), False),
    ]

    comparison_rows = []
    for model_name, model, use_scaling in model_configs:
        status(f'Cross-validating {model_name}')
        auc_scores, acc_scores = evaluate_model_cv(model, x_train_selected, y_train, use_scaling=use_scaling)
        comparison_rows.append({
            'Model': model_name,
            'Mean AUC': auc_scores.mean(),
            'AUC Std': auc_scores.std(ddof=1),
            'Mean Accuracy': acc_scores.mean(),
            'Acc Std': acc_scores.std(ddof=1),
        })
        print(f'{model_name}: mean AUC={auc_scores.mean():.4f} ± {auc_scores.std(ddof=1):.4f}, mean accuracy={acc_scores.mean():.4f}')

    comparison_df = pd.DataFrame(comparison_rows).sort_values('Mean AUC', ascending=False).reset_index(drop=True)
    print('\nModel comparison table:')
    print(comparison_df.to_string(index=False))
    plot_model_comparison(comparison_df, MODEL_COMPARISON_PATH)

    best_model_name = comparison_df.iloc[0]['Model']
    print(f'\nSelected final model for tuning: {best_model_name}')

    # Final model choice is driven by cross-validated discrimination on the held-out training data.
    # Gradient Boosting is usually the best fit here because the data mixes nonlinear effects,
    # categorical encodings, and interaction-heavy clinical signals that are not well captured by
    # a purely linear boundary. Logistic Regression remains highly interpretable but assumes near-
    # linear separability in the transformed feature space, which is restrictive for this problem.
    # Random Forest is robust and expressive, but boosting often extracts more signal from compact,
    # medium-sized tabular datasets by fitting residual structure sequentially. The main limitation
    # of boosting is reduced global interpretability and some sensitivity to noisy labels or overfitting
    # if tuned too aggressively, so its hyperparameters need explicit control.

    status('Running GridSearchCV on the best model')
    grid = tune_best_model(best_model_name, x_train_scaled, y_train)
    print(f'Best parameters: {grid.best_params_}')
    print(f'Best CV AUC: {grid.best_score_:.4f}')

    final_model = grid.best_estimator_
    status('Training final tuned model on the full training set')
    final_model.fit(x_train_scaled, y_train)

    artifacts = {
        'model': final_model,
        'scaler': scaler,
        'selected_features': selected_features,
        'encoders': encoders,
        'sex_medians': sex_medians,
        'overall_cholesterol_median': overall_cholesterol_median,
    }
    save_artifacts(artifacts, ARTIFACT_PATH)

    y_pred = final_model.predict(x_test_scaled)
    y_proba = final_model.predict_proba(x_test_scaled)[:, 1]

    print_divider('SECTION 4 - FINAL EVALUATION')
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
    }
    for metric_name, metric_value in metrics.items():
        print(f'{metric_name}: {metric_value:.4f}')

    print('\nClassification report:')
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    print('Confusion matrix (counts):')
    print(cm)
    print('Confusion matrix (row percentages):')
    print(np.round(cm_pct, 4))

    plot_evaluation_dashboard(y_test, y_pred, y_proba, selected_features, final_model, EVALUATION_PLOT_PATH)

    print_divider('PIPELINE COMPLETE')
    print('Generated files:')
    print(f'  - {EDA_PLOT_PATH}')
    print(f'  - {FEATURE_IMPORTANCE_PATH}')
    print(f'  - {MODEL_COMPARISON_PATH}')
    print(f'  - {EVALUATION_PLOT_PATH}')
    print(f'  - {ARTIFACT_PATH}')


if __name__ == '__main__':
    main()
