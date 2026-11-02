"""
test_models.py - Validates classification models and association rule parameters.
"""

import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from mlxtend.frequent_patterns import fpgrowth, association_rules
from scripts.data_utils import load_raw_data

def test_classification_pipelines():
    train_df, test_df = load_raw_data()
    num_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    
    # Impute modes
    for c in cat_cols:
        train_df[c] = train_df[c].fillna(train_df[c].mode()[0])
        test_df[c] = test_df[c].fillna(train_df[c].mode()[0])
        
    X_train = train_df[num_cols + cat_cols]
    y_train = (train_df['income'] == '>50K').astype(int)
    X_test = test_df[num_cols + cat_cols]
    y_test = (test_df['income'] == '>50K').astype(int)
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    
    # Decision Tree
    dt = Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=42))])
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)
    dt_f1 = f1_score(y_test, dt_preds)
    assert dt_acc > 0.84, f"Decision tree accuracy too low: {dt_acc}"
    assert dt_f1 > 0.60, f"Decision tree positive F1 too low: {dt_f1}"
    
    # KNN on subsample
    knn = Pipeline([('prep', preprocessor), ('clf', KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1))])
    knn.fit(X_train.iloc[:5000], y_train.iloc[:5000])
    knn_preds = knn.predict(X_test.iloc[:1000])
    knn_acc = accuracy_score(y_test.iloc[:1000], knn_preds)
    assert knn_acc > 0.80, f"KNN accuracy too low: {knn_acc}"

def test_association_rules_thresholds():
    train_df, _ = load_raw_data()
    tx_data = pd.DataFrame()
    tx_data['Age'] = pd.cut(train_df['age'], bins=[0, 29, 50, 120], labels=['Age:<30', 'Age:30-50', 'Age:>50'])
    tx_data['Hours'] = pd.cut(train_df['hours-per-week'], bins=[0, 34, 45, 120], labels=['Hours:<35', 'Hours:35-45', 'Hours:>45'])
    tx_data['CapGain'] = train_df['capital-gain'].apply(lambda x: 'CapGain:NonZero' if x > 0 else 'CapGain:Zero')
    tx_data['Income'] = train_df['income']
    
    tx_encoded = pd.get_dummies(tx_data, prefix='', prefix_sep='')
    freq_items = fpgrowth(tx_encoded, min_support=0.10, use_colnames=True)
    rules = association_rules(freq_items, metric='confidence', min_threshold=0.70)
    
    assert len(rules) > 0, "No rules found at min_threshold=0.70"
    high_lift = rules[rules['lift'] > 1.2]
    assert len(high_lift) > 0, "No rules with lift > 1.2"

