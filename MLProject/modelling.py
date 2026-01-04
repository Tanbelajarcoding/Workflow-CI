"""
MLflow Project - Credit Card Fraud Detection Training
=====================================================
Script untuk MLProject dengan parameterized training

Author: Sulthan
Date: January 2026
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, log_loss)
import argparse
import warnings

warnings.filterwarnings('ignore')


def load_data(train_path, test_path):
    """Load training and testing data"""
    print("Loading data...")
    print(f"   Train: {train_path}")
    print(f"   Test: {test_path}")
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Credit Card Fraud dataset menggunakan kolom 'Class'
    X_train = train_data.drop('Class', axis=1)
    y_train = train_data['Class']
    
    X_test = test_data.drop('Class', axis=1)
    y_test = test_data['Class']
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train labels: {y_train.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, params):
    """Train model with given parameters"""
    print("\n" + "="*70)
    print("MLFLOW PROJECT - CREDIT CARD FRAUD DETECTION")
    print("="*70)
    
    # MLflow autolog
    mlflow.sklearn.autolog(log_models=True, log_input_examples=True)
    
    with mlflow.start_run(run_name="MLProject_CreditCard_Fraud"):
        
        # Log parameters
        print("\nLogging parameters...")
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # Create and train model
        print("\nTraining model...")
        
        # Parse class_weight
        class_weight_val = params['class_weight']
        if class_weight_val == 'None':
            class_weight_val = None
        elif class_weight_val != 'balanced':
            try:
                class_weight_val = eval(class_weight_val)
            except:
                class_weight_val = 'balanced'
        
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'] if params['max_depth'] > 0 else None,
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            class_weight=class_weight_val,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        print("\nMaking predictions...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        y_train_pred = model.predict(X_train)
        
        # Calculate metrics
        print("\nCalculating metrics...")
        
        # Test metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        test_log_loss = log_loss(y_test, y_pred_proba)
        
        # Train metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
        
        # Log metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_metric("test_log_loss", test_log_loss)
        
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_f1_score", train_f1)
        
        # Print results
        print("\n" + "="*70)
        print("TRAINING RESULTS")
        print("="*70)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Training F1 Score: {train_f1:.4f}")
        print(f"\nTest Accuracy:     {test_accuracy:.4f}")
        print(f"Test Precision:    {test_precision:.4f}")
        print(f"Test Recall:       {test_recall:.4f}")
        print(f"Test F1 Score:     {test_f1:.4f}")
        print(f"Test ROC AUC:      {test_roc_auc:.4f}")
        print(f"Test Log Loss:     {test_log_loss:.4f}")
        print("="*70)
        
        # Save model
        print("\nSaving model...")
        mlflow.sklearn.log_model(model, "model", registered_model_name="credit-card-fraud-detection")
        
        print("\nTraining completed successfully!")
        
        return model


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Credit Card Fraud Detection Model')
    
    parser.add_argument('--train-data', type=str, default='creditcard_train_balanced.csv',
                       help='Path to training data')
    parser.add_argument('--test-data', type=str, default='creditcard_test.csv',
                       help='Path to testing data')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of trees in random forest')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum depth of trees (0 for None)')
    parser.add_argument('--min-samples-split', type=int, default=5,
                       help='Minimum samples required to split node')
    parser.add_argument('--min-samples-leaf', type=int, default=2,
                       help='Minimum samples required at leaf node')
    parser.add_argument('--max-features', type=str, default='sqrt',
                       help='Number of features to consider for split')
    parser.add_argument('--class-weight', type=str, default='balanced',
                       help='Class weights for imbalanced data')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Prepare parameters dict
    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'max_features': args.max_features,
        'class_weight': args.class_weight
    }
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(args.train_data, args.test_data)
    
    # Train model
    model = train_model(X_train, X_test, y_train, y_test, params)
    
    print("\nMLflow Project execution completed!")
