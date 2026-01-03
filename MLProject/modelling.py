"""
MLflow Project - Model Training Script
=======================================
Script untuk MLProject yang mendukung parameterized training

Author: [Nama Anda]
Date: October 2025
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import argparse
import warnings

warnings.filterwarnings('ignore')


def load_data(train_path, test_path):
    """Load training and testing data"""
    print(f"📁 Loading data...")
    print(f"   Train: {train_path}")
    print(f"   Test: {test_path}")
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    X_train = train_data.drop('quality_encoded', axis=1)
    y_train = train_data['quality_encoded']
    
    X_test = test_data.drop('quality_encoded', axis=1)
    y_test = test_data['quality_encoded']
    
    print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, params):
    """Train model with given parameters"""
    print("\n" + "="*70)
    print("🚀 MLFLOW PROJECT TRAINING")
    print("="*70)
    
    # Set tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Wine_Quality_MLProject")
    
    with mlflow.start_run(run_name="MLProject_Training"):
        
        # Log parameters
        print("\n📝 Logging parameters...")
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # Create and train model
        print("\n🔧 Training model...")
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'] if params['max_depth'] > 0 else None,
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        print("\n🔮 Making predictions...")
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        # Calculate metrics
        print("\n📊 Calculating metrics...")
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Log metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("overfitting_gap", train_accuracy - test_accuracy)
        
        # Print results
        print("\n📊 Model Performance:")
        print(f"   Train Accuracy: {train_accuracy:.4f}")
        print(f"   Test Accuracy:  {test_accuracy:.4f}")
        print(f"   Test Precision: {test_precision:.4f}")
        print(f"   Test Recall:    {test_recall:.4f}")
        print(f"   Test F1 Score:  {test_f1:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Low', 'Medium', 'High']))
        
        # Save artifacts
        print("\n💾 Saving artifacts...")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, 
                             index=['Low', 'Medium', 'High'],
                             columns=['Low', 'Medium', 'High'])
        cm_df.to_csv("confusion_matrix.csv")
        mlflow.log_artifact("confusion_matrix.csv")
        
        # Feature Importances
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importance.to_csv("feature_importances.csv", index=False)
        mlflow.log_artifact("feature_importances.csv")
        
        # Log model
        print("\n💾 Logging model...")
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="WineQualityClassifier_MLProject"
        )
        
        # Tags
        mlflow.set_tag("training_type", "mlflow_project")
        mlflow.set_tag("framework", "scikit-learn")
        
        print("\n✅ Training completed successfully!")
        print(f"✅ Run ID: {mlflow.active_run().info.run_id}")


def main():
    """Main function with argparse"""
    parser = argparse.ArgumentParser(description='Train Wine Quality Model')
    
    parser.add_argument('--train-data', type=str, default='wine_quality_train.csv',
                        help='Path to training data')
    parser.add_argument('--test-data', type=str, default='wine_quality_test.csv',
                        help='Path to testing data')
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Number of trees in random forest')
    parser.add_argument('--max-depth', type=int, default=20,
                        help='Maximum depth of trees (0 for None)')
    parser.add_argument('--min-samples-split', type=int, default=5,
                        help='Minimum samples required to split')
    parser.add_argument('--min-samples-leaf', type=int, default=2,
                        help='Minimum samples required at leaf')
    parser.add_argument('--max-features', type=str, default='sqrt',
                        help='Max features for split')
    
    args = parser.parse_args()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(args.train_data, args.test_data)
    
    # Prepare parameters
    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'max_features': args.max_features
    }
    
    # Train model
    train_model(X_train, X_test, y_train, y_test, params)
    
    print("\n" + "="*70)
    print("🎉 MLPROJECT PIPELINE COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main()
