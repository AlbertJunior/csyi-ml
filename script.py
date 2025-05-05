from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import sklearn
import joblib
import boto3
import pathlib
from io import StringIO 
import argparse
import joblib
import os
import numpy as np
import pandas as pd


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

    
if __name__ == "__main__":

    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--n_estimators", type=int, default=1, required=True)
    parser.add_argument("--random_state", type=int, default=0)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test-dir", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")

    args, _ = parser.parse_known_args()

    print("[INFO] Reading data")
    print(args)
    train_df = pd.read_csv(os.path.join(args.train_dir, args.train_file), header=None)
    test_df = pd.read_csv(os.path.join(args.test_dir, args.test_file), header=None)
    
    print("Building training and testing datasets")
    X_train = train_df.iloc[:, 1:]
    X_test = test_df.iloc[:, 1:]
    y_train = pd.Series(train_df.iloc[:, 0])
    y_test = pd.Series(test_df.iloc[:, 0])
    
    print("Data Shape: ")
    print("---- SHAPE OF TRAINING DATA ----")
    print(X_train.shape)
    print(y_train.shape)
    print("---- SHAPE OF TESTING DATA ----")
    print(X_test.shape)
    print(y_test.shape)
    
    print("Training RandomForest Model.....")
    model =  RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, verbose = 1, n_jobs=-1)
    model.fit(X_train, y_train)
    
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model,model_path)
    print("Model persisted at " + model_path)

    
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test,y_pred_test)
    test_rep = classification_report(y_test,y_pred_test)

    print()
    print("---- METRICS RESULTS FOR TESTING DATA ----")
    print()
    print("Total Rows are: ", X_test.shape[0])
    print('[TESTING] Model Accuracy is: ', test_acc)
    print('[TESTING] Testing Report: ')
    print(test_rep)