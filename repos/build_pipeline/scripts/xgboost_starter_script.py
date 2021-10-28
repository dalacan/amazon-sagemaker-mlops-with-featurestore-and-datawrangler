import argparse
import json
import os
import pickle

import boto3
import pandas as pd
import xgboost as xgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here
    parser.add_argument("--num_round", type=int, default=999)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--nfold", type=int, default=5)
    parser.add_argument("--early_stopping_rounds", type=int, default=10)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--min_child_weight", type=int, default=300)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument(
        "--train_data_path", type=str, default=os.environ.get("SM_CHANNEL_TRAIN")
    )
    parser.add_argument("--bucket", type=str)
    parser.add_argument("--object", type=str)

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )

    args = parser.parse_args()

    s3_client = boto3.client("s3")

    data = pd.read_csv(f"{args.train_data_path}/train.csv")
    train = data.drop("fare_amount", axis=1)
    label = pd.DataFrame(data["fare_amount"])
    dtrain = xgb.DMatrix(train, label=label)

    params = {"max_depth": args.max_depth, "eta": args.eta, "objective": args.objective, "gamma": args.gamma, "min_child_weight": args.min_child_weight, "subsample": args.subsample }
    num_boost_round = args.num_round
    nfold = args.nfold
    early_stopping_rounds = args.early_stopping_rounds

    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        nfold=nfold,
        early_stopping_rounds=early_stopping_rounds,
        metrics=["rmse","mae"],
        seed=0,
    )

    print(f"[0]#011train-rmse:{cv_results.iloc[-1]['train-rmse-mean']}")
    print(f"[1]#011validation-rmse:{cv_results.iloc[-1]['test-rmse-mean']}")

    metrics_data = {
        "regression_metrics": {
            "validation:rmse": {
                "value": cv_results.iloc[-1]["test-rmse-mean"],
                "standard_deviation": cv_results.iloc[-1]["test-rmse-std"],
            },
            "train:rmse": {
                "value": cv_results.iloc[-1]["train-rmse-mean"],
                "standard_deviation": cv_results.iloc[-1]["train-rmse-std"],
            },
            "validation:mae": {
                "value": cv_results.iloc[-1]["test-mae-mean"],
                "standard_deviation": cv_results.iloc[-1]["test-mae-std"],
            },
            "train:mae": {
                "value": cv_results.iloc[-1]["train-mae-mean"],
                "standard_deviation": cv_results.iloc[-1]["train-mae-std"],
            },
        }
    }
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=len(cv_results))

    # Save the model to the location specified by ``model_dir``
    metrics_location = args.output_data_dir + "/metrics.json"
    model_location = args.model_dir + "/xgboost-model"

    with open(metrics_location, "w") as f:
        json.dump(metrics_data, f)

    s3_client.upload_file(
        Filename=metrics_location, Bucket=args.bucket, Key=args.object
    )

    with open(model_location, "wb") as f:
        pickle.dump(model, f)
