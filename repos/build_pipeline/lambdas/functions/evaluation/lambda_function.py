import json
import logging
import os

import boto3

s3_client = boto3.client("s3")
topic_arn = os.getenv("TOPIC_ARN")


def lambda_handler(event, context):
    """ """
    logging.info(event)

    bucket_name = event["bucket_name"]
    key_name = event["key_name"]

    metric_object = s3_client.get_object(Bucket=bucket_name, Key=key_name)

    metric_data = json.loads(metric_object['Body'].read())
    
    evaluation_metric = metric_data['regression_metrics']['validation:rmse']['value']

    return {"statusCode": 200, "body": float(evaluation_metric)}
