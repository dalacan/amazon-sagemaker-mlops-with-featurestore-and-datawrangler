import json
import logging
import os

import boto3
import pandas as pd

logger = logging.getLogger()

region = os.environ["region"]
endpoint_name = os.environ["endpoint_name"]
content_type = os.environ["content_type"]
nyctaxi_fg_name = os.environ["nyctaxi_fg_name"]

boto_session = boto3.Session(region_name=region)
featurestore_runtime = boto_session.client(
    service_name="sagemaker-featurestore-runtime", region_name=region
)
client_sm = boto_session.client("sagemaker-runtime", region_name=region)

col_order = [
    "passenger_count",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "geo_distance",
    "hour",
    "weekday",
    "month"
]


def lambda_handler(event, context):
    # Get data from online feature store
    logger.info(event)
    nyctaxi_id = str(event["queryStringParameters"]["FS_ID"])

    nyctaxi_response = featurestore_runtime.get_record(
        FeatureGroupName=nyctaxi_fg_name,
        RecordIdentifierValueAsString=str(nyctaxi_id),
    )

    if nyctaxi_response.get("Record"):
        nyctaxi_record = nyctaxi_response["Record"]
        nyctaxi_df = pd.DataFrame(nyctaxi_record).set_index("FeatureName")
    else:
        logging.info("No Record returned / Record Key in NYCTAXI feature group\n")
        return {
            "statusCode": 404,
            "body": json.dumps(
                {"Error": "Record not found in NYCTAXI feature group"}
            ),
        }

    try:
        data_input = ",".join(nyctaxi_df["ValueAsString"])

        logging.info("data_input: ", data_input)
        response = client_sm.invoke_endpoint(
            EndpointName=endpoint_name, Body=data_input, ContentType=content_type
        )

        fare_amount = json.loads(response["Body"].read())
        logging.info(f"fare_amount: {fare_amount}")

        return {
            "statusCode": 200,
            "body": json.dumps({"nyctaxi_id": nyctaxi_id, "fare_amount": fare_amount}),
        }
    except Exception:
        logging.exception(f"internal error")
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"Error": f"internal error. Check Logs for more details"}
            ),
        }
