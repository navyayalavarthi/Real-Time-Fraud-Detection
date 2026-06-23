
import boto3
import json

REGION = "us-east-2"
ENDPOINT_NAME = "fraud-detection-endpoint"

def lambda_handler(event, context):
    try:
        transaction = event.get("transaction", event)
        runtime = boto3.client("sagemaker-runtime", region_name=REGION)

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(transaction)
        )

        result = json.loads(response["Body"].read())
        prob = result["fraud_probability"]

        if prob > 0.7:
            alert = "CRITICAL: High probability fraud"
        elif prob > 0.4:
            alert = "WARNING: Medium probability fraud"
        else:
            alert = "CLEAR: Transaction looks normal"

        return {
            "statusCode": 200,
            "body": json.dumps({
                "fraud_probability": prob,
                "is_fraud": result["is_fraud"],
                "risk_level": result["risk_level"],
                "recommended_action": result["recommended_action"],
                "alert": alert
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
