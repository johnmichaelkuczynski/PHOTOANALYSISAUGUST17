import boto3
import os

rekog = boto3.client(
    'rekognition',
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

# Just test the API connection with a harmless list call
try:
    response = rekog.list_collections()
    print("✅ SUCCESS. Your Rekognition API key is working.")
    print(response)
except Exception as e:
    print("❌ FAILED. Here's the real error:")
    print(e)