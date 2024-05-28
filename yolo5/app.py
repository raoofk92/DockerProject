import time
from pathlib import Path
from flask import Flask, request , jsonify
from detect import run
import uuid
import yaml
from loguru import logger
import os
import boto3
from pymongo import mongo_client

images_bucket = os.environ['BUCKET_NAME']

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

MONGO_URI = "mongodb://mongo1:27017/"
DATABASE_NAME = "Raoof_Database"
COLLECTION_NAME = "Raoof_Collection"


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Generates a UUID for this current prediction HTTP request. This id can be used as a reference in logs to identify and track individual prediction requests.
    prediction_id = str(uuid.uuid4())

    logger.info(f'prediction: {prediction_id}. start processing')

    # Receives a URL parameter representing the image to download from S3
    s3_img_path = request.args.get('imgName')
    img_name = s3_img_path[s3_img_path.rindex("/")+1:]


    s3 = boto3.client('s3')
    s3.download_file(images_bucket,s3_img_path, img_name)
    
    #  The bucket name is provided as an env var BUCKET_NAME.
    original_img_path = img_name

    logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')

    # Predicts the objects in the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path,
        project='static/data',
        name=prediction_id,
        save_txt=True
    )

    logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

    # This is the path for the predicted image with labels
    # The predicted image typically includes bounding boxes drawn around the detected objects, along with class labels and possibly confidence scores.
    try:
        predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')
        s3_predicted_directory_path = 'predicted'
        predicted_file_name = f'{Path(img_name).stem}-predicted{Path(img_name).suffix}'
        full_name_s3 = f'{s3_predicted_directory_path}/{prediction_id}/{predicted_file_name}'
        s3.upload_file(predicted_img_path, images_bucket, full_name_s3)
        logger.info(f'Uploaded predicted image to S3: {full_name_s3}')
    except Exception as e:
        logger.error(f'Error uploading predicted image to S3: {str(e)}')
        return jsonify({"error": "Failed to upload predicted image to S3"}), 500

    # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).
    try:
        predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')
        s3_predicted_directory_path = 'predicted'
        predicted_file_name = f'{Path(img_name).stem}-predicted{Path(img_name).suffix}'
        full_name_s3 = f'{s3_predicted_directory_path}/{prediction_id}/{predicted_file_name}'
        s3.upload_file(str(predicted_img_path), images_bucket, full_name_s3)
        logger.info(f'Uploaded predicted image to S3: {full_name_s3}')
    except Exception as e:
        logger.error(f'Error uploading predicted image to S3: {str(e)}')
        return jsonify({"error": "Failed to upload predicted image to S3"}), 500


    # Parse prediction labels and create a summary
    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')
    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
                'class': names[int(l[0])],
                'cx': float(l[1]),
                'cy': float(l[2]),
                'width': float(l[3]),
                'height': float(l[4]),
            } for l in labels]
    elif not pred_summary_path.exists():
        logger.error(f'prediction: {prediction_id}. prediction result not found.')
        return jsonify({"error": "prediction result not found"}), 404

    with open(pred_summary_path) as f:
        labels = [line.split() for line in f.read().splitlines()]
        labels = [{
            'class': names[int(l[0])],
            'cx': float(l[1]),
            'cy': float(l[2]),
            'width': float(l[3]),
            'height': float(l[4])
        } for l in labels]


        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': original_img_path,
            'predicted_img_path': predicted_img_path,
            'labels': labels,
            'time': time.time()
        }

        # TODO store the prediction_summary in MongoDB
    try:
        client = mongo_client(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        collection.insert_one(prediction_summary)
        logger.info(f'Prediction summary stored in MongoDB: {prediction_summary}')
    except Exception as e:
        logger.error(f'Error storing prediction summary in MongoDB: {str(e)}')
        return jsonify({"error": "Failed to store prediction summary in MongoDB"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)