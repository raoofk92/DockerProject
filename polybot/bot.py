import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import requests
import boto3
from botocore.exceptions import NoCredentialsError


class Bot:
    def __init__(self, token, telegram_chat_url):
        self.telegram_bot_client = telebot.TeleBot(token)
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)
        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def upload_to_s3(self, file_path, bucket_name, s3_file_name):
        s3 = boto3.client('s3')
        try:
            s3.upload_file(file_path, bucket_name, s3_file_name)
            logger.info(f"Upload Successful: {file_path} to bucket {bucket_name} as {s3_file_name}")
        except FileNotFoundError:
            logger.error(f"The file was not found: {file_path}")
            return None
        except NoCredentialsError:
            logger.error("Credentials not available")
            return None

        return f"s3://{bucket_name}/{s3_file_name}"

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(chat_id, InputFile(img_path))

    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class ObjectDetectionBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            try:
                photo_path = self.download_user_photo(msg)
                logger.info(f"Photo downloaded to: {photo_path}")

                s3_bucket_name = os.environ['S3_BUCKET_NAME']
                s3_file_name = os.path.basename(photo_path)
                s3_url = self.upload_to_s3(photo_path, s3_bucket_name, s3_file_name)

                if not s3_url:
                    self.send_text(msg['chat']['id'], "Failed to upload image to S3.")
                    return

                logger.info(f"Photo uploaded to S3: {s3_url}")

                yolo5_url = os.environ['YOLO5_URL']
                response = requests.post(f"{yolo5_url}/predict?imgName={s3_file_name}")

                if response.status_code == 200:
                    result = response.json()
                    self.send_text(msg['chat']['id'], f"Detection results: {result}")

                    if 'predicted_img_path' in result:
                        local_predicted_img_path = result['predicted_img_path']
                        self.send_photo(msg['chat']['id'], local_predicted_img_path)
                else:
                    self.send_text(msg['chat']['id'], "Failed to get prediction from YOLO5 service.")
                    logger.error(f"YOLO5 response error: {response.status_code}, {response.text}")

            except Exception as e:
                logger.error(f"Error in handling message: {str(e)}")
                self.send_text(msg['chat']['id'], "An error occurred while processing your image.")
        else:
            self.send_text(msg['chat']['id'], "Please send a photo for object detection.")