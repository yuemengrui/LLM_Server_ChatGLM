# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os

FASTAPI_TITLE = 'LLM_Server_ChatGLM'
FASTAPI_HOST = '0.0.0.0'
FASTAPI_PORT = 24621

MODEL_REGISTER_URL = 'http://paimongpt_server:24601/ai/model/register'
THIS_SERVER_HOST = 'http://paimongpt_chatglm_server:' + str(FASTAPI_PORT)

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

CHATGLM_CONFIG = {
    "model_name": "ChatGLM2_6B_32k",
    "model_name_or_path": "/workspace/Models/ChatGLM",
    "device": "cuda"
}

# API LIMIT
API_LIMIT = {
    "chat": "15/minute",
    "token_count": "60/minute",
}
