# *_*coding:utf-8 *_*
# @Author : YueMengRui

FASTAPI_HOST = '0.0.0.0'
FASTAPI_PORT = 5000

CHATGLM_CONFIG = {
    "model_name": "ChatGLM2_6B_32k",
    "model_name_or_path": "",
    "device": "cuda"
}

# API LIMIT
API_LIMIT = {
    "chat": "15/minute",
    "token_count": "60/minute",
}
