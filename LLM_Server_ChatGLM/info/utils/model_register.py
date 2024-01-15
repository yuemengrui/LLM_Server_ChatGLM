# *_*coding:utf-8 *_*
# @Author : YueMengRui
import requests
from configs import MODEL_REGISTER_URL, THIS_SERVER_HOST


def register_model_to_server(model_name: str):
    req_data = {
        "type": "llm",
        "model_name": model_name,
        "url_prefix": THIS_SERVER_HOST,
        "info": {}
    }

    _ = requests.post(url=MODEL_REGISTER_URL, json=req_data)
