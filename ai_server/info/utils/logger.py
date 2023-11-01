# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
from loguru import logger


class MyLogger:
    def __init__(self, log_dir='logs', rotation="00:00", retention=30):
        self.log_dir = log_dir
        self.rotation = rotation
        self.retention = retention
        self.logger = self.configure_logger()

    def configure_logger(self):
        os.makedirs(self.log_dir, exist_ok=True)

        logger.add(
            sink=f"{self.log_dir}/{{time:YYYY-MM-DD}}.log",
            rotation=self.rotation,
            retention=self.retention,
            mode="a+",
            compression="zip",
            enqueue=True,
            backtrace=True,
            encoding="utf-8",
            # format="{time:YYYY-mm-dd HH:mm:ss.SSS} | {thread.name} | {level} | {module}:{line} -  {message}"
        )

        return logger

    def __getattr__(self, level: str):
        return getattr(self.logger, level)
