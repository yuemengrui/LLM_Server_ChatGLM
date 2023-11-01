# *_*coding:utf-8 *_*
import time
from info.configs import *
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from info.utils.logger import MyLogger

limiter = Limiter(key_func=lambda *args, **kwargs: '127.0.0.1')
app = FastAPI(title="LLM_Server_ChatGLM")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

logger = MyLogger()


@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"start request {request.method} {request.url.path}")
    start = time.time()

    response = await call_next(request)

    cost = time.time() - start
    logger.info(f"end request {request.method} {request.url.path} {cost:.3f}s")
    return response

from info.libs.ai.models.chatglm import ChatGLM

chatglm = ChatGLM(logger=logger, **CHATGLM_CONFIG)


from info.modules import register_router

register_router(app)
