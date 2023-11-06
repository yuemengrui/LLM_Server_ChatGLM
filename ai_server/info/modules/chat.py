# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
from mylogger import logger
from fastapi import APIRouter, Request
from info import limiter, chatglm
from configs import API_LIMIT
from .protocol import ChatRequest, TokenCountRequest, ErrorResponse, TokenCountResponse
from fastapi.responses import JSONResponse, StreamingResponse
from info.utils.response_code import RET, error_map

router = APIRouter()


@router.api_route('/ai/llm/chat', methods=['POST'], summary="Chat")
@limiter.limit(API_LIMIT['chat'])
def llm_chat(request: Request,
             req: ChatRequest
             ):
    logger.info(str(req.dict()))

    token_counter_resp = chatglm.check_token_len(req.prompt)
    if not token_counter_resp[0]:
        return JSONResponse(ErrorResponse(errcode=RET.TOKEN_OVERFLOW,
                                          errmsg=error_map[
                                                     RET.TOKEN_OVERFLOW] + u"当前prompt token:{} 支持的最大token:{}".format(
                                              token_counter_resp[1], token_counter_resp[2])).dict(), status_code=413)

    if req.stream:
        def stream_generator():
            for resp in chatglm.lets_chat(**req.dict()):
                yield json.dumps(resp, ensure_ascii=False)
            logger.info(str(resp) + '\n')

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        resp = chatglm.lets_chat(**req.dict())
        return JSONResponse(resp)


@router.api_route('/ai/llm/token_count', methods=['POST'], summary="token count")
@limiter.limit(API_LIMIT['token_count'])
def count_token(request: Request,
                req: TokenCountRequest
                ):
    logger.info(str(req.dict()))

    token_counter_resp = chatglm.check_token_len(req.prompt)

    return JSONResponse(TokenCountResponse(model_name=token_counter_resp[3],
                                           prompt=req.prompt,
                                           prompt_tokens=token_counter_resp[1],
                                           max_tokens=token_counter_resp[2],
                                           status='ok' if token_counter_resp[0] else 'token_overflow').dict())
