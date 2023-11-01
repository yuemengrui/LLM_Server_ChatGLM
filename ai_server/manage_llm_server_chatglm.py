# *_*coding:utf-8 *_*
import uvicorn
from info import app
from info.configs import *

if __name__ == '__main__':
    uvicorn.run(app,
                host=FASTAPI_HOST,
                port=FASTAPI_PORT,
                workers=1
                )
