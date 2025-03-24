from __future__ import annotations
from flask import make_response, Request
from http import HTTPStatus

KEY = "content"

# from py.modules.period.period_manager import PeriodManager


def build_response(content: dict = {}, status_code: int = HTTPStatus.OK.value):
    response = make_response({KEY: content}, status_code)
    
    _headers = ["Access-Control-Allow-{}".format(h) for h in ["Origin", "Headers", "Methods"]]
    value = "*"
    _ = [response.headers.add(h, value) for h in _headers]
    # response.headers.add("Access-Control-Allow-Credentials", "true")
    # response.headers.add("Access-Control-Max-Age", "3600")
    print(response.headers)
    if status_code== HTTPStatus.NO_CONTENT.value:
        return ("", HTTPStatus.NO_CONTENT.value, response.headers)
    return response



def get_request_input(request: Request, default=KEY):
    # print("external IP:", requests.get("https://ident.me").content)
    
    if request.method=="OPTIONS":
        return "", HTTPStatus.NO_CONTENT.value

    key = KEY
    request_json = request.get_json()
    

    if request.args and key in request.args:
        return request.args.get(key), HTTPStatus.OK.value
    elif request_json and key in request_json:
        return request_json[key], HTTPStatus.OK.value
    else:
        print("params not defined, defaulting to {}".format(default))

        return default, HTTPStatus.NO_CONTENT.value


    