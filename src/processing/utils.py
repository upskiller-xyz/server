# Daylight Factor Simulation Server
# Copyright (C) 2024 BIMTech Innovations AB (developed by the Upskiller group)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU GPL v3.0 along with this program.
# If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from flask import make_response, Request
from http import HTTPStatus

from .external import EXTERNAL_KEYS


# from py.modules.period.period_manager import PeriodManager


def build_response(content: dict = {}, status_code: int = HTTPStatus.OK.value):
    response = make_response({EXTERNAL_KEYS.CONTENT.value: content}, status_code)

    _headers = [
        "Access-Control-Allow-{}".format(h) for h in ["Origin", "Headers", "Methods"]
    ]
    value = "*"
    _ = [response.headers.add(h, value) for h in _headers]
    # response.headers.add("Access-Control-Allow-Credentials", "true")
    # response.headers.add("Access-Control-Max-Age", "3600")
    print(response.headers)
    if status_code == HTTPStatus.NO_CONTENT.value:
        return ("", HTTPStatus.NO_CONTENT.value, response.headers)
    return response


def get_request_input(request: Request, default=EXTERNAL_KEYS.CONTENT.value):
    # print("external IP:", requests.get("https://ident.me").content)

    if request.method == "OPTIONS":
        return "", HTTPStatus.NO_CONTENT.value

    key = EXTERNAL_KEYS.CONTENT.value
    request_json = request.get_json()

    if request.args and key in request.args:
        return request.args.get(key), HTTPStatus.OK.value
    elif request_json and key in request_json:
        return request_json[key], HTTPStatus.OK.value
    else:
        print("params not defined, defaulting to {}".format(default))

        return default, HTTPStatus.NO_CONTENT.value


TARGET_SIZE = 256
OUTPUT_SIZE = 128
