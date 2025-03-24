# /usr/bin/env python3
from __future__ import annotations

from flask import Flask, request
from flask_cors import CORS
from http import HTTPStatus
import socket
import tensorflow as tf

from src.utils import get_request_input, build_response


app = Flask("Daylight server")
CORS(app)
socket.socket().setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)

model = tf.keras.models.load_model("generator_model.keras")

@app.route('/daylight_factor', methods=['POST'])
def daylight_factor():
    params, status_code = get_request_input(request)
    print("PARAMS::", params)
    if status_code != HTTPStatus.OK.value:
        return build_response({}, status_code)
    
    image = params["image"]
    inp = tf.reshape(image, (-1, 256, 256, 3))
    result = model(inp, training=False).numpy()
    return build_response(result.tolist(), status_code)

@app.route('/test', methods=['GET'])
def test():
    params, status_code = get_request_input(request)
    print("PARAMS::", params)
    if status_code != HTTPStatus.OK.value:
        return build_response({}, status_code)
    
    return {"content": 1}

if __name__ == '__main__':
	app.run(debug=True, host="0.0.0.0", port=8081)