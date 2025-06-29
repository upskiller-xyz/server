# /usr/bin/env python3
# Daylight Factor Estimation Server
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

from flask import Flask, request, jsonify
from flask_cors import CORS
from http import HTTPStatus
import json
import numpy as np
import os
import socket
import tensorflow as tf
import logging

import sys

sys.path.append(".")

from processing.colorconverter import ColorConverter
from processing.encoder import Encoder
from processing.image_manager import ImageManager
from processing.image_transformer import SceneSettings

logging.basicConfig(level=logging.DEBUG)

from processing.coord import Coord
from processing.external import EXTERNAL_KEYS
from processing.prediction_input import PredictionInput, Matrix
from processing import pipeline as p
from processing import pipelineinput as inpt
from processing import stats as st

from processing.utils import get_request_input, build_response

app = Flask("Daylight server")
CORS(app)
socket.socket().setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)


# model = tf.keras.models.load_model("generator_model.keras")


@app.route("/daylight_factor", methods=["POST"])
def daylight_factor():
    params, status_code = get_request_input(request)
    if status_code != HTTPStatus.OK.value:
        return build_response({}, status_code)

    image = params["image"]
    inp = tf.reshape(image, (-1, 256, 256, 3))
    # TODO: load the model
    model = None
    result = model(inp, training=False).numpy()
    if result.size == 0 or np.all(result == 0):
        return build_response({}, HTTPStatus.INTERNAL_SERVER_ERROR.value)
    return build_response(result.tolist(), status_code)


@app.route("/test", methods=["GET"])
def test():
    params, status_code = get_request_input(request)
    if status_code != HTTPStatus.OK.value:
        return build_response({}, status_code)

    return {"content": 1}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(threadName)s - %(module)s - %(funcName)s - %(message)s",
)

SERVER_PORT = int(os.getenv("PORT", 8081))
SERVER_BASE_URL = f"http://localhost:{SERVER_PORT}"


@app.route("/get_df", methods=["POST"])
def get_df():
    """
    Endpoint that receives a set of images per apartment with transformation parameters and returns a estimation matrix for DF factor.
    """

    try:
        fls = request.files.getlist("file")
        image_strings = [x.read() for x in fls]
        angles = json.loads(request.form.get("rotation"))
        coords = json.loads(request.form.get("translation"))

        coord = SceneSettings.to_pixels(Coord(coords["x"], coords["y"]))

        matrices = [Matrix(angles[i], coord) for i in range(len(image_strings))]
        pred_inps = [
            inpt.GetOneDfPipelineInput(
                PredictionInput(id=i, image=image_strings[i], params=matrices[i])
            )
            for i in range(len(image_strings))
        ]
        res = p.GetDfPipeline.run(inpt.GetDfPipelineInput(pred_inps))
        if res.value.np_image.size == 0 or np.all(res.value.np_image == 0):
            return build_response(
                {EXTERNAL_KEYS.CONTENT.value: "", EXTERNAL_KEYS.SUCCESS.value: False},
                HTTPStatus.INTERNAL_SERVER_ERROR.value,
            )

        res_string = Encoder.np_to_base64(res.value.np_image)
        return (
            jsonify(
                {
                    EXTERNAL_KEYS.CONTENT.value: res_string,
                    EXTERNAL_KEYS.SUCCESS.value: True,
                }
            ),
            HTTPStatus.OK.value,
        )
    except Exception as e:
        app.logger.error(f"Error in /get_df: {e}", exc_info=True)
        return (
            jsonify(
                {EXTERNAL_KEYS.CONTENT.value: "", EXTERNAL_KEYS.SUCCESS.value: False}
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@app.route("/get_stats", methods=["POST"])
def get_stats():
    """
    Endpoint that receives a set of images per apartment with transformation parameters and returns a estimation matrix for DF factor.
    """
    try:
        res = next(iter(request.files.values())).read()
        inp = ImageManager.load_image(res)[:, :, :3]
        matrix = ColorConverter.get_values(inp)
        statspack = st.StatsPack.build(matrix)
        if len(statspack.content) == 0:
            return build_response(
                {EXTERNAL_KEYS.METRICS.value: {}, EXTERNAL_KEYS.SUCCESS.value: False},
                HTTPStatus.INTERNAL_SERVER_ERROR.value,
            )
        res = statspack.out
        res.update(
            {
                EXTERNAL_KEYS.SUCCESS.value: True,
            }
        )
        return jsonify(res), HTTPStatus.OK.value
    except Exception as e:
        app.logger.error(f"Error in /get_stats: {e}", exc_info=True)
        return (
            jsonify(
                {EXTERNAL_KEYS.METRICS.value: {}, EXTERNAL_KEYS.SUCCESS.value: False}
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@app.route("/to_values", methods=["POST"])
def to_values():
    """
    Endpoint that receives a daylight factor estimation expressed in RGB and returns the same estimation expressed in DF values.
    """

    try:
        res = next(iter(request.files.values())).read()
        inp = ImageManager.load_image(res)[:, :, :3]
        matrix = ColorConverter.get_values(inp)
        res = Encoder.np_to_base64(matrix)
        return (
            jsonify(
                {EXTERNAL_KEYS.SUCCESS.value: True, EXTERNAL_KEYS.CONTENT.value: res}
            ),
            HTTPStatus.OK.value,
        )
    except Exception as e:
        app.logger.error(f"Error: {e}", exc_info=True)
        return (
            jsonify(
                {EXTERNAL_KEYS.METRICS.value: {}, EXTERNAL_KEYS.SUCCESS.value: False}
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@app.route("/to_rgb", methods=["POST"])
def get_rgb():
    """
    Endpoint that receives a daylight factor estimation expressed inDF values and returns the same estimation expressed in RGB.
    """
    try:
        res = next(iter(request.files.values())).read()
        try:
            inp = Encoder.base64_to_np(res)
        except Exception as e:
            app.logger.error(f"Decoding error: {e}", exc_info=True)
            return (
                jsonify(
                    {
                        EXTERNAL_KEYS.SUCCESS.value: False,
                        "error": "Invalid input encoding",
                    }
                ),
                HTTPStatus.BAD_REQUEST.value,
            )


        matrix = ColorConverter.values_to_image(inp)
        res = Encoder.np_to_base64(matrix)
        return (
            jsonify(
                {EXTERNAL_KEYS.SUCCESS.value: True, EXTERNAL_KEYS.CONTENT.value: res}
            ),
            HTTPStatus.OK.value,
        )
    except Exception as e:
        app.logger.error(f"Error: {e}", exc_info=True)
        return (
            jsonify(
                {EXTERNAL_KEYS.METRICS.value: {}, EXTERNAL_KEYS.SUCCESS.value: False}
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


################################################################################

if __name__ == "__main__":

    # if INITIALIZATION_SUCCESSFUL:
    app.logger.info(
        f"Flask app '{app.name}' starting on host 0.0.0.0, port {SERVER_PORT}. Debug mode: {app.debug}"
    )
    app.run(debug=True, host="0.0.0.0", port=8081)
