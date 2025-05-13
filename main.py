# /usr/bin/env python3
from __future__ import annotations

from flask import Flask, request
from flask_cors import CORS
from http import HTTPStatus
import socket
import tensorflow as tf

from src.utils import get_request_input, build_response

# Libraries for Grasshopper Orchestrator
import requests
import numpy as np
from flask import jsonify, send_file, make_response
from PIL import Image
import os
import io
import json
import logging
from google.cloud import storage
# Outsourced functions for Grasshopper Orchestrator
from src.df_inference import preprocess_for_df_ml, postprocess_df_ml_prediction
from src.df_align_image import align_df_image
from src.df_image_processing import generate_value_matrix_for_single_gan_image, aggregate_multiple_value_matrices_gan, calculate_gan_metrics_from_values, render_final_gan_image_from_values, create_color_maps_from_data 

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

########## GH-ORCHESTRATOR ##########

### Configurations ###
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(module)s - %(funcName)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_MODEL_FILENAME = 'generator.keras' # pix2pix modell
JSON_COLORSCALE_FILENAME = 'DF-colorscale.json' # json file for matching df-values with colors

ML_MODEL_GCS_URI = os.getenv('ML_MODEL_GCS_URI', 'gs://df_experiments/02cdb119-e772-45fc-84fe-03f0c91021fc/generator.keras') #Keras-File from random training

JSON_COLORSCALE_GCS_BUCKET_NAME = os.getenv('JSON_COLORSCALE_GCS_BUCKET_NAME', 'daylight_analysis_assets')
JSON_COLORSCALE_GCS_BLOB_NAME = os.getenv('JSON_COLORSCALE_GCS_BLOB_NAME', 'colorscale_df.json')

ml_model = None
gan_value_map_reverse = {}
gan_known_colors_rgb = np.array([])
gan_known_colors_lab = np.array([])
gan_known_values = np.array([])

INITIALIZATION_SUCCESSFUL = False
SERVER_PORT = int(os.getenv('PORT', 8081))
SERVER_BASE_URL = f"http://localhost:{SERVER_PORT}"

EXPECTED_ML_IMG_SIZE = (256, 256)
ML_NORMALIZATION_RANGE = (-1, 1) # Training uses in image_manager.py "(image / 127.5) - 1" to normalize to [-1, 1]
SCENE_WIDTH_MM = 12800.0 # Dimension of the image area in Rhino (mm)
SCENE_HEIGHT_MM = 12800.0 
GAN_PIXELS_PER_MM_X_ML = EXPECTED_ML_IMG_SIZE[0] / SCENE_WIDTH_MM
GAN_PIXELS_PER_MM_Y_ML = EXPECTED_ML_IMG_SIZE[1] / SCENE_HEIGHT_MM


def initialize_server_resources():
    global ml_model, INITIALIZATION_SUCCESSFUL
    global gan_value_map_reverse, gan_known_colors_rgb, \
           gan_known_colors_lab, gan_known_values
    
    # Temporary status, gonna be assigned at the end INITIALIZATION_SUCCESSFUL
    current_function_success_status = True 
    app.logger.info("Starting server resource initialization...")

    # 1. Load ML Modell
    if not ML_MODEL_GCS_URI:
        app.logger.error("ML_MODEL_GCS_URI environment variable not set. Cannot load model.")
        current_function_success_status = False
    else:
        try:
            app.logger.info(f"Attempting to load ML model directly from GCS URI: {ML_MODEL_GCS_URI}")
            ml_model = tf.keras.models.load_model(ML_MODEL_GCS_URI, compile=False)
            app.logger.info(f"ML model ('{os.path.basename(ML_MODEL_GCS_URI)}') successfully loaded from GCS.")
        except Exception as e:
            app.logger.error(f"Error loading ML model from GCS URI '{ML_MODEL_GCS_URI}': {e}", exc_info=True)
            current_function_success_status = False

    # 2. Load Color Matching JSON
    if current_function_success_status: 
        if not JSON_COLORSCALE_GCS_BUCKET_NAME or not JSON_COLORSCALE_GCS_BLOB_NAME:
            app.logger.error("JSON_COLORSCALE GCS Bucket or Blob name not set in env variables. Cannot load color map.")
            current_function_success_status = False
        else:
            storage_client = None
            try:
                storage_client = storage.Client() # Authenticate
                bucket = storage_client.bucket(JSON_COLORSCALE_GCS_BUCKET_NAME)
                blob = bucket.blob(JSON_COLORSCALE_GCS_BLOB_NAME)
                
                app.logger.info(f"Downloading color map from GCS: gs://{JSON_COLORSCALE_GCS_BUCKET_NAME}/{JSON_COLORSCALE_GCS_BLOB_NAME}")
                json_string_content = blob.download_as_text(encoding="utf-8-sig") # utf-8-sig für BOM
                json_data_from_gcs = json.loads(json_string_content)
                
                maps_data = create_color_maps_from_data(json_data_from_gcs, app.logger)
                
                gan_value_map_reverse.clear()
                gan_value_map_reverse.update(maps_data['value_map_reverse'])
                
                # Global Variablen Update
                gan_known_colors_rgb = maps_data['known_colors_rgb']
                gan_known_colors_lab = maps_data['known_colors_lab']
                gan_known_values = maps_data['known_values']
                
                app.logger.info(f"Color map ('{JSON_COLORSCALE_GCS_BLOB_NAME}') successfully downloaded from GCS and processed.")

            except FileNotFoundError: 
                app.logger.error(f"Color map blob '{JSON_COLORSCALE_GCS_BLOB_NAME}' not found in GCS bucket '{JSON_COLORSCALE_GCS_BUCKET_NAME}'.")
                current_function_success_status = False
            except json.JSONDecodeError as e:
                 app.logger.error(f"Error decoding JSON from GCS blob '{JSON_COLORSCALE_GCS_BLOB_NAME}': {e}", exc_info=True)
                 current_function_success_status = False
            except Exception as e:
                app.logger.error(f"Error loading/processing Color map from GCS (gs://{JSON_COLORSCALE_GCS_BUCKET_NAME}/{JSON_COLORSCALE_GCS_BLOB_NAME}): {e}", exc_info=True)
                current_function_success_status = False

    INITIALIZATION_SUCCESSFUL = current_function_success_status
    if not INITIALIZATION_SUCCESSFUL:
        app.logger.critical("!!! Overall server resource initialization FAILED (GCS). Review logs. !!!")
    else:
        app.logger.info("Overall server resources initialization process (GCS) complete.")



### Helpers for HTTP Calls  ###
def internal_post_image_to_endpoint(endpoint_url, image_data_bytes, image_filename="image.png", additional_form_data=None, timeout_seconds=30):
    """
    Sends image data and optionally additional form data to an internal endpoint.
    Adjusts the file field name based on the target endpoint.
    """
    file_field_name = 'image_file'
    files = {file_field_name: (image_filename, image_data_bytes, 'image/png')}

    try:
        log_data_info = f"(data keys: {list(additional_form_data.keys())})" if additional_form_data else "(no additional form data)"
        app.logger.debug(f"Internal POST to {endpoint_url} with image '{image_filename}' (field: '{file_field_name}') {log_data_info}")
        
        response = requests.post(endpoint_url, files=files, data=additional_form_data, timeout=timeout_seconds)
        
        response.raise_for_status() 
        return response
    except requests.exceptions.Timeout:
        app.logger.error(f"Timeout during internal request to {endpoint_url}")
        raise
    except requests.exceptions.RequestException as e:
        failed_url = e.request.url if e.request else endpoint_url
        app.logger.error(f"Internal request to {failed_url} failed: {e}", exc_info=True)
        raise

def internal_post_json_to_endpoint(endpoint_url, json_data):
    """"Sends JSON data to an internal endpoint."""
    try:
        app.logger.debug(f"Internal POST to {endpoint_url} with JSON: {json_data}")
        response = requests.post(endpoint_url, json=json_data)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Internal request to {endpoint_url} failed: {e}")
        raise


### Endpoints Grasshopper Orchestrator  ###
@app.route('/df_ml_inference', methods=['POST'])
def df_ml_inference_route():
    if not INITIALIZATION_SUCCESSFUL or not ml_model:
        app.logger.error("/df_ml_inference: Service not available due to initialization failure or model not loaded.")
        return jsonify({"error": "ML inference service not available.", "success": False}), HTTPStatus.SERVICE_UNAVAILABLE.value

    image_file = request.files.get('image_file')
    if not image_file:
        return jsonify({"error": "No 'image_file' provided in the request.", "success": False}), HTTPStatus.BAD_REQUEST.value

    
    app.logger.info(f"[{request.remote_addr}] /df_ml_inference received image: {image_file.filename}")
    try:
        # Load image from bytes and prepare as RGB
        image_pil_rgb = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        
        # Preprocessing
        preprocessed_tensor = preprocess_for_df_ml(image_pil_rgb, app.logger, EXPECTED_ML_IMG_SIZE, ML_NORMALIZATION_RANGE)
        
        # ML-Inference
        app.logger.debug("Starting ML prediction...")
        prediction_tensor = ml_model.predict(preprocessed_tensor)
        app.logger.debug("ML prediction finished.")
        
        # Postprocessing
        output_image_pil_rgba = postprocess_df_ml_prediction(prediction_tensor, app.logger, ML_NORMALIZATION_RANGE)
        
        # Return result image as bytes
        img_byte_arr = io.BytesIO()
        output_image_pil_rgba.save(img_byte_arr, format='PNG') 
        img_byte_arr.seek(0)
        
        app.logger.info(f"[{request.remote_addr}] /df_ml_inference successfully processed image '{image_file.filename}'.")
        return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name=f"ml_output_{image_file.filename}")

    except Exception as e:
        app.logger.error(f"Error during ML inference for image '{image_file.filename}': {e}", exc_info=True)
        return jsonify({"error": f"ML inference processing error: {str(e)}", "success": False}), HTTPStatus.INTERNAL_SERVER_ERROR.value



@app.route('/df_align', methods=['POST'])
def df_align_route():
    if not INITIALIZATION_SUCCESSFUL:
        app.logger.error("/df_align: Service not available due to general initialization failure.")
        return jsonify({"error": "Alignment service not available (init failed).", "success": False}), HTTPStatus.SERVICE_UNAVAILABLE.value

    image_file = request.files.get('image_file') 
    rotation_rad_str = request.form.get('rotation_rad')
    translation_mm_xyz_str = request.form.get('translation_mm_xyz') # JSON-String "[x,y,z]"

    if not all([image_file, rotation_rad_str, translation_mm_xyz_str]):
        missing_fields = []
        if not image_file: missing_fields.append("'image_file'")
        if not rotation_rad_str: missing_fields.append("'rotation_rad'")
        if not translation_mm_xyz_str: missing_fields.append("'translation_mm_xyz'")
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}.", "success": False}), HTTPStatus.BAD_REQUEST.value
    
    original_filename = image_file.filename if image_file.filename else "unknown_image"
    app.logger.info(f"[{request.remote_addr}] /df_align received image '{original_filename}' for alignment. R={rotation_rad_str}, T={translation_mm_xyz_str}")

    try:
        # Load RGBA from /df_ml_inference
        image_pil_rgba = Image.open(io.BytesIO(image_file.read()))
        
        rotation_rad = float(rotation_rad_str)
        translation_mm_xyz = json.loads(translation_mm_xyz_str)

        if not (isinstance(translation_mm_xyz, list) and len(translation_mm_xyz) >= 2 and all(isinstance(c, (int, float)) for c in translation_mm_xyz[:2])):
             raise ValueError("translation_mm_xyz must be a list of at least two numbers, e.g., [x, y, z] or [x,y]")

        transformed_pil_rgba = align_df_image(image_pil_rgba, rotation_rad, translation_mm_xyz, EXPECTED_ML_IMG_SIZE, GAN_PIXELS_PER_MM_X_ML, GAN_PIXELS_PER_MM_Y_ML, app.logger)
        
        img_byte_arr = io.BytesIO()
        transformed_pil_rgba.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        app.logger.info(f"[{request.remote_addr}] /df_align successfully aligned image '{original_filename}'.")
        return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name=f"aligned_{original_filename}")

    except ValueError as ve: #
        app.logger.error(f"Error in /df_align due to invalid parameter format for '{original_filename}': {ve}", exc_info=True)
        return jsonify({"error": f"Invalid parameter format: {str(ve)}", "success": False}), HTTPStatus.BAD_REQUEST.value
    except Exception as e:
        app.logger.error(f"Error during alignment for image '{original_filename}': {e}", exc_info=True)
        return jsonify({"error": f"Image alignment processing error: {str(e)}", "success": False}), HTTPStatus.INTERNAL_SERVER_ERROR.value



@app.route('/gh_images_to_values', methods=['POST'])
def gh_images_to_values_route():
    if not INITIALIZATION_SUCCESSFUL:
        return jsonify({"error": "Service not available (init failed).", "success": False}), HTTPStatus.SERVICE_UNAVAILABLE.value

    files = request.files.getlist("aligned_images[]") 
    if not files:
        return jsonify({"error": "No image files ('aligned_images[]') provided.", "success": False}), HTTPStatus.BAD_REQUEST.value
    
    app.logger.info(f"[{request.remote_addr}] /gh_images_to_values received {len(files)} aligned image(s).")
    try:
        value_matrices_list_np = []
        non_white_masks_list_np = []

        for i, file_storage in enumerate(files):
            # Pics in RGBA
            image_pil_rgba = Image.open(io.BytesIO(file_storage.read()))
            
            v_matrix_np, nw_mask_np = generate_value_matrix_for_single_gan_image(image_pil_rgba, app.logger, gan_known_colors_lab, gan_known_values)
            value_matrices_list_np.append(v_matrix_np)
            non_white_masks_list_np.append(nw_mask_np)
        
        # Aggregating matrices
        summed_values_np, combined_mask_np = aggregate_multiple_value_matrices_gan(value_matrices_list_np, non_white_masks_list_np, app.logger, EXPECTED_ML_IMG_SIZE)
        
        app.logger.info(f"[{request.remote_addr}] /gh_images_to_values processing complete. Output summed_value_matrix shape: {summed_values_np.shape}")
        return jsonify({
            "summed_value_matrix": summed_values_np.tolist(),
            "combined_non_white_mask": combined_mask_np.tolist(),
            "num_source_images_processed": len(files)
        }), HTTPStatus.OK.value

    except Exception as e:
        app.logger.error(f"Error in /gh_images_to_values: {e}", exc_info=True)
        return jsonify({"error": f"Error processing images to values: {str(e)}", "success": False}), HTTPStatus.INTERNAL_SERVER_ERROR.value



@app.route('/gh_metrics_and_render', methods=['POST'])
def gh_metrics_and_render_route():
    if not INITIALIZATION_SUCCESSFUL:
        return jsonify({"error": "Service not available (init failed).", "success": False}), HTTPStatus.SERVICE_UNAVAILABLE.value

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided (expected summed_value_matrix and combined_non_white_mask).", "success": False}), HTTPStatus.BAD_REQUEST.value

    summed_value_matrix_json = data.get('summed_value_matrix')
    combined_non_white_mask_json = data.get('combined_non_white_mask')

    if summed_value_matrix_json is None or combined_non_white_mask_json is None:
        return jsonify({"error": "Missing 'summed_value_matrix' or 'combined_non_white_mask' in JSON payload.", "success": False}), HTTPStatus.BAD_REQUEST.value
    
    app.logger.info(f"[{request.remote_addr}] /gh_metrics_and_render received value matrix for final processing.")
    try:
        values_np = np.array(summed_value_matrix_json, dtype=np.float32)
        mask_np = np.array(combined_non_white_mask_json, dtype=bool)

        # Calculate metrics
        avg_value, ratio_gt1 = calculate_gan_metrics_from_values(values_np, mask_np, app.logger)
        
        # "render" final picture
        final_image_pil = render_final_gan_image_from_values(values_np, mask_np, app.logger, EXPECTED_ML_IMG_SIZE, gan_value_map_reverse)
        
        app.logger.info(f"[{request.remote_addr}] /gh_metrics_and_render: AvgValue={avg_value:.3f}, RatioGT1={ratio_gt1:.3f}")

        img_byte_arr = io.BytesIO()
        final_image_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        response = make_response(send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name="final_rendered_image.png"))
        # Header for GH-Script
        response.headers['X-Processing-Success'] = 'True' 
        response.headers['X-Average-Value'] = str(round(avg_value, 5))
        response.headers['X-Ratio-Pixels-GT1'] = str(round(ratio_gt1, 5))

        return response

    except Exception as e:
        app.logger.error(f"Error in /gh_metrics_and_render: {e}", exc_info=True)
        return jsonify({"error": f"Error during metrics calculation or final rendering: {str(e)}", "success": False}), HTTPStatus.INTERNAL_SERVER_ERROR.value



@app.route('/df_gh_orchestrator', methods=['POST'])
def df_gh_orchestrator_route():
    if not INITIALIZATION_SUCCESSFUL:
        app.logger.error("Orchestrator: Service not available due to initialization failure.")
        return jsonify({"error": "Orchestration service not available (initialization failed).", "success": False}), HTTPStatus.SERVICE_UNAVAILABLE.value
    
    ml_output_images_bytes_list = []
    aligned_images_bytes_list = []

    form_data = request.form

    ### Metadata extraction and validation
    try:
        # Number of windows / pictures
        num_images_gh_str = form_data.get('num_images')
        if not num_images_gh_str:
            return jsonify({"error": "Missing 'num_images' in form data.", "success": False}), HTTPStatus.BAD_REQUEST.value
        
        num_images_gh = int(num_images_gh_str)
        if not (isinstance(num_images_gh, int) and num_images_gh > 0):
            return jsonify({"error": "'num_images' must be a positive integer.", "success": False}), HTTPStatus.BAD_REQUEST.value

        app.logger.info(f"Orchestrator received request for {num_images_gh} image(s).")

        if not request.files:
             return jsonify({"error": "No image files found in request.", "success": False}), HTTPStatus.BAD_REQUEST.value
        
        temp_files_dict = {k: v for k, v in request.files.items()}
        if len(temp_files_dict) != num_images_gh:
             return jsonify({"error": f"Number of received image files ({len(temp_files_dict)}) does not match num_images ({num_images_gh}).", "success": False}), HTTPStatus.BAD_REQUEST.value
        
        try: # Sort by Suffix (z.B. floorplan0.png, floorplan1.png)
            sorted_filenames = sorted(temp_files_dict.keys(), key=lambda x: int(x.replace('floorplan','').replace('.png','')))
        except ValueError: 
            return jsonify({"error": "Image filenames not in expected format 'floorplanX.png' (e.g., 'floorplan0.png').", "success": False}), HTTPStatus.BAD_REQUEST.value
        
        input_images_bytes_list = [temp_files_dict[fname].read() for fname in sorted_filenames]
        app.logger.info(f"Orchestrator successfully parsed {num_images_gh} images and their metadata.")

        # Rotation for Alignment
        rotations_rad_gh_str = form_data.get('rotations_rad')

        # Translation for Alignment
        translations_mm_gh_str = form_data.get('translations_mm')

        if not rotations_rad_gh_str or not translations_mm_gh_str:
            missing = []
            if not rotations_rad_gh_str: missing.append("'rotations_rad'")
            if not translations_mm_gh_str: missing.append("'translations_mm'")
            return jsonify({"error": f"Missing metadata: {', '.join(missing)} in form data.", "success": False}), HTTPStatus.BAD_REQUEST.value

        try:
            # Parse JSON to py-lists
            rotations_rad_gh = json.loads(rotations_rad_gh_str)
            translations_mm_gh = json.loads(translations_mm_gh_str)

            # Validate whether they are lists and the length matches num_images
            if not (isinstance(rotations_rad_gh, list) and len(rotations_rad_gh) == num_images_gh and \
                    isinstance(translations_mm_gh, list) and len(translations_mm_gh) == num_images_gh):
                raise TypeError("Rotations/Translations must be lists matching num_images count.")
            
            # # Optional Translation as Liststructure ([[x,y,z],...])
            # if not all(isinstance(t, list) and len(t) >= 2 for t in translations_mm_gh):
            #     raise TypeError("Each translation must be a list of coordinates (e.g., [x, y, z]).")

        except (json.JSONDecodeError, TypeError) as e:
             app.logger.error(f"Orchestrator: Error parsing rotations/translations from form data: {e}", exc_info=True)
             return jsonify({"error": f"Invalid format for rotations_rad or translations_mm: Must be valid JSON lists. {str(e)}", "success": False}), HTTPStatus.BAD_REQUEST.value

    except (TypeError, ValueError) as e: 
        app.logger.error(f"Orchestrator: Error parsing input data: {e}", exc_info=True)
        return jsonify({"error": f"Invalid input data format: {str(e)}", "success": False}), HTTPStatus.BAD_REQUEST.value
    except Exception as e: # Generic Error
        app.logger.error(f"Orchestrator: Error during initial data processing: {e}", exc_info=True)
        return jsonify({"error": f"Orchestrator setup error: {str(e)}", "success": False}), HTTPStatus.INTERNAL_SERVER_ERROR.value
    
    ### ML-Inference
    try:
        # Inference for every Pic via internal Endpoint
        for i in range(num_images_gh):
            current_original_filename = sorted_filenames[i]
            image_bytes_to_send = input_images_bytes_list[i]
            
            app.logger.info(f"Orchestrator: Calling /df_ml_inference for image {i} ('{current_original_filename}')")
            inference_response = internal_post_image_to_endpoint(f"{SERVER_BASE_URL}/df_ml_inference", image_bytes_to_send, image_filename=current_original_filename)
            
            ml_output_images_bytes_list.append(inference_response.content)
            app.logger.info(f"Orchestrator: ML inference for image {i} completed and result stored.")

        ### Align pictures
        app.logger.info(f"Orchestrator: Starting alignment step for {len(ml_output_images_bytes_list)} ML output images.")
        for i in range(num_images_gh):
            current_ml_output_bytes = ml_output_images_bytes_list[i]
            
            original_filename_stem = sorted_filenames[i].replace('.png', '') # z.B. "floorplan0"
            
            app.logger.info(f"Orchestrator: Calling /df_align for ML output image {i} (derived from '{sorted_filenames[i]}')")
            
            # Payload for /df_align 
            align_form_payload = {"rotation_rad": str(rotations_rad_gh[i]), "translation_mm_xyz": json.dumps(translations_mm_gh[i])}
            align_response = internal_post_image_to_endpoint(f"{SERVER_BASE_URL}/df_align", current_ml_output_bytes, image_filename=f"ml_output_for_align_{original_filename_stem}.png", additional_form_data=align_form_payload)
            
            aligned_images_bytes_list.append(align_response.content)
        app.logger.info(f"Orchestrator: Image alignment completed for all {len(aligned_images_bytes_list)} images.")


        ### Convert Pixels to Values
        files_for_value_conversion_step = [
            ('aligned_images[]', (f'aligned_gan_image_{i}.png', img_bytes, 'image/png')) 
            for i, img_bytes in enumerate(aligned_images_bytes_list)
        ]
        app.logger.info(f"Orchestrator: Calling /gh_images_to_values with {len(files_for_value_conversion_step)} aligned images.")
        
        # Direct Call with request (instead of internal_post...)
        values_response = requests.post(
            f"{SERVER_BASE_URL}/gh_images_to_values", 
            files=files_for_value_conversion_step, 
            timeout=60 # Adapt?
        )
        values_response.raise_for_status() 
        values_data_json = values_response.json() 
        app.logger.info(f"Orchestrator: /gh_images_to_values responded (processed {values_data_json.get('num_source_images_processed')} images).")

        ### Calculate Metrics and render final picture ---
        payload_for_metrics_render_step = {
            "summed_value_matrix": values_data_json["summed_value_matrix"],
            "combined_non_white_mask": values_data_json["combined_non_white_mask"]
        }
        app.logger.info(f"Orchestrator: Calling /gh_metrics_and_render with aggregated value data.")
        final_processing_response_object = internal_post_json_to_endpoint(f"{SERVER_BASE_URL}/gh_metrics_and_render", payload_for_metrics_render_step)
        app.logger.info(f"Orchestrator: /gh_metrics_and_render responded with final image and metrics.")

        # Send Pic + Header to GH 
        grasshopper_final_response = make_response(final_processing_response_object.content)
        # Transfer Headers
        for header_key in ['Content-Type', 'X-Processing-Success', 'X-Average-Value', 'X-Ratio-Pixels-GT1', 'Content-Disposition']:
            if header_key in final_processing_response_object.headers:
                grasshopper_final_response.headers[header_key] = final_processing_response_object.headers[header_key]
        
        # Fallback for Content-Disposition (?)
        if 'Content-Disposition' not in grasshopper_final_response.headers:
            grasshopper_final_response.headers['Content-Disposition'] = 'attachment; filename=final_orchestrated_output.png'

        app.logger.info("Orchestrator: Successfully processed request through all stages. Sending final response to Grasshopper.")
        return grasshopper_final_response

    except requests.exceptions.HTTPError as e:
        # error handling for HTTP errors from internal services
        response_details = {}
        failed_url = e.request.url if e.request else "Unknown URL during internal request"
        try:
            # get JSON details from the error response of the internal service
            response_details = e.response.json()
            # take the ‘error’ message or the first 200 characters of the text if no JSON or ‘error’ key
            error_content_for_log = response_details.get("error", e.response.text[:200])
        except ValueError:
            error_content_for_log = e.response.text[:200]
            response_details = {"raw_error_response": error_content_for_log}

        app.logger.error(
            f"Orchestrator: HTTPError during internal call to {failed_url} - Status {e.response.status_code} - Details: {error_content_for_log}",
            exc_info=True # Adds traceback to the log
        )
        return jsonify({
            "error": "Internal service error during pipeline execution.",
            "details": response_details, # Pass on the (possibly shortened) response from the internal service
            "failed_service_url": failed_url,
            "internal_status_code": e.response.status_code,
            "success": False
        }), HTTPStatus.INTERNAL_SERVER_ERROR.value # Or e.response.status_code

    except Exception as e: 
        # Generic Fallback
        app.logger.error(f"Orchestrator: Unhandled exception during main processing: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Orchestrator internal error.",
            "details": str(e),
            "success": False
        }), HTTPStatus.INTERNAL_SERVER_ERROR.value

########################################################################################################



if __name__ == '__main__':
    initialize_server_resources()
    if INITIALIZATION_SUCCESSFUL:
        app.logger.info(f"Flask app '{app.name}' starting on host 0.0.0.0, port {SERVER_PORT}. Debug mode: {app.debug}")
        # `threaded=True` ist wichtig für den Flask-Development-Server, wenn interne HTTP-Aufrufe gemacht werden.
        app.run(debug=True, host="0.0.0.0", port=8081)
    else:
        app.logger.critical(f"Flask app '{app.name}' could NOT start due to CRITICAL initialization errors.")
