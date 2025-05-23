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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

#model = tf.keras.models.load_model("generator_model.keras")

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

ML_MODEL_GCS_URI = os.getenv('ML_MODEL_GCS_URI', 'gs://df_experiments/fa3a571f-b1da-47db-8b0e-db7caaec9e8e/best_generator.keras')

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
FINAL_OUTPUT_IMG_SIZE = (128, 128)


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
                
                maps_data = create_color_maps_from_data(json_data_from_gcs)
                
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


# Outsourced functions for Grasshopper Orchestrator
def process_single_image_pipeline(image_idx, input_image_bytes, original_filename, rotation_rad, translation_mm_xyz, 
                                  server_base_url_ref, known_colors_lab_ref, known_values_ref):
    """
    Processes a single image by ML inference, alignment and generation of the value matrix.
    Returns (value_matrix_np, non_white_mask_np, image_idx) or throws an exception.
    """
    try:
        app.logger.info(f"Pipeline Task {image_idx}: Start für Bild '{original_filename}'")

        # Step 1: Call ML reference
        app.logger.info(f"Pipeline Task {image_idx}: Rufe /df_ml_inference für '{original_filename}' auf")
        inference_response = internal_post_image_to_endpoint(f"{server_base_url_ref}/df_ml_inference", input_image_bytes, image_filename=original_filename)
        ml_output_bytes = inference_response.content
        app.logger.info(f"Pipeline Task {image_idx}: ML-Inferenz für '{original_filename}' abgeschlossen.")

        # Step 2: Call image alignment
        original_filename_stem = original_filename.replace('.png', '')
        app.logger.info(f"Pipeline Task {image_idx}: Rufe /df_align für ML-Output von '{original_filename}' auf")
        align_form_payload = {
            "rotation_rad": str(rotation_rad),
            "translation_mm_xyz": json.dumps(translation_mm_xyz) 
        }
        align_response = internal_post_image_to_endpoint(f"{server_base_url_ref}/df_align", ml_output_bytes, image_filename=f"ml_output_for_align_{original_filename_stem}.png", additional_form_data=align_form_payload)
        aligned_image_bytes = align_response.content
        app.logger.info(f"Pipeline Task {image_idx}: Ausrichtung für '{original_filename}' abgeschlossen.")

        # Step 3: Create value matrix for single aligned image
        app.logger.info(f"Pipeline Task {image_idx}: Erzeuge Wertematrix für ausgerichtetes Bild '{original_filename}'")
        aligned_image_pil_rgba = Image.open(io.BytesIO(aligned_image_bytes))
        
        v_matrix_np, nw_mask_np = generate_value_matrix_for_single_gan_image(aligned_image_pil_rgba, known_colors_lab_ref, known_values_ref)
        app.logger.info(f"Pipeline Task {image_idx}: Wertematrix für '{original_filename}' erzeugt.")
        return v_matrix_np, nw_mask_np, image_idx

    except requests.exceptions.HTTPError as e_http:
        failed_url = e_http.request.url if e_http.request else "Unbekannte URL in Pipeline"
        error_content = "N/A"
        try:
            # #  extract the error details from the JSON response of the internal service
            error_content = e_http.response.json().get("error", e_http.response.text[:200])
        except (json.JSONDecodeError, ValueError): 
            error_content = e_http.response.text[:200]
        app.logger.error(f"Pipeline Task {image_idx} ('{original_filename}'): HTTPError beim internen Aufruf an {failed_url} - Status {e_http.response.status_code} - Details: {error_content}", exc_info=False)
        raise Exception(f"Pipeline fehlgeschlagen für Bild {image_idx} ('{original_filename}') bei {failed_url}. Interner Service Fehler: {error_content} (Status: {e_http.response.status_code})") from e_http
    except Exception as e_pipe:
        app.logger.error(f"Pipeline Task {image_idx} ('{original_filename}'): Allgemeiner Fehler in der Verarbeitung: {e_pipe}", exc_info=True)
        raise Exception(f"Pipeline fehlgeschlagen für Bild {image_idx} ('{original_filename}'): {str(e_pipe)}") from e_pipe


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
        preprocessed_tensor = preprocess_for_df_ml(image_pil_rgb, EXPECTED_ML_IMG_SIZE, ML_NORMALIZATION_RANGE)
        
        # ML-Inference
        app.logger.debug("Starting ML prediction...")
        prediction_tensor = ml_model.predict(preprocessed_tensor)
        app.logger.debug("ML prediction finished.")
        
        # Postprocessing
        output_image_pil_rgba = postprocess_df_ml_prediction(prediction_tensor, ML_NORMALIZATION_RANGE)
        
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

        transformed_pil_rgba = align_df_image(image_pil_rgba, rotation_rad, translation_mm_xyz, EXPECTED_ML_IMG_SIZE, GAN_PIXELS_PER_MM_X_ML, GAN_PIXELS_PER_MM_Y_ML)
        
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
            
            v_matrix_np, nw_mask_np = generate_value_matrix_for_single_gan_image(image_pil_rgba, gan_known_colors_lab, gan_known_values)
            value_matrices_list_np.append(v_matrix_np)
            non_white_masks_list_np.append(nw_mask_np)
        
        # Aggregating matrices
        summed_values_np, combined_mask_np = aggregate_multiple_value_matrices_gan(value_matrices_list_np, non_white_masks_list_np, EXPECTED_ML_IMG_SIZE)
        
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

        # Calculate metrics (based on 256x256)
        avg_value, ratio_gt1 = calculate_gan_metrics_from_values(values_np, mask_np)
        
        # "render" final picture (256x256)
        final_image_pil_256 = render_final_gan_image_from_values(values_np, mask_np, EXPECTED_ML_IMG_SIZE, gan_value_map_reverse)
        
        app.logger.info(f"[{request.remote_addr}] /gh_metrics_and_render: AvgValue={avg_value:.3f}, RatioGT1={ratio_gt1:.3f}")

        # Scale to 128x128 for Output
        app.logger.info(f"Resizing final image from {EXPECTED_ML_IMG_SIZE} to {FINAL_OUTPUT_IMG_SIZE} using NEAREST resampling...")
        final_image_to_send_pil_128 = final_image_pil_256.resize(FINAL_OUTPUT_IMG_SIZE, Image.Resampling.NEAREST)
        app.logger.info(f"Image resized to {FINAL_OUTPUT_IMG_SIZE}.")

        img_byte_arr = io.BytesIO()
        final_image_to_send_pil_128.save(img_byte_arr, format='PNG') # Verwende das skalierte Bild
        img_byte_arr.seek(0)

        response = make_response(send_file(
            img_byte_arr, 
            mimetype='image/png', 
            as_attachment=True, 
            download_name=f"final_rendered_image_{FINAL_OUTPUT_IMG_SIZE[0]}x{FINAL_OUTPUT_IMG_SIZE[1]}.png"))
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
    
    # Lists to store results from parallel processing
    value_matrices_from_pipeline = []
    non_white_masks_from_pipeline = []

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
        
        try: # Sort by Suffix (e.g., floorplan0.png, floorplan1.png)
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
            # Optional deeper validation for translation structure could be added here if needed
            # e.g., if not all(isinstance(t, list) and len(t) >= 2 for t in translations_mm_gh):
            # raise TypeError("Each translation must be a list of coordinates (e.g., [x, y, z]).")

        except (json.JSONDecodeError, TypeError) as e:
            app.logger.error(f"Orchestrator: Error parsing rotations/translations from form data: {e}", exc_info=True)
            return jsonify({"error": f"Invalid format for rotations_rad or translations_mm: Must be valid JSON lists. {str(e)}", "success": False}), HTTPStatus.BAD_REQUEST.value

    except (TypeError, ValueError) as e: 
        app.logger.error(f"Orchestrator: Error parsing input data: {e}", exc_info=True)
        return jsonify({"error": f"Invalid input data format: {str(e)}", "success": False}), HTTPStatus.BAD_REQUEST.value
    except Exception as e:
        app.logger.error(f"Orchestrator: Error during initial data processing: {e}", exc_info=True)
        return jsonify({"error": f"Orchestrator setup error: {str(e)}", "success": False}), HTTPStatus.INTERNAL_SERVER_ERROR.value
    
    ### Parallel Processing: ML-Inference -> Alignment -> Value Matrix Generation for each image
    app.logger.info(f"Orchestrator: Starting parallel pipeline for {num_images_gh} images.")
    
    futures = []
    # Determine number of workers for the ThreadPoolExecutor
    num_workers = min(num_images_gh, (os.cpu_count() or 1) * 5) 
    app.logger.info(f"Orchestrator: Using up to {num_workers} workers for parallel image processing.")

    # This outer try-except block is for orchestrator-level issues during the parallel submission or final aggregation/rendering.
    # Errors within individual pipeline tasks are handled inside the future.result() loop.
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i in range(num_images_gh):
                # Submit each image processing task to the executor
                future = executor.submit(
                    process_single_image_pipeline, # The new helper function defined above
                    i,                             # Index of the image for tracking/logging
                    input_images_bytes_list[i],
                    sorted_filenames[i],
                    rotations_rad_gh[i],
                    translations_mm_gh[i],
                    SERVER_BASE_URL,        # Pass SERVER_BASE_URL (global)
                    gan_known_colors_lab,   # Pass gan_known_colors_lab (global)
                    gan_known_values        # Pass gan_known_values (global)
                )
                futures.append(future)

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    v_matrix_np, nw_mask_np, processed_idx = future.result() # This will re-raise exceptions from the task
                    value_matrices_from_pipeline.append(v_matrix_np)
                    non_white_masks_from_pipeline.append(nw_mask_np)
                    app.logger.info(f"Orchestrator: Pipeline Task {processed_idx} completed successfully.")
                except Exception as e_future:
                    # An error occurred in one of the pipeline tasks
                    app.logger.error(f"Orchestrator: A pipeline task failed catastrophically: {e_future}", exc_info=False) # exc_info=False as e_future contains details
                    # Fail the entire orchestration if any sub-task fails
                    return jsonify({
                        "error": "Orchestrator error: A sub-task in the parallel image processing pipeline failed.",
                        "details": str(e_future), # Contains details from the failing pipeline task
                        "success": False
                    }), HTTPStatus.INTERNAL_SERVER_ERROR.value
        
        # Check if all images were processed (e.g., no unexpected errors that didn't get caught above)
        if len(value_matrices_from_pipeline) != num_images_gh:
            app.logger.error(f"Orchestrator: Mismatch in expected ({num_images_gh}) and processed ({len(value_matrices_from_pipeline)}) images after parallel pipeline.")
            return jsonify({
                "error": "Orchestrator error: Not all images were processed successfully in the pipeline.",
                "success": False
            }), HTTPStatus.INTERNAL_SERVER_ERROR.value

        app.logger.info(f"Orchestrator: Parallel pipeline processing completed for all {num_images_gh} images.")

        ### Aggregate Value Matrices (directly in the orchestrator)
        app.logger.info(f"Orchestrator: Aggregating {len(value_matrices_from_pipeline)} value matrices.")
        summed_values_np, combined_mask_np = aggregate_multiple_value_matrices_gan(
            value_matrices_from_pipeline,
            non_white_masks_from_pipeline,
            EXPECTED_ML_IMG_SIZE # Global constant
        )
        app.logger.info(f"Orchestrator: Aggregation complete. Summed values shape: {summed_values_np.shape}, Combined mask non-white pixels: {np.sum(combined_mask_np)}")

        ### Calculate Metrics and render final picture ---
        # Construct payload for the /gh_metrics_and_render endpoint
        payload_for_metrics_render_step = {
            "summed_value_matrix": summed_values_np.tolist(),     # Convert NumPy array to list for JSON
            "combined_non_white_mask": combined_mask_np.tolist()  # Convert NumPy array to list for JSON
        }
        app.logger.info(f"Orchestrator: Calling /gh_metrics_and_render with aggregated value data.")
        final_processing_response_object = internal_post_json_to_endpoint(
            f"{SERVER_BASE_URL}/gh_metrics_and_render", # SERVER_BASE_URL is global
            payload_for_metrics_render_step
        )
        app.logger.info(f"Orchestrator: /gh_metrics_and_render responded with final image and metrics.")

        ### Send final picture and headers to Grasshopper
        grasshopper_final_response = make_response(final_processing_response_object.content)
        # Transfer relevant headers from the internal response to the final response
        for header_key in ['Content-Type', 'X-Processing-Success', 'X-Average-Value', 'X-Ratio-Pixels-GT1', 'Content-Disposition']:
            if header_key in final_processing_response_object.headers:
                grasshopper_final_response.headers[header_key] = final_processing_response_object.headers[header_key]
        
        # Fallback for Content-Disposition if not set by the metrics_and_render endpoint
        if 'Content-Disposition' not in grasshopper_final_response.headers:
            # FINAL_OUTPUT_IMG_SIZE is a global tuple e.g., (128,128)
            grasshopper_final_response.headers['Content-Disposition'] = f'attachment; filename=final_orchestrated_output_{FINAL_OUTPUT_IMG_SIZE[0]}x{FINAL_OUTPUT_IMG_SIZE[1]}.png'

        app.logger.info("Orchestrator: Successfully processed request through all stages. Sending final response to Grasshopper.")
        return grasshopper_final_response

    # Exception handling for HTTPError from internal calls made *directly by the orchestrator*
    # (e.g., the call to /gh_metrics_and_render).
    # Errors from pipeline tasks submitted to ThreadPoolExecutor are caught in the as_completed loop.
    except requests.exceptions.HTTPError as e_http_orch:
        response_details = {}
        failed_url = e_http_orch.request.url if e_http_orch.request else "Unknown URL during orchestrator's internal call"
        try:
            response_details = e_http_orch.response.json()
            error_content_for_log = response_details.get("error", e_http_orch.response.text[:200])
        except (json.JSONDecodeError, ValueError):
            error_content_for_log = e_http_orch.response.text[:200]
            response_details = {"raw_error_response": error_content_for_log}

        app.logger.error(
            f"Orchestrator: HTTPError during direct internal call to {failed_url} - Status {e_http_orch.response.status_code} - Details: {error_content_for_log}",
            exc_info=True 
        )
        status_code_to_return = e_http_orch.response.status_code if e_http_orch.response is not None else HTTPStatus.INTERNAL_SERVER_ERROR.value
        return jsonify({
            "error": "Internal service error during pipeline execution (orchestrator direct call).",
            "details": response_details, 
            "failed_service_url": failed_url,
            "internal_status_code": e_http_orch.response.status_code if e_http_orch.response is not None else None,
            "success": False
        }), status_code_to_return

    # Generic Fallback for orchestrator-level errors (e.g., issues with ThreadPoolExecutor setup, unexpected errors)
    except Exception as e_orch: 
        app.logger.error(f"Orchestrator: Unhandled exception during main processing: {str(e_orch)}", exc_info=True)
        return jsonify({
            "error": "Orchestrator internal error.",
            "details": str(e_orch),
            "success": False
        }), HTTPStatus.INTERNAL_SERVER_ERROR.value

########################################################################################################

initialize_server_resources()

if __name__ == '__main__':
    
    if INITIALIZATION_SUCCESSFUL:
        app.logger.info(f"Flask app '{app.name}' starting on host 0.0.0.0, port {SERVER_PORT}. Debug mode: {app.debug}")
        app.run(debug=True, host="0.0.0.0", port=8081)
    else:
        app.logger.critical(f"Flask app '{app.name}' could NOT start due to CRITICAL initialization errors.")
