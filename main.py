# /usr/bin/env python3
from __future__ import annotations

from flask import Flask, request
from flask_cors import CORS
from http import HTTPStatus
import socket
import tensorflow as tf

from src.utils import get_request_input, build_response

# Libraries for Grasshopper Orchestrator
# import requests
# import numpy as np
from flask import jsonify  #, send_file, make_response
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from PIL import Image
import os
# import io
# import json
import logging
# from google.cloud import storage
# import base64
# # Outsourced functions for Grasshopper Orchestrator
# from src.df_inference import preprocess_for_df_ml, postprocess_df_ml_prediction
# from src.df_align_image import align_df_image
# from src.df_image_processing import generate_value_matrix_for_single_gan_image, aggregate_multiple_value_matrices_gan, calculate_gan_metrics_from_values, render_final_gan_image_from_values, create_color_maps_from_data 


from processing import pipeline_input as inpt
from processing.pipeline import GetDfPipeline

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
    # TODO: load the model
    model = None
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

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ML_MODEL_FILENAME = 'generator.keras' # pix2pix modell
# JSON_COLORSCALE_FILENAME = 'DF-colorscale.json' # json file for matching df-values with colors

# ML_MODEL_GCS_URI = os.getenv('ML_MODEL_GCS_URI', 'gs://df_experiments/fa3a571f-b1da-47db-8b0e-db7caaec9e8e/best_generator.keras')

# JSON_COLORSCALE_GCS_BUCKET_NAME = os.getenv('JSON_COLORSCALE_GCS_BUCKET_NAME', 'daylight_analysis_assets')
# JSON_COLORSCALE_GCS_BLOB_NAME = os.getenv('JSON_COLORSCALE_GCS_BLOB_NAME', 'colorscale_df.json')

# ml_model = None
# gan_value_map_reverse = {}
# gan_known_colors_rgb = np.array([])
# gan_known_colors_lab = np.array([])
# gan_known_values = np.array([])

# INITIALIZATION_SUCCESSFUL = False
SERVER_PORT = int(os.getenv('PORT', 8081))
SERVER_BASE_URL = f"http://localhost:{SERVER_PORT}"

# EXPECTED_ML_IMG_SIZE = (256, 256)
# ML_NORMALIZATION_RANGE = (-1, 1) # Training uses in image_manager.py "(image / 127.5) - 1" to normalize to [-1, 1]
# SCENE_WIDTH_MM = 12800.0 # Dimension of the image area in Rhino (mm)
# SCENE_HEIGHT_MM = 12800.0 
# GAN_PIXELS_PER_MM_X_ML = EXPECTED_ML_IMG_SIZE[0] / SCENE_WIDTH_MM
# GAN_PIXELS_PER_MM_Y_ML = EXPECTED_ML_IMG_SIZE[1] / SCENE_HEIGHT_MM
# FINAL_OUTPUT_IMG_SIZE = (128, 128)


# def initialize_server_resources():
#     global ml_model, INITIALIZATION_SUCCESSFUL
#     global gan_value_map_reverse, gan_known_colors_rgb, \
#            gan_known_colors_lab, gan_known_values
    
#     # Temporary status, gonna be assigned at the end INITIALIZATION_SUCCESSFUL
#     current_function_success_status = True 
#     app.logger.info("Starting server resource initialization...")

#     # 1. Load ML Modell
#     # if not ML_MODEL_GCS_URI:
#     #     app.logger.error("ML_MODEL_GCS_URI environment variable not set. Cannot load model.")
#     #     current_function_success_status = False
#     else:
#         try:
#             # app.logger.info(f"Attempting to load ML model directly from GCS URI: {ML_MODEL_GCS_URI}")
#             ml_model = tf.keras.models.load_model(ML_MODEL_GCS_URI, compile=False)
#             # app.logger.info(f"ML model ('{os.path.basename(ML_MODEL_GCS_URI)}') successfully loaded from GCS.")
#         except Exception as e:
#             app.logger.error(f"Error loading ML model from GCS URI '{ML_MODEL_GCS_URI}': {e}", exc_info=True)
#             current_function_success_status = False

#     # 2. Load Color Matching JSON
#     if current_function_success_status: 
#         if not JSON_COLORSCALE_GCS_BUCKET_NAME or not JSON_COLORSCALE_GCS_BLOB_NAME:
#             app.logger.error("JSON_COLORSCALE GCS Bucket or Blob name not set in env variables. Cannot load color map.")
#             current_function_success_status = False
#         else:
#             storage_client = None
#             try:
#                 storage_client = storage.Client() # Authenticate
#                 bucket = storage_client.bucket(JSON_COLORSCALE_GCS_BUCKET_NAME)
#                 blob = bucket.blob(JSON_COLORSCALE_GCS_BLOB_NAME)
                
#                 app.logger.info(f"Downloading color map from GCS: gs://{JSON_COLORSCALE_GCS_BUCKET_NAME}/{JSON_COLORSCALE_GCS_BLOB_NAME}")
#                 json_string_content = blob.download_as_text(encoding="utf-8-sig") # utf-8-sig für BOM
#                 json_data_from_gcs = json.loads(json_string_content)
                
#                 maps_data = create_color_maps_from_data(json_data_from_gcs)
                
#                 gan_value_map_reverse.clear()
#                 gan_value_map_reverse.update(maps_data['value_map_reverse'])
                
#                 # Global Variablen Update
#                 gan_known_colors_rgb = maps_data['known_colors_rgb']
#                 gan_known_colors_lab = maps_data['known_colors_lab']
#                 gan_known_values = maps_data['known_values']
                
#                 app.logger.info(f"Color map ('{JSON_COLORSCALE_GCS_BLOB_NAME}') successfully downloaded from GCS and processed.")

#             except FileNotFoundError: 
#                 app.logger.error(f"Color map blob '{JSON_COLORSCALE_GCS_BLOB_NAME}' not found in GCS bucket '{JSON_COLORSCALE_GCS_BUCKET_NAME}'.")
#                 current_function_success_status = False
#             except json.JSONDecodeError as e:
#                  app.logger.error(f"Error decoding JSON from GCS blob '{JSON_COLORSCALE_GCS_BLOB_NAME}': {e}", exc_info=True)
#                  current_function_success_status = False
#             except Exception as e:
#                 app.logger.error(f"Error loading/processing Color map from GCS (gs://{JSON_COLORSCALE_GCS_BUCKET_NAME}/{JSON_COLORSCALE_GCS_BLOB_NAME}): {e}", exc_info=True)
#                 current_function_success_status = False

#     INITIALIZATION_SUCCESSFUL = current_function_success_status
#     if not INITIALIZATION_SUCCESSFUL:
#         app.logger.critical("!!! Overall server resource initialization FAILED (GCS). Review logs. !!!")
#     else:
#         app.logger.info("Overall server resources initialization process (GCS) complete.")



# # Outsourced functions for Grasshopper Orchestrator
# def process_single_image_pipeline(image_idx, input_image_bytes, original_filename, rotation_rad, translation_mm_xyz, 
#                                   server_base_url_ref, known_colors_lab_ref, known_values_ref):
#     """
#     Processes a single image by ML inference, alignment and generation of the value matrix.
#     Returns (value_matrix_np, non_white_mask_np, image_idx) or throws an exception.
#     """
#     try:
#         # inference_response = internal_post_image_to_endpoint(f"{server_base_url_ref}/df_ml_inference", input_image_bytes, image_filename=original_filename)
#         ml_output_bytes = inference_response.content
#         # app.logger.info(f"Pipeline Task {image_idx}: ML-Inferenz für '{original_filename}' abgeschlossen.")

#         # Step 2: Call image alignment
#         original_filename_stem = original_filename.replace('.png', '')
#         # app.logger.info(f"Pipeline Task {image_idx}: Rufe /df_align für ML-Output von '{original_filename}' auf")
#         align_form_payload = {
#             "rotation_rad": str(rotation_rad),
#             "translation_mm_xyz": json.dumps(translation_mm_xyz) 
#         }
#         # align_response = internal_post_image_to_endpoint(f"{server_base_url_ref}/df_align", ml_output_bytes, image_filename=f"ml_output_for_align_{original_filename_stem}.png", additional_form_data=align_form_payload)
#         # aligned_image_bytes = align_response.content
#         # app.logger.info(f"Pipeline Task {image_idx}: Ausrichtung für '{original_filename}' abgeschlossen.")

#         # Step 3: Create value matrix for single aligned image
#         # app.logger.info(f"Pipeline Task {image_idx}: Erzeuge Wertematrix für ausgerichtetes Bild '{original_filename}'")
#         # aligned_image_pil_rgba = Image.open(io.BytesIO(aligned_image_bytes))
        
#         v_matrix_np, nw_mask_np = generate_value_matrix_for_single_gan_image(aligned_image_pil_rgba, known_colors_lab_ref, known_values_ref)
#         # app.logger.info(f"Pipeline Task {image_idx}: Wertematrix für '{original_filename}' erzeugt.")
#         return v_matrix_np, nw_mask_np, image_idx



# ### Endpoints Grasshopper Orchestrator  ###
# @app.route('/df_ml_inference', methods=['POST'])
# def df_ml_inference_route():
#     # if not INITIALIZATION_SUCCESSFUL or not ml_model:
#     #     app.logger.error("/df_ml_inference: Service not available due to initialization failure or model not loaded.")
#     #     return jsonify({"error": "ML inference service not available.", "success": False}), HTTPStatus.SERVICE_UNAVAILABLE.value

#     image_file = request.files.get('image_file')
#     # if not image_file:
#     #     return jsonify({"error": "No 'image_file' provided in the request.", "success": False}), HTTPStatus.BAD_REQUEST.value

    
#     # app.logger.info(f"[{request.remote_addr}] /df_ml_inference received image: {image_file.filename}")
#     try:
#         # Load image from bytes and prepare as RGB
#         image_pil_rgb = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        
#         # Preprocessing
#         preprocessed_tensor = preprocess_for_df_ml(image_pil_rgb, EXPECTED_ML_IMG_SIZE, ML_NORMALIZATION_RANGE)
        
#         # ML-Inference
#         app.logger.debug("Starting ML prediction...")
#         prediction_tensor = ml_model.predict(preprocessed_tensor)
#         app.logger.debug("ML prediction finished.")
        
#         # Postprocessing
#         output_image_pil_rgba = postprocess_df_ml_prediction(prediction_tensor, ML_NORMALIZATION_RANGE)
        
#         # Return result image as bytes
#         img_byte_arr = io.BytesIO()
#         output_image_pil_rgba.save(img_byte_arr, format='PNG') 
#         img_byte_arr.seek(0)
        
#         app.logger.info(f"[{request.remote_addr}] /df_ml_inference successfully processed image '{image_file.filename}'.")
#         return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name=f"ml_output_{image_file.filename}")

#     except Exception as e:
#         app.logger.error(f"Error during ML inference for image '{image_file.filename}': {e}", exc_info=True)
#         return jsonify({"error": f"ML inference processing error: {str(e)}", "success": False}), HTTPStatus.INTERNAL_SERVER_ERROR.value



# # @app.route('/df_align', methods=['POST'])
# # def df_align_route():
# #     # if not INITIALIZATION_SUCCESSFUL:
# #     #     app.logger.error("/df_align: Service not available due to general initialization failure.")
# #     #     return jsonify({"error": "Alignment service not available (init failed).", "success": False}), HTTPStatus.SERVICE_UNAVAILABLE.value

# #     image_file = request.files.get('image_file') 
# #     rotation_rad_str = request.form.get('rotation_rad')
# #     translation_mm_xyz_str = request.form.get('translation_mm_xyz') # JSON-String "[x,y,z]"

# #     # if not all([image_file, rotation_rad_str, translation_mm_xyz_str]):
# #     #     missing_fields = []
# #     #     if not image_file: missing_fields.append("'image_file'")
# #     #     if not rotation_rad_str: missing_fields.append("'rotation_rad'")
# #     #     if not translation_mm_xyz_str: missing_fields.append("'translation_mm_xyz'")
# #     #     return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}.", "success": False}), HTTPStatus.BAD_REQUEST.value
    
# #     original_filename = image_file.filename if image_file.filename else "unknown_image"
# #     # app.logger.info(f"[{request.remote_addr}] /df_align received image '{original_filename}' for alignment. R={rotation_rad_str}, T={translation_mm_xyz_str}")

# #     try:
# #         # Load RGBA from /df_ml_inference
# #         image_pil_rgba = Image.open(io.BytesIO(image_file.read()))
        
# #         rotation_rad = float(rotation_rad_str)
# #         translation_mm_xyz = json.loads(translation_mm_xyz_str)

# #         # if not (isinstance(translation_mm_xyz, list) and len(translation_mm_xyz) >= 2 and all(isinstance(c, (int, float)) for c in translation_mm_xyz[:2])):
# #         #      raise ValueError("translation_mm_xyz must be a list of at least two numbers, e.g., [x, y, z] or [x,y]")

# #         transformed_pil_rgba = align_df_image(image_pil_rgba, rotation_rad, translation_mm_xyz, EXPECTED_ML_IMG_SIZE, GAN_PIXELS_PER_MM_X_ML, GAN_PIXELS_PER_MM_Y_ML)
        
# #         img_byte_arr = io.BytesIO()
# #         transformed_pil_rgba.save(img_byte_arr, format='PNG')
# #         img_byte_arr.seek(0)
        
# #         # app.logger.info(f"[{request.remote_addr}] /df_align successfully aligned image '{original_filename}'.")
# #         return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name=f"aligned_{original_filename}")

# #     # except ValueError as ve: #
# #     #     app.logger.error(f"Error in /df_align due to invalid parameter format for '{original_filename}': {ve}", exc_info=True)
# #     #     return jsonify({"error": f"Invalid parameter format: {str(ve)}", "success": False}), HTTPStatus.BAD_REQUEST.value
# #     # except Exception as e:
# #     #     app.logger.error(f"Error during alignment for image '{original_filename}': {e}", exc_info=True)
# #     #     return jsonify({"error": f"Image alignment processing error: {str(e)}", "success": False}), HTTPStatus.INTERNAL_SERVER_ERROR.value



# @app.route('/gh_images_to_values', methods=['POST'])
# def gh_images_to_values_route():
#     if not INITIALIZATION_SUCCESSFUL:
#         return jsonify({"error": "Service not available (init failed).", "success": False}), HTTPStatus.SERVICE_UNAVAILABLE.value

#     files = request.files.getlist("aligned_images[]") 
#     if not files:
#         return jsonify({"error": "No image files ('aligned_images[]') provided.", "success": False}), HTTPStatus.BAD_REQUEST.value
    
#     app.logger.info(f"[{request.remote_addr}] /gh_images_to_values received {len(files)} aligned image(s).")
#     try:
#         value_matrices_list_np = []
#         non_white_masks_list_np = []

#         for i, file_storage in enumerate(files):
#             # Pics in RGBA
#             image_pil_rgba = Image.open(io.BytesIO(file_storage.read()))
            
#             v_matrix_np, nw_mask_np = generate_value_matrix_for_single_gan_image(image_pil_rgba, gan_known_colors_lab, gan_known_values)
#             value_matrices_list_np.append(v_matrix_np)
#             non_white_masks_list_np.append(nw_mask_np)
        
#         # Aggregating matrices
#         summed_values_np, combined_mask_np = aggregate_multiple_value_matrices_gan(value_matrices_list_np, non_white_masks_list_np, EXPECTED_ML_IMG_SIZE)
        
#         app.logger.info(f"[{request.remote_addr}] /gh_images_to_values processing complete. Output summed_value_matrix shape: {summed_values_np.shape}")
#         return jsonify({
#             "summed_value_matrix": summed_values_np.tolist(),
#             "combined_non_white_mask": combined_mask_np.tolist(),
#             "num_source_images_processed": len(files)
#         }), HTTPStatus.OK.value

#     except Exception as e:
#         app.logger.error(f"Error in /gh_images_to_values: {e}", exc_info=True)
#         return jsonify({"error": f"Error processing images to values: {str(e)}", "success": False}), HTTPStatus.INTERNAL_SERVER_ERROR.value


# @app.route('/calculate_metrics', methods=['POST'])
# def calculate_metrics_route():
#     """
#     Calculates analysis metrics from a value matrix and a mask.
#     This endpoint is only responsible for quantitative evaluation.
#     """
#     if not INITIALIZATION_SUCCESSFUL:
#         return jsonify({"error": "Service not available (init failed).", "success": False}), HTTPStatus.SERVICE_UNAVAILABLE.value

#     data = request.get_json()
#     if not data or 'summed_value_matrix' not in data or 'combined_non_white_mask' not in data:
#         return jsonify({"error": "Missing 'summed_value_matrix' or 'combined_non_white_mask' in JSON payload.", "success": False}), HTTPStatus.BAD_REQUEST.value

#     app.logger.info(f"[{request.remote_addr}] /calculate_metrics received request.")
#     try:
#         values_np = np.array(data['summed_value_matrix'], dtype=np.float32)
#         mask_np = np.array(data['combined_non_white_mask'], dtype=bool)

#         avg_value, ratio_gt1 = calculate_gan_metrics_from_values(values_np, mask_np)
        
#         response_payload = {
#             "success": True,
#             "metrics": {
#                 "average_value": round(avg_value, 5),
#                 "ratio_gt1": round(ratio_gt1, 5)
#             },
#             "message": "Metrics calculated successfully."
#         }
#         app.logger.info(f"[{request.remote_addr}] /calculate_metrics: AvgValue={avg_value:.3f}, RatioGT1={ratio_gt1:.3f}")
#         return jsonify(response_payload), HTTPStatus.OK.value
#     except Exception as e:
#         app.logger.error(f"Error in /calculate_metrics: {e}", exc_info=True)
#         return jsonify({"error": f"Error during metrics calculation: {str(e)}", "success": False}), HTTPStatus.INTERNAL_SERVER_ERROR.value


# @app.route('/render_image', methods=['POST'])
# def render_image_route():
#     """
#     Renders a visualization image from a value matrix and mask.
#     This endpoint is only responsible for generating the prediction image.
#     """
#     # if not INITIALIZATION_SUCCESSFUL:
#     #     return jsonify({"error": "Service not available (init failed).", "success": False}), HTTPStatus.SERVICE_UNAVAILABLE.value

#     data = request.get_json()
#     try:
#         values_np = np.array(data['summed_value_matrix'], dtype=np.float32)
#         mask_np = np.array(data['combined_non_white_mask'], dtype=bool)

#         final_image_pil_256 = render_final_gan_image_from_values(values_np, mask_np, EXPECTED_ML_IMG_SIZE, gan_value_map_reverse)
        
#         final_image_to_send_pil_128 = final_image_pil_256.resize(FINAL_OUTPUT_IMG_SIZE, Image.Resampling.NEAREST)

#         img_byte_arr = io.BytesIO()
#         final_image_to_send_pil_128.save(img_byte_arr, format='PNG')
#         image_bytes = img_byte_arr.getvalue()
#         image_base64_str = base64.b64encode(image_bytes).decode('utf-8')

#         output_filename = f"final_rendered_image_{FINAL_OUTPUT_IMG_SIZE[0]}x{FINAL_OUTPUT_IMG_SIZE[1]}.png"
        
#         response_payload = {
#             "success": True,
#             "filename": output_filename,
#             "image_base64": image_base64_str,
#             "mimetype": "image/png",
#             "message": "Image rendered successfully."
#         }
#         # app.logger.info(f"[{request.remote_addr}] /render_image: Successfully rendered and encoded image.")
#         return jsonify(response_payload), HTTPStatus.OK.value
#     # except Exception as e:
#     #     app.logger.error(f"Error in /render_image: {e}", exc_info=True)
#     #     return jsonify({"error": f"Error during final rendering: {str(e)}", "success": False}), HTTPStatus.INTERNAL_SERVER_ERROR.value

@app.route('/get_df', methods=['POST'])
def get_df():
    """
    Endpoint that receives a set of images per apartment with transformation parameters and returns a simulation matrix for DF factor.
    """
    image_string = request.form.get('image')
    # rotations per image
    # translations per image
    inp = inpt.GetDfPipelineInput(image_string)
    res = GetDfPipeline.run(inp)
    return jsonify(res.value), HTTPStatus.OK.value




# @app.route('/df_gh_orchestrator', methods=['POST'])
# def df_gh_orchestrator_route():
    
#     form_data = request.form

#     value_matrices_from_pipeline = []
#     non_white_masks_from_pipeline = []

#     ### Metadata extraction and validation
#     try:
        
#         temp_files_dict = {k: v for k, v in request.files.items()}
        
        
#         try: 
#             sorted_filenames = sorted(temp_files_dict.keys(), key=lambda x: int(x.replace('floorplan','').replace('.png','')))
#         # except ValueError: 
#         #     app.logger.warning("Orchestrator: Image filenames not in expected format 'floorplanX.png'.")
#         #     return jsonify({"error": "Image filenames not in expected format 'floorplanX.png' (e.g., 'floorplan0.png').", "success": False}), HTTPStatus.BAD_REQUEST.value
        
#         input_images_bytes_list = [temp_files_dict[fname].read() for fname in sorted_filenames]
#         rotations_rad_gh_str = form_data.get('rotations_rad')
#         translations_mm_gh_str = form_data.get('translations_mm')


#         try:
#             rotations_rad_gh = json.loads(rotations_rad_gh_str)
#             translations_mm_gh = json.loads(translations_mm_gh_str)

#     try:
#         with ThreadPoolExecutor(max_workers=num_workers) as executor:
#             for i in range(num_images_gh):
#                 future = executor.submit(
#                     process_single_image_pipeline,
#                     i,
#                     input_images_bytes_list[i],
#                     sorted_filenames[i],
#                     rotations_rad_gh[i],
#                     translations_mm_gh[i],
#                     SERVER_BASE_URL, 
#                     gan_known_colors_lab, 
#                     gan_known_values
#                 )
#                 futures.append(future)

#             for future in as_completed(futures):
#                 try:
#                     v_matrix_np, nw_mask_np, processed_idx = future.result()
#                     value_matrices_from_pipeline.append(v_matrix_np)
#                     non_white_masks_from_pipeline.append(nw_mask_np)
#                     app.logger.info(f"Orchestrator: Pipeline Task {processed_idx} for '{sorted_filenames[processed_idx]}' completed successfully.")
#                 # except Exception as e_future:
#                 #     # Ein Fehler in einer der Pipeline-Aufgaben ist aufgetreten
#                 #     app.logger.error(f"Orchestrator: A pipeline task failed catastrophically: {e_future}", exc_info=False)
#                 #     return jsonify({
#                 #         "error": "Orchestrator error: A sub-task in the parallel image processing pipeline failed.",
#                 #         "details": str(e_future),
#                 #         "success": False
#                 #     }), HTTPStatus.INTERNAL_SERVER_ERROR.value
        
#         # if len(value_matrices_from_pipeline) != num_images_gh:
#         #     app.logger.error(f"Orchestrator: Mismatch in expected ({num_images_gh}) and processed ({len(value_matrices_from_pipeline)}) images after parallel pipeline.")
#         #     return jsonify({
#         #         "error": "Orchestrator error: Not all images were processed successfully in the pipeline.",
#         #         "success": False
#         #     }), HTTPStatus.INTERNAL_SERVER_ERROR.value

#         # app.logger.info(f"Orchestrator: Parallel pipeline processing completed for all {num_images_gh} images.")

#         ### Aggregate Value Matrices (directly in the orchestrator)
#         # app.logger.info(f"Orchestrator: Aggregating {len(value_matrices_from_pipeline)} value matrices.")
#         summed_values_np, combined_mask_np = aggregate_multiple_value_matrices_gan(
#             value_matrices_from_pipeline,
#             non_white_masks_from_pipeline,
#             EXPECTED_ML_IMG_SIZE
#         )
#         # app.logger.info(f"Orchestrator: Aggregation complete. Summed values shape: {summed_values_np.shape}, Combined mask non-white pixels: {np.sum(combined_mask_np)}")

#         ### Calculate Metrics and Render Final Image in Parallel ###
#         payload_for_final_steps = {
#             "summed_value_matrix": summed_values_np.tolist(),
#             "combined_non_white_mask": combined_mask_np.tolist()
#         }
        
#         metrics_data = {}
#         render_data = {}

#         with ThreadPoolExecutor(max_workers=2) as executor:
#             # app.logger.info("Orchestrator: Calling /calculate_metrics and /render_image in parallel.")
            
#             # Submit metrics calculation
#             future_metrics = executor.submit(
#                 internal_post_json_to_endpoint, 
#                 f"{SERVER_BASE_URL}/calculate_metrics", 
#                 payload_for_final_steps
#             )
#             # Submit image rendering
#             future_render = executor.submit(
#                 internal_post_json_to_endpoint, 
#                 f"{SERVER_BASE_URL}/render_image", 
#                 payload_for_final_steps
#             )
            
#             # Wait for results and process them
#             metrics_response = future_metrics.result()
#             render_response = future_render.result()
            
#             metrics_response.raise_for_status() # Will raise HTTPError if the call failed
#             render_response.raise_for_status()  # Will raise HTTPError if the call failed
            
#             metrics_data = metrics_response.json()
#             render_data = render_response.json()

#         # app.logger.info("Orchestrator: Both metrics and render endpoints responded.")

#         ### Combine results and send final JSON response to the client ###
#         final_success = metrics_data.get("success", False) and render_data.get("success", False)
        
#         response_to_gh = {
#             "success": final_success,
#             "filename": render_data.get("filename"),
#             "image_base64": render_data.get("image_base64"),
#             "mimetype": render_data.get("mimetype", "image/png"),
#             "metrics": metrics_data.get("metrics"),
#             "message": "Orchestration completed." if final_success else "Orchestration failed in final processing step."
#         }
        
#         if response_to_gh.get("success"):
#             app.logger.info(f"Orchestrator: Successfully processed request. Metrics: {response_to_gh.get('metrics')}. Sending JSON response to Grasshopper.")
#         else:
#             app.logger.warning(f"Orchestrator: Processing chain reported failure. Details: {response_to_gh.get('message')}. Sending JSON error to Grasshopper.")
        
#         return jsonify(response_to_gh), HTTPStatus.OK.value # OK, auch wenn success=False im Payload, da der Orchestrator selbst gelaufen ist

#     # except requests.exceptions.HTTPError as e_http_orch:
#     #     response_details = {}
#     #     failed_url = e_http_orch.request.url if e_http_orch.request else "Unknown URL (orchestrator direct call)"
#     #     try:
#     #         response_details = e_http_orch.response.json()
#     #         error_content_for_log = response_details.get("error", e_http_orch.response.text[:200])
#     #     except (json.JSONDecodeError, ValueError):
#     #         error_content_for_log = e_http_orch.response.text[:200]
#     #         response_details = {"raw_error_response": error_content_for_log}
#     #     app.logger.error(f"Orchestrator: HTTPError during direct internal call to {failed_url} - Status {e_http_orch.response.status_code if e_http_orch.response is not None else 'N/A'} - Details: {error_content_for_log}", exc_info=True)
#     #     status_code_to_return = e_http_orch.response.status_code if e_http_orch.response is not None else HTTPStatus.INTERNAL_SERVER_ERROR.value
#     #     return jsonify({
#     #         "error": "Internal service error during pipeline execution (orchestrator direct call).",
#     #         "details": response_details, 
#     #         "failed_service_url": failed_url,
#     #         "internal_status_code": e_http_orch.response.status_code if e_http_orch.response is not None else None,
#     #         "success": False
#     #     }), status_code_to_return

#     # except Exception as e_orch: 
#     #     app.logger.error(f"Orchestrator: Unhandled exception during main processing: {str(e_orch)}", exc_info=True)
#     #     return jsonify({
#     #         "error": "Orchestrator internal error.",
#     #         "details": str(e_orch),
#     #         "success": False
#     #     }), HTTPStatus.INTERNAL_SERVER_ERROR.value

########################################################################################################

# initialize_server_resources()

if __name__ == '__main__':
    
    # if INITIALIZATION_SUCCESSFUL:
    app.logger.info(f"Flask app '{app.name}' starting on host 0.0.0.0, port {SERVER_PORT}. Debug mode: {app.debug}")
    app.run(debug=True, host="0.0.0.0", port=8081)
