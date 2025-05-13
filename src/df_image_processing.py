# src/image_processing_gan.py
import os
import json
import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def create_color_maps_from_data(json_data_list: list, logger): # 
    """
    Creates the various color mapping structures from an already
    parsed list of dictionaries.
    """
    logger.info(f"Processing color map data containing {len(json_data_list)} entries.")
    
    value_map_reverse = {}
    known_colors_rgb_list = []
    known_values_list = []
    valid_entries = 0

    for i, item in enumerate(json_data_list): # Iterate over datalist
        if not isinstance(item, dict) or not all(k in item for k in ["Color", "Value"]):
            logger.warning(f"Entry #{i+1} in color map data is not a dict or is missing 'Color' or 'Value', skipping: {item}")
            continue
        
        color_list, value_from_json = item["Color"], item["Value"]

        if not (isinstance(color_list, list) and len(color_list) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in color_list)):
            logger.warning(f"Entry #{i+1}: Invalid color format {color_list}. Expected [R,G,B] with int 0-255. Skipping.")
            continue
        if not isinstance(value_from_json, (int, float)):
            logger.warning(f"Entry #{i+1}: Invalid value '{value_from_json}' for color {color_list}. Not a number. Skipping.")
            continue
        
        float_value = float(value_from_json)
        if float_value in value_map_reverse:
            logger.warning(f"Entry #{i+1}: Duplicate value {float_value}. Color {color_list} will overwrite previous mapping.")
        
        value_map_reverse[float_value] = np.array(color_list, dtype=np.uint8)
        known_colors_rgb_list.append(color_list)
        known_values_list.append(float_value)
        valid_entries += 1

    if not valid_entries:
        logger.error("No valid color-value pairs found in the provided data.")
        raise ValueError("No valid entries in color map data")

    known_colors_rgb_np = np.array(known_colors_rgb_list, dtype=np.uint8)
    known_colors_lab_np = np.array([])
    if known_colors_rgb_np.size > 0:
        known_colors_lab_np = rgb2lab(known_colors_rgb_np.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    
    known_values_np = np.array(known_values_list, dtype=np.float64)
    
    logger.info(f"Color maps successfully created from data: {valid_entries} valid entries processed.")
    return {
        'value_map_reverse': value_map_reverse,
        'known_colors_rgb': known_colors_rgb_np,
        'known_colors_lab': known_colors_lab_np,
        'known_values': known_values_np
    }


def load_color_map_from_json(json_path, logger):
    """
    Lädt die Farb-Wert-Zuordnung aus der JSON-Datei.
    Gibt ein Dictionary mit den geladenen Karten zurück.
    """
    logger.info(f"Attempting to load GAN color map from: {json_path}")
    if not os.path.exists(json_path):
        logger.error(f"JSON color map not found at: {json_path}")
        raise FileNotFoundError(f"JSON color map not found at: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8-sig') as f: # utf-8-sig BOM-Handling
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"ERROR: JSON file '{json_path}' is not correctly formatted: {e}", exc_info=True)
        raise ValueError(f"Invalid JSON format in {json_path}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading JSON color map '{json_path}': {e}", exc_info=True)
        raise RuntimeError(f"Could not load color map {json_path}") from e

    value_map_reverse = {}
    known_colors_rgb_list = []
    known_values_list = []
    valid_entries = 0

    for i, item in enumerate(data):
        if not all(k in item for k in ["Color", "Value"]):
            logger.warning(f"Entry #{i+1} in JSON '{os.path.basename(json_path)}' is missing 'Color' or 'Value', skipping: {item}")
            continue
        color_list, value_from_json = item["Color"], item["Value"]

        if not (isinstance(color_list, list) and len(color_list) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in color_list)):
            logger.warning(f"Entry #{i+1} in '{os.path.basename(json_path)}': Invalid color format {color_list}. Expected [R,G,B] with int 0-255. Skipping.")
            continue
        if not isinstance(value_from_json, (int, float)):
            logger.warning(f"Entry #{i+1} in '{os.path.basename(json_path)}': Invalid value '{value_from_json}' for color {color_list}. Not a number. Skipping.")
            continue
        
        float_value = float(value_from_json)
        if float_value in value_map_reverse: # Prüfe auf doppelte Werte
            logger.warning(f"Entry #{i+1} in '{os.path.basename(json_path)}': Duplicate value {float_value}. Color {color_list} will overwrite previous mapping for this value.")
        
        value_map_reverse[float_value] = np.array(color_list, dtype=np.uint8)
        known_colors_rgb_list.append(color_list)
        known_values_list.append(float_value)
        valid_entries += 1

    if not valid_entries:
        logger.error(f"No valid color-value pairs found in '{os.path.basename(json_path)}'.")
        raise ValueError(f"No valid entries in color map {json_path}")

    known_colors_rgb_np = np.array(known_colors_rgb_list, dtype=np.uint8)
    known_colors_lab_np = np.array([])
    if known_colors_rgb_np.size > 0:
        # Convert RGB to LAB (skimage.color.rgb2lab expects values in the range [0,1])
        # Reshape to (N, 1, 3) so that rgb2lab processes a list of colors correctly
        known_colors_lab_np = rgb2lab(known_colors_rgb_np.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    
    known_values_np = np.array(known_values_list, dtype=np.float64)
    
    logger.info(f"GAN color map '{os.path.basename(json_path)}' successfully loaded: {valid_entries} valid entries.")
    return {
        'value_map_reverse': value_map_reverse,
        'known_colors_rgb': known_colors_rgb_np,
        'known_colors_lab': known_colors_lab_np, # For color similarity
        'known_values': known_values_np          # Associated numerical values
    }


def generate_value_matrix_for_single_gan_image(image_pil_rgba: Image.Image, logger, gan_known_colors_lab_np: np.ndarray, gan_known_values_np: np.ndarray):
    """
    Wrapper function, which now calls the fast, vectorized method.
    """
    # Replacing older methode with faster, vectorized method  
    logger.info(f"Generating value matrix for single GAN image for image size {image_pil_rgba.size}")

    value_matrix, non_white_mask = fast_value_matrix(image_pil_rgba, gan_known_colors_lab_np, gan_known_values_np, logger)
    return value_matrix, non_white_mask


def aggregate_multiple_value_matrices_gan(value_matrices_list_np, non_white_masks_list_np, logger, image_size_wh_for_fallback):
    """
    Sums a list of value matrices and combines their masks by logical OR.
    """
    if not value_matrices_list_np:
        logger.warning("No value matrices provided for GAN aggregation.")
        # Return of empty or appropriately sized zero arrays, based on GAN_EXPECTED_ML_IMG_SIZE
        h, w = image_size_wh_for_fallback[1], image_size_wh_for_fallback[0] # Pillow (W,H) -> Numpy (H,W)
        return np.zeros((h,w), dtype=np.float64), np.zeros((h,w), dtype=bool)

    logger.debug(f"Aggregating {len(value_matrices_list_np)} GAN value matrices.")
    
    # Make sure that all matrices have the same shape before stacking
    
    try:
        stacked_values = np.stack(value_matrices_list_np, axis=0)
        summed_values_np = np.sum(stacked_values, axis=0)
        
        stacked_masks = np.stack(non_white_masks_list_np, axis=0)
        combined_mask_np = np.any(stacked_masks, axis=0)
    except ValueError as e:
        logger.error(f"Error stacking matrices/masks for GAN aggregation (likely shape mismatch): {e}", exc_info=True)
        # Fallback
        h, w = image_size_wh_for_fallback[1], image_size_wh_for_fallback[0]
        return np.zeros((h,w), dtype=np.float64), np.zeros((h,w), dtype=bool)
        # raise ValueError("Shape mismatch during aggregation") from e

    logger.debug(f"GAN Aggregation complete. Summed values shape: {summed_values_np.shape}, Combined mask non-white pixels: {np.sum(combined_mask_np)}")
    return summed_values_np, combined_mask_np


def calculate_gan_metrics_from_values(values_np, non_white_mask_np, logger):
    """
    Calculates metrics (average value, ratio >1) from the value matrix and mask.
    """
    num_non_white_pixels = np.sum(non_white_mask_np)
    average_value = 0.0
    ratio_gt1 = 0.0

    if num_non_white_pixels > 0:
        relevant_values = values_np[non_white_mask_np] # Extract only values from non-white pixels
        average_value = np.sum(relevant_values) / num_non_white_pixels
        
        num_pixels_value_gt_1 = np.sum(relevant_values > 1.0)
        ratio_gt1 = num_pixels_value_gt_1 / num_non_white_pixels
        logger.debug(f"GAN Metrics calculated: AvgValue={average_value:.4f}, RatioGT1={ratio_gt1:.4f}, NonWhitePixels={num_non_white_pixels}")
    else:
        logger.info("No non-white pixels found for GAN metric calculation, metrics will be 0.0.")
        
    return average_value, ratio_gt1


def render_final_gan_image_from_values(
        values_np: np.ndarray,          # (summed) values,
        non_white_mask_np: np.ndarray,  # mask of non-white pixels
        logger,
        image_size_wh: tuple,
        value_map_reverse: dict
    ) -> Image.Image:
    """
    Creates the final output image (PIL image, RGB) in a vectorized way
    based on the value matrix.
    """
    height, width = image_size_wh[1], image_size_wh[0] # Numpy-Reihenfolge (H,W)
    logger.info(f"Rendering final GAN image from values (vectorized). Target image size: ({width}x{height})")

    if not value_map_reverse:
        logger.warning("value_map_reverse is empty! Returning a default white image.")
        return Image.new("RGB", image_size_wh, (255, 255, 255))

    # 1. Value Matrix
    values_flat = values_np.ravel() # Shape: (H*W,)

    # Create lookup arrays from value_map_reverse
    available_map_keys_sorted = sorted(list(value_map_reverse.keys()))
    if not available_map_keys_sorted:
         logger.warning("value_map_reverse has no keys after sorting! Returning a default white image.")
         return Image.new("RGB", image_size_wh, (255, 255, 255))

    available_vals_np = np.asarray(available_map_keys_sorted, dtype=np.float64) # Shape: (K,)

    # Create the color lookup table in the same order as available_vals_np
    color_lut_list = [value_map_reverse[v] for v in available_vals_np]
    color_lut_np = np.stack(color_lut_list).astype(np.uint8) # Shape: (K,3)

    # For  value in values_flat, find the index of the nearest value in available_vals_np
    # Broadcasting results in (H*W, K) for the difference matrix
    abs_diff_matrix = np.abs(values_flat[:, None] - available_vals_np)
    idx_flat = abs_diff_matrix.argmin(axis=1) # Indizes der nächsten Nachbarn, Shape: (H*W,)

    # 2. Color-Lookup
    rgb_flat = color_lut_np[idx_flat] # Shape: (H*W,3)

    # 3. set areas that were originally white (or masked as not relevant) to white
    # non_white_mask_np has Shape (H,W)
    rgb_flat[~non_white_mask_np.ravel()] = [255, 255, 255] # Background

    # 4. Back to a pic
    output_image_data_np = rgb_flat.reshape(height, width, 3) 

    final_image_pil = Image.fromarray(output_image_data_np, 'RGB')
    logger.info(f"Vectorized rendering complete. {np.sum(non_white_mask_np)} non-background pixels processed.")
    return final_image_pil



def fast_value_matrix(image_pil_rgba: Image.Image, 
                      known_colors_lab_np: np.ndarray, 
                      known_values_np: np.ndarray, 
                      logger): # Logger hinzugefügt für Konsistenz
    """
    Creates the value matrix and non-white mask in a vectorized way.
    """
    logger.debug(f"Starting fast_value_matrix for image size {image_pil_rgba.size}")

    # Convert to RGB and then to NumPy array for the mask (uint8)
    image_pil_rgb_uint8 = image_pil_rgba.convert("RGB")
    img_rgb_uint8_np = np.asarray(image_pil_rgb_uint8)

    # Create the non-white mask (with uint8 values)
    # A pixel is "not white" if at least one color channel is < 250.
    non_white_mask_np = (img_rgb_uint8_np < 250).any(axis=2) # Shape (H, W)

    # Convert to RGB and then to float32 for LAB conversion
    img_rgb_float32_np = img_rgb_uint8_np.astype(np.float32) / 255.0 # Normalisieren auf [0,1]
    lab_img_np = rgb2lab(img_rgb_float32_np)  # Shape (H, W, 3)

    # Broadcasting for distance calculation
    diff_np = lab_img_np[:, :, np.newaxis, :] - known_colors_lab_np.reshape(1, 1, -1, 3)
    # diff_np now is (H, W, N, 3)

    dist_np = np.linalg.norm(diff_np, axis=3)
    # # dist_np is (H, W, N)

    idx_np = dist_np.argmin(axis=2)  # Indices of the next color for each pixel
    # idx_np is (H, W)

    value_matrix_np = known_values_np[idx_np]
    # value_matrix_np hat is (H, W)

    # For pixels that have been masked as “white” (i.e. not in non_white_mask_np),
    # the value should be 0 if the color distance search has shown something else here.
    # map_color_to_value_gan has already taken this into account.
    value_matrix_np[~non_white_mask_np] = 0.0 

    logger.debug(f"fast_value_matrix complete. Value matrix shape: {value_matrix_np.shape}")
    return value_matrix_np, non_white_mask_np