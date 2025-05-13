# df_inference.py
import numpy as np
from PIL import Image

def preprocess_for_df_ml(image_pil_rgb, logger, target_ml_size_wh, normalization_range):
    """
    Prepares a PIL RGB image for the ML model.
    """
    logger.debug(f"Preprocessing image for ML. Original size: {image_pil_rgb.size}, Target ML size: {target_ml_size_wh}")
    try:
        if image_pil_rgb.mode != 'RGB':
            logger.warning(f"Input image for preprocessing is mode {image_pil_rgb.mode}, converting to RGB.")
            image_pil_rgb = image_pil_rgb.convert('RGB')

        img_resized_pil = image_pil_rgb.resize(target_ml_size_wh, Image.Resampling.BICUBIC)
        img_array_float32 = np.array(img_resized_pil, dtype=np.float32)

        if normalization_range == (-1, 1):
            img_normalized = (img_array_float32 / 127.5) - 1.0
        # elif normalization_range == (0, 1):
        #     img_normalized = img_array_float32 / 255.0
        else:
            logger.warning(f"Unknown ML_NORMALIZATION_RANGE: {normalization_range}. Image will not be normalized.")
            img_normalized = img_array_float32
        
        img_batch = np.expand_dims(img_normalized, axis=0)
        logger.debug(f"Preprocessing complete. Output shape for ML: {img_batch.shape}")
        return img_batch
    except Exception as e:
        logger.error(f"Error during ML preprocessing: {e}", exc_info=True)
        raise


def postprocess_df_ml_prediction(prediction_batch_np, logger, normalization_range):
    """
    Converts the NumPy array of the ML model back into a PIL image (RGB).
    """
    logger.debug(f"Postprocessing ML prediction. Input prediction shape: {prediction_batch_np.shape}")
    try:
        if not isinstance(prediction_batch_np, np.ndarray):
            raise TypeError("ML model output for postprocessing was not a NumPy array.")

        if prediction_batch_np.ndim == 4 and prediction_batch_np.shape[0] == 1:
            pred_np = prediction_batch_np[0] # remove Batch-Dimension
        elif prediction_batch_np.ndim == 3:
            pred_np = prediction_batch_np
        else:
            raise ValueError(f"Unexpected dimensions of ML model output: {prediction_batch_np.shape}")

        # Scale back to [0, 255]
        min_val_pred = np.min(pred_np)
        max_val_pred = np.max(pred_np)
        logger.debug(f"ML prediction value range before scaling: [{min_val_pred:.3f}, {max_val_pred:.3f}]")

        if max_val_pred > min_val_pred:
            if normalization_range == (-1, 1):
                pred_scaled_np = (pred_np + 1.0) * 0.5 * 255.0
            # elif normalization_range == (0, 1):
            #     pred_scaled_np = pred_np * 255.0
            else:
                logger.warning("ML output range for scaling is not strictly defined by ML_NORMALIZATION_RANGE, using min/max scaling.")
                pred_scaled_np = ((pred_np - min_val_pred) / (max_val_pred - min_val_pred)) * 255.0
        else: 
            logger.warning("All values in ML prediction are identical.")
            # Fallback to a
            if normalization_range == (-1,1): constant_scaled_value = (min_val_pred + 1.0) * 0.5 * 255.0
            elif normalization_range == (0,1): constant_scaled_value = min_val_pred * 255.0
            else: constant_scaled_value = 127.5
            pred_scaled_np = np.full_like(pred_np, np.clip(constant_scaled_value,0,255), dtype=np.float32)

        pred_uint8_np = np.clip(pred_scaled_np, 0, 255).astype(np.uint8)

        # Test for 3 channels (RGB)
        if pred_uint8_np.ndim == 2: # Grayscale (H, W)
            logger.info("ML output is grayscale, converting to RGB.")
            pred_uint8_np = np.stack((pred_uint8_np,) * 3, axis=-1)
        elif pred_uint8_np.ndim == 3 and pred_uint8_np.shape[-1] == 1: # Grayscale (H, W, 1)
            logger.info("ML output is (H,W,1), converting to RGB.")
            pred_uint8_np = np.concatenate([pred_uint8_np] * 3, axis=-1)
        
        if pred_uint8_np.shape[-1] != 3:
            raise ValueError(f"Postprocessed ML output does not have 3 channels (RGB): {pred_uint8_np.shape}")

        output_image_pil_rgb = Image.fromarray(pred_uint8_np, 'RGB')
        output_image_pil_rgba = output_image_pil_rgb.convert('RGBA') # Alpha is 255 (opak)


        logger.debug(f"Postprocessing complete. Output PIL image mode: {output_image_pil_rgba.mode}, size: {output_image_pil_rgba.size}")
        return output_image_pil_rgba
    except Exception as e:
        logger.error(f"Error during ML postprocessing: {e}", exc_info=True)
        raise