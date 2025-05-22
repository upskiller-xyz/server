import math
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def align_df_image(image_pil_rgba: Image.Image, rotation_rad: float, translation_gh_mm_xyz: list, target_image_size_wh: tuple, scene_to_pixels_scale_x: float, scene_to_pixels_scale_y: float) -> Image.Image:
    """
    Transforms (rotates, moves) a single RGBA PIL image.
    Empty areas are filled with white.
    """
    logger.debug(f"Aligning image. Input size: {image_pil_rgba.size}, Target output size: {target_image_size_wh}, Rotation (rad): {rotation_rad:.3f}, Translation (mm): {translation_gh_mm_xyz}")
    
    try:
        white_background_rgba = Image.new("RGBA", target_image_size_wh, (255, 255, 255, 255))
        angle_deg = math.degrees(rotation_rad)
        rotated_content_transparent = image_pil_rgba.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=False)
                
        # (translation_gh_mm_xyz[0]) -> positive is "up" -> Pillow's 'f' (vertical offset)
        # (translation_gh_mm_xyz[1]) -> positive is "left" -> Pillow's 'c' (horizontal offset)
        param_c_pil = translation_gh_mm_xyz[1] * scene_to_pixels_scale_y
        param_f_pil = translation_gh_mm_xyz[0] * scene_to_pixels_scale_x

        # TranslationMatrix:
        # a, b, c
        # d, e, f
        affine_matrix_translation_only = (1, 0, param_c_pil, 0, 1, param_f_pil)
        logger.debug(f"Affine translation params (corrected): c={param_c_pil:.2f}px, f={param_f_pil:.2f}px (angle_deg={angle_deg:.2f})")

        try: # Newer
             transform_method = Image.AFFINE
        except AttributeError:
             transform_method = Image.Transform.AFFINE
        
        transformed_content_on_transparent_bg = rotated_content_transparent.transform(
            target_image_size_wh, 
            transform_method, 
            affine_matrix_translation_only, 
            resample=Image.Resampling.BICUBIC
        )
        
        final_aligned_image_pil_rgba = Image.alpha_composite(white_background_rgba, transformed_content_on_transparent_bg)
        
        logger.debug(f"Alignment complete. Output image mode: {final_aligned_image_pil_rgba.mode}, size: {final_aligned_image_pil_rgba.size}")
        return final_aligned_image_pil_rgba
    except Exception as e:
        logger.error(f"Error during image alignment: {e}", exc_info=True)
        raise