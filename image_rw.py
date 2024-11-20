import cv2
import easyocr as eocr
import numpy as np

import translation_helper as th

class image_rw:

    _bbox_scalefactor = 1.0

    def __init__(self, bbox_scalefactor=None):
        self.bbox_scalefactor = bbox_scalefactor         if bbox_scalefactor is not None else self._bbox_scalefactor



    def get_text_boxes(self, image_filepath, source_lang):
        if source_lang == th.Language.JAPANESE: source_abrv = "ja"

        image = cv2.imread(image_filepath)

        # rescale image
        new_height = 1500
        aspect_ratio = image.shape[1] / image.shape[0]
        new_width = int(new_height * aspect_ratio)
        image = cv2.resize(image, (new_width, new_height))

        # rotate image to the left for text detection
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # read page for boxes
        reader = eocr.Reader([source_abrv], model_storage_directory='models', user_network_directory='models', recog_network='japanese_g2')
        #text_boxes_list = reader.readtext(image, paragraph=True, x_ths=.15, y_ths=.15, rotation_info=[90,180,270], add_margin=0.15) #original
        text_boxes_list = reader.readtext(image, paragraph=True, x_ths=.05, y_ths=.5, ycenter_ths=.1, add_margin=0.1, height_ths=.4, min_size=51) 

        # rotate image back to upright
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        return text_boxes_list, image
    


    def get_bboxes(self, text, image):
        original_bbox, text = text  # need extra _ variable if paragraph = false
        
        #print("Original bbox:", original_bbox)
        rotated_bbox = self.rotate_bbox_90_clockwise(original_bbox, image.shape)
        #print("Rotated bbox:", rotated_bbox)
        bbox = self.expand_bbox_proportionally(rotated_bbox, self.bbox_scalefactor, image.shape)
        #print("Scaled bbox:", bbox)
        top_left, top_right, bottom_right, bottom_left = bbox # bbox
        x, y = int(top_left[0]), int(top_left[1])
        w, h = int(bottom_right[0] - top_left[0]), int(bottom_right[1] - top_left[1])
        #print("x:", x, "y:", y, "w:", w, "h:", h)


        ctop_left, ctop_right, cbottom_right, cbottom_left = rotated_bbox # bbox
        cx, cy = int(ctop_left[0]), int(ctop_left[1])
        cw, ch = int(cbottom_right[0] - ctop_left[0]), int(cbottom_right[1] - ctop_left[1])

        # Crop the text region from the original image
        cropped_textbox = image[y:y + h, x:x + w] #cropped_image
        #cropped_textbox = image[cy:cy + ch, cx:cx + cw] #cropped_image


        ################# draw rectangles on pic #################
        #cv2.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)

        return rotated_bbox, bbox, cropped_textbox, x, y, w, h
    


    def remove_text(self, image, rotated_bbox):
        # erase original text on image
        # Create a blank mask of the same size as the image
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # Draw the text bounding boxes as filled rectangles on the mask
        cv2.fillPoly(mask, np.array([rotated_bbox], np.int32), (255))
        # Perform inpainting
        inpaint_radius = 4  # You can adjust this value as needed
        inpaint_method = cv2.INPAINT_TELEA
        image = cv2.inpaint(image, mask, inpaint_radius, inpaint_method)

        return image



    def expand_bbox_proportionally(self, bbox, expand_factor, image_shape):
        height, width = image_shape[:2]
        top_left, top_right, bottom_right, bottom_left = bbox

        bbox_width = top_right[0] - top_left[0]
        bbox_height = bottom_left[1] - top_left[1]

        expanded_width = int(bbox_width * expand_factor + 30)
        expanded_height = int(bbox_height * expand_factor + 30)

        width_diff = (expanded_width - bbox_width) // 2
        height_diff = (expanded_height - bbox_height) // 2

        expanded_top_left = (max(0, top_left[0] - width_diff), max(0, top_left[1] - height_diff))
        expanded_top_right = (min(width, top_right[0] + width_diff), max(0, top_right[1] - height_diff))
        expanded_bottom_right = (min(width, bottom_right[0] + width_diff), min(height, bottom_right[1] + height_diff))
        expanded_bottom_left = (max(0, bottom_left[0] - width_diff), min(height, bottom_left[1] + height_diff))

        expanded_bbox = [expanded_top_left, expanded_top_right, expanded_bottom_right, expanded_bottom_left]

        return expanded_bbox



    # moves the bboxes to their correct spot when the image is rotated
    def rotate_bbox_90_clockwise(self, bbox, image_shape):
        width, height = image_shape[:2]
        center = (width // 2, height // 2)

        top_left, top_right, bottom_right, bottom_left = bbox

        new_top_left = (height - bottom_left[1], bottom_left[0])
        new_top_right = (height - top_left[1], top_left[0])
        new_bottom_right = (height - top_right[1], top_right[0])
        new_bottom_left = (height - bottom_right[1], bottom_right[0])

        rotated_bbox = (new_top_left, new_top_right, new_bottom_right, new_bottom_left)

        return rotated_bbox