# Python 3.10.10
# torch-2.0.0%2Bcu118-cp310-cp310-win_amd64.whl   https://pytorch.org/                      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# opencv_python-4.7.0.72-cp37-abi3-win_amd64.whl  https://pypi.org/project/opencv-python/   pip install opencv-contrib-python-headless
# manga_ocr-0.1.8.                                https://github.com/kha-white/manga-ocr    pip3 install manga-ocr
# deep_translator-1.10.1                          https://github.com/nidhaloff/deep-translator   pip install -U deep-translator 

# not used
# easyocr-1.6.2-py3-none-any.whl                  https://pypi.org/project/easyocr/         pip install easyocr
# https://www.jaided.ai/easyocr/documentation/
# pip install matplotlib


import os
import time
import textwrap
import numpy as np
import cv2 as cv
import easyocr as eocr
from manga_ocr import MangaOcr
from deep_translator import DeeplTranslator
from deep_translator import GoogleTranslator

import text_box_intersections


def expand_bbox_proportionally(bbox, expand_factor, image_shape):
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
def rotate_bbox_90_clockwise(bbox, image_shape):
    width, height = image_shape[:2]
    center = (width // 2, height // 2)

    top_left, top_right, bottom_right, bottom_left = bbox

    new_top_left = (height - bottom_left[1], bottom_left[0])
    new_top_right = (height - top_left[1], top_left[0])
    new_bottom_right = (height - top_right[1], top_right[0])
    new_bottom_left = (height - bottom_right[1], bottom_right[0])

    rotated_bbox = (new_top_left, new_top_right, new_bottom_right, new_bottom_left)

    return rotated_bbox

def scale_text_to_bbox(text, bbox, font, initial_font_scale, thickness, max_width):
    top_left, top_right, bottom_right, bottom_left = bbox
    bbox_width = top_right[0] - top_left[0]

    # Calculate the size of the text with the initial font scale
    (text_width, text_height), _ = cv.getTextSize(text, font, initial_font_scale, thickness)

    # Calculate the font scale needed to fit the text within the bounding box width
    adjusted_font_scale = min(initial_font_scale * bbox_width / text_width, max_width / text_width)

    return adjusted_font_scale

def wrap_text(text, font, font_scale, thickness, max_width):
    words = text.split()
    lines = []
    line = ""

    for word in words:
        temp_line = line + " " + word if line else word
        temp_line_width, _ = cv.getTextSize(temp_line, font, font_scale, thickness)[0]

        if temp_line_width > max_width:
            lines.append(line)
            line = word
        else:
            line = temp_line

    lines.append(line)
    return lines

def scaleText(fontScale, thickness, whiteThickness, fontScaleFactor):
    # maybe scale amount based in if fontScale is above or below certain value
    fontScale *= fontScaleFactor
    thickness = round(thickness * fontScaleFactor)
    whiteThickness = round(whiteThickness * (fontScaleFactor * .7))

    return fontScale, thickness, whiteThickness


############################################################## TRANSLATE A FULL PAGE FULL PROCESS #########################################################################################
def translate_page(imagePath, page):

    image = cv.imread(imagePath)

    # rescale image
    new_height = 1500
    aspect_ratio = image.shape[1] / image.shape[0]
    new_width = int(new_height * aspect_ratio)
    image = cv.resize(image, (new_width, new_height))

    # rotate image to the left for text detection
    image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)

    # read page for boxes
    reader = eocr.Reader(['ja'], model_storage_directory='models', user_network_directory='models', recog_network='japanese_g2')
    #textList = reader.readtext(image, paragraph=True, x_ths=.15, y_ths=.15, rotation_info=[90,180,270], add_margin=0.15) #
    textList = reader.readtext(image, paragraph=True, x_ths=.15, y_ths=.15, rotation_info=[90,180,270], add_margin=0.20)

    # rotate image back to upright
    image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)

    originalText = []
    originalCrops = []
    originalCropLocations = []
    bboxes = []

    ############################################################## DETECT TEXT AND MAKE BBOXES #########################################################################################
    # Iterate through the detected text regions and get text
    for i, text in enumerate(textList):
        original_bbox, text = text
        
        #print("Original bbox:", original_bbox)
        rotated_bbox = rotate_bbox_90_clockwise(original_bbox, image.shape)
        #print("Rotated bbox:", rotated_bbox)
        bbox = expand_bbox_proportionally(rotated_bbox, 1.2, image.shape)
        #print("Scaled bbox:", bbox)
        top_left, top_right, bottom_right, bottom_left = bbox


        x, y = int(top_left[0]), int(top_left[1])
        w, h = int(bottom_right[0] - top_left[0]), int(bottom_right[1] - top_left[1])
        print("x:", x, "y:", y, "w:", w, "h:", h)

        ################# draw rectangles on pic #################
        cv.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)

        # Crop the text region from the original image
        finalCrop = image[y:y + h, x:x + w] #cropped_image
        
        croppedFiles = os.path.join(cropped_folder, page.removesuffix(".png"), f'cropped_text_{i}.jpg')
        if not os.path.exists(croppedFiles.removesuffix(f'cropped_text_{i}.jpg')):
            os.makedirs(croppedFiles.removesuffix(f'cropped_text_{i}.jpg'))

        # Save the cropped image to a file
        cv.imwrite(croppedFiles, finalCrop)

        ############################################################## MANGA OCR TO FIND TEXT #########################################################################################
        # use manga ocr on text location
        cropTexted = mocr(croppedFiles)
        print(cropTexted)

        # erase original text on image
        # Create a blank mask of the same size as the image
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # Draw the text bounding boxes as filled rectangles on the mask
        cv.fillPoly(mask, np.array([rotated_bbox], np.int32), (255))
        # Perform inpainting
        inpaint_radius = 4  # You can adjust this value as needed
        inpaint_method = cv.INPAINT_TELEA
        image = cv.inpaint(image, mask, inpaint_radius, inpaint_method)

        originalText.append(cropTexted)
        originalCrops.append(finalCrop)
        originalCropLocations.append((x,y))
        bboxes.append(bbox)


    # check to make sure no boxes intersect
    #bboxes = text_box_intersections.check_all_intersections(bboxes)
    #for bbox in bboxes:
    #    cv.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (255, 0, 0), 5)
        

    ############################################################## TRANSLATE #########################################################################################
    # Concatenate the text with a unique separator
    separator = "\n"
    concatenated_text = separator.join(originalText)

    # Translate the entire concatenated text
    translated_text = GoogleTranslator(source='ja', target='en').translate(concatenated_text)
    #translated_text = "EEEE EEEE EEEEE EEEEE EEEEEEEE EE EEE E EEEE"
    print(translated_text)

    # Split the translated text back into individual translations
    translated_texts = translated_text.split(separator)


    ############################################################## APPLY TRANSLATED TEXT TO IMAGE ####################################################################
    for i, translated_box in enumerate(translated_texts):
        print("\ntranslated box: " + translated_box)

        finalCrop = originalCrops[i]
        bbox = bboxes[i]

        translated_box = translated_box.strip()

        boxHeight, boxWidth, channels = finalCrop.shape

        ############################################################## FONT SETTINGS #########################################################################################
        # Set the font, scale, and color of the text
        font = cv.FONT_HERSHEY_SIMPLEX
        original_font_scale = 1
        font_color = (200, 0, 0)  # Red color (B, G, R, A)
        original_thickness = 2
        original_white_thickness = 8
        line_spacing = 5

        x, y = originalCropLocations[i]

        ############################################################## SCALE TEXT BASE ON LONGEST WORD #########################################################################################
        #get words for each box
        words = translated_box.split()
        #if (words):
        longest_word = max(words, key=len)
        #else:
        #    print("WORD DIDNT GET TRANSLATED PROPERLY, SKIPPING")
        #    continue
        

        #find longest word and set font scale so that it fits in the box
        (longestTextWidth, text_height), _ = cv.getTextSize(longest_word, font, original_font_scale, original_thickness)
        print("Longest word: '" + longest_word + "' text width: " + str(longestTextWidth))
        #font_scale = scale_text_to_bbox(longest_word, bbox, font, font_scale, thickness, longestTextWidth - 20)
        font_scale_factor = (boxWidth - 10) / longestTextWidth
        font_scale, thickness, white_thickness = scaleText(original_font_scale, original_thickness, original_white_thickness, font_scale_factor)
        print("font scale: " + str(font_scale))


        ############################################################## DETERMINE WHERE TEXT WILL WRAP #########################################################################################
        # Add the text to the box image, wrapping it within the box
        y_offset = 0
        line = ""
        lines = []
        for word in words:
            temp_line = line + " " + word if line else word
            #font_scale = scale_text_to_bbox(temp_line, bbox, font, font_scale, thickness, width)
            (text_width, text_height), _ = cv.getTextSize(temp_line, font, font_scale, thickness)
            text_height += line_spacing
            
            if text_width <= boxWidth:
                line = temp_line
            else:
                #font_scale = scale_text_to_bbox(line, bbox, font, font_scale, thickness, width)
                (text_width, text_height), _ = cv.getTextSize(line, font, font_scale, thickness)
                text_height += line_spacing
                x_offset = ((boxWidth - text_width) // 2)# + (text_width // 2)
                #white text and then black text
                lines.append(line)
                #cv.putText(box_image, line, (x_offset, y_offset + text_height), font, font_scale, (255, 255, 255), thickness + 8, cv.LINE_AA)
                #cv.putText(box_image, line, (x_offset, y_offset + text_height), font, font_scale, (0, 0, 0), thickness, cv.LINE_AA)
                y_offset += text_height# - 5
                line = word

        if line:
            #font_scale = scale_text_to_bbox(line, bbox, font, font_scale, thickness, width)
            (text_width, text_height), _ = cv.getTextSize(line, font, font_scale, thickness)
            text_height += line_spacing
            x_offset = ((boxWidth - text_width) // 2)# + (text_width // 2)
            lines.append(line)
            #cv.putText(box_image, line, (x_offset, y_offset + text_height), font, font_scale, (255, 255, 255), thickness + 8, cv.LINE_AA)
            #cv.putText(box_image, line, (x_offset, y_offset + text_height), font, font_scale, (0, 0, 0), thickness, cv.LINE_AA)

        
        ############################################################## GET TOTAL HEIGHT OF TEXT AND ADJUST FONTSCALE #########################################################################################
        # Calculate the total height of the wrapped text
        #total_text_height = sum(cv.getTextSize(line, font, font_scale, thickness)[0][1] + line_spacing for line in lines) - line_spacing
        total_text_height = 0
        for line in lines:
            (text_width, text_height), _ = cv.getTextSize(line, font, font_scale, thickness)
            total_text_height += text_height + line_spacing
        total_text_height -= line_spacing

        # if total height is greater than box height
        while (total_text_height > (boxHeight)): # - 20
            print("old font scale factor: " + str(font_scale_factor))
            print("boxheight: " + str(boxHeight) + " totalTextHeight: " + str(total_text_height))
            font_scale_factor = font_scale_factor * ((boxHeight) / total_text_height) #boxHeight - 20
            print("new font scale factor: " + str(font_scale_factor))
            font_scale, thickness, white_thickness = scaleText(original_font_scale, original_thickness, original_white_thickness, font_scale_factor)
            print("font_scale: " + str(font_scale) + " thickness: " + str(thickness) + " white_thickness: " + str(white_thickness))

            # recalculate height
            total_text_height = 0
            for line in lines:
                (text_width, text_height), _ = cv.getTextSize(line, font, font_scale, thickness)
                total_text_height += text_height + line_spacing
            total_text_height -= line_spacing

        # Calculate the initial y_offset to center the text vertically
        y_offset = (boxHeight - (total_text_height + text_height)) // 2


        ############################################################## WRITE TEXT TO PAGE #########################################################################################
        # Add the text to the box image, wrapping it within the box
        for line in lines:
            (text_width, text_height), _ = cv.getTextSize(line, font, font_scale, thickness)
            text_height += line_spacing

            # Calculate the x_offset for the current line
            x_offset = (boxWidth - text_width) // 2
            cv.putText(image, line, (bbox[0][0] + x_offset, bbox[0][1] + y_offset + text_height), font, font_scale, (255, 255, 255), thickness + white_thickness, cv.LINE_AA)
            cv.putText(image, line, (bbox[0][0] + x_offset, bbox[0][1] + y_offset + text_height), font, font_scale, (0, 0, 0), thickness, cv.LINE_AA)
            y_offset += text_height + line_spacing

        #blend with original image
        #box_image = cv.rotate(box_image, cv.ROTATE_90_COUNTERCLOCKWISE)
        # Blend the box image with the original image using cv2.addWeighted()
        #alpha = 0  # Weight of the original image (0 <= alpha <= 1)
        #beta = 1 - alpha  # Weight of the box_image
        #image[y:y + boxWidth, x:x + boxHeight] = cv.addWeighted(image[y:y + boxWidth, x:x + boxHeight], alpha, box_image, beta, 0)


    ############################################################## SAVE FINAL IMAGE ########################################################################################
    # Save the modified image
    savename = page.removesuffix(".png") + "_translated.jpg"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print("savename: " + savename)
    cv.imwrite(os.path.join(output_folder, savename), image)
    if debug:
        cv.imshow(page, image)
    




#---------------------------------------------- MAIN ---------------------------------------------------
start_time = time.perf_counter()

mocr = MangaOcr()

debug = True

#for single image
imagePath = 'ahmad/022.jpg'

#for multiple images
base_folder = 'ahmad'
cropped_folder = os.path.join(base_folder, 'cropped_images')
output_folder = os.path.join(base_folder, 'translated')
#folderToTranslate = 'test'
#cropped_folder = 'cropped_images'
#output_path = 'output'


# Create the output folder if it doesn't exist
if not os.path.exists(cropped_folder):
    os.makedirs(cropped_folder)

entries = os.listdir(base_folder)
pagesList = [entry for entry in entries if os.path.isfile(os.path.join(base_folder, entry))]
print("Pages to translate: ")
print(pagesList)

translate_page(imagePath, "eeeeee.png")

# translate all pages in folder
#for page in pagesList:
#    translate_page(base_folder + "/" + page, page)


end_time = time.perf_counter()
time_taken = end_time - start_time
print(f"Time taken: {time_taken:.6f} seconds")

cv.waitKey(0)  # Wait for a key press to close the window
cv.destroyAllWindows()  # Destroy all windows

