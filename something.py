# Python 3.10.10
# torch-2.0.0%2Bcu118-cp310-cp310-win_amd64.whl   https://pytorch.org/                      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# opencv_python-4.7.0.72-cp37-abi3-win_amd64.whl  https://pypi.org/project/opencv-python/   pip install opencv-contrib-python-headless
# easyocr-1.6.2-py3-none-any.whl                  https://pypi.org/project/easyocr/         pip install easyocr
# pip install matplotlib

# https://www.jaided.ai/easyocr/documentation/
# https://github.com/kha-white/manga-ocr
# https://github.com/nidhaloff/deep-translator

import os
import textwrap
import numpy as np
import cv2 as cv
import easyocr as eocr
from manga_ocr import MangaOcr
from deep_translator import DeeplTranslator
from deep_translator import GoogleTranslator


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


def scale_text_to_bbox(text, bbox, font, initial_font_scale, thickness, max_width):
    top_left, top_right, bottom_right, bottom_left = bbox
    bbox_width = top_right[0] - top_left[0]

    # Calculate the size of the text with the initial font scale
    (text_width, text_height), _ = cv.getTextSize(text, font, initial_font_scale, thickness)

    # Calculate the font scale needed to fit the text within the bounding box width
    adjusted_font_scale = min(initial_font_scale * bbox_width / text_width, max_width / text_width)

    return adjusted_font_scale

def rotate_point(origin, point, angle_rad):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle_rad) * (px - ox) - np.sin(angle_rad) * (py - oy)
    qy = oy + np.sin(angle_rad) * (px - ox) + np.cos(angle_rad) * (py - oy)

    return int(qx), int(qy)
'''
def rotate_bbox(bbox, image_shape, angle_degrees):
    angle_rad = np.radians(angle_degrees)

    image_height, image_width = image_shape[:2]
    image_center = (image_width // 2, image_height // 2)

    rotated_bbox = []
    for point in bbox:
        rotated_point = rotate_point(image_center, point, angle_rad)
        rotated_bbox.append(rotated_point)

    return rotated_bbox
'''
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
    fontScale *= fontScaleFactor
    thickness = round(thickness * fontScaleFactor)
    whiteThickness = round(whiteThickness * fontScaleFactor)

    return fontScale, thickness, whiteThickness



def translate_page(imagePath):

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
    textList = reader.readtext(image, paragraph=True, x_ths=.15, y_ths=.15, rotation_info=[90,180,270], add_margin=0.15) #

    # rotate image back to upright
    image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)

    originalText = []
    originalCrops = []
    originalCropLocations = []
    bboxes = []

    # Iterate through the detected text regions and get text
    for i, text in enumerate(textList):
        bbox, text = text
        
        print("Original bbox:", bbox)
        bbox = rotate_bbox_90_clockwise(bbox, image.shape)
        print("Rotated bbox:", bbox)
        bbox = expand_bbox_proportionally(bbox, 1.2, image.shape)
        print("Scaled bbox:", bbox)
        
        
        top_left, top_right, bottom_right, bottom_left = bbox


        x, y = int(top_left[0]), int(top_left[1])
        w, h = int(bottom_right[0] - top_left[0]), int(bottom_right[1] - top_left[1])
        print("x:", x, "y:", y, "w:", w, "h:", h)

        #cv.rectangle(image, (5, 5), (10, 10), (0, 255, 0), 5)

        #draw rectangles on pic
        cv.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
        # Crop the text region from the original image
        finalCrop = image[y:y + h, x:x + w] #cropped_image

        # Rotate the cropped image 90 degrees
        #finalCrop = cv.rotate(cropped_image, cv.ROTATE_90_CLOCKWISE)
        
        croppedLocation = os.path.join(cropped_folder, f'cropped_text_{i}.jpg')
        # Save the cropped image to a file
        cv.imwrite(croppedLocation, finalCrop)

        # use manga ocr on text location
        cropTexted = mocr(croppedLocation)
        print(cropTexted)

        originalText.append(cropTexted)
        originalCrops.append(finalCrop)
        originalCropLocations.append((x,y))
        bboxes.append(bbox)
        

    #### Translate
    # Concatenate the text with a unique separator
    separator = "|"
    concatenated_text = separator.join(originalText)

    # Translate the entire concatenated text
    translated_text = GoogleTranslator(source='ja', target='en').translate(concatenated_text)
    #translated_text = "EEEE EEEE EEEEE EEEEE EEEEEEEE EE EEE E EEEE"
    print(translated_text)

    # Split the translated text back into individual translations
    translated_texts = translated_text.split(separator)


    #### apply text to image
    for i, translated_box in enumerate(translated_texts):
        print("\ntranslated box: " + translated_box)

        finalCrop = originalCrops[i]
        bbox = bboxes[i]

        translated_box = translated_box.strip()
        #if ~translated_box

        # Create a black image with the same size as the cropped image
        #black_image = np.zeros_like(finalCrop)
        boxHeight, boxWidth, channels = finalCrop.shape
        white_image = np.full((boxHeight, boxWidth, channels), 255, dtype=np.uint8)
        alpha = 0.2
        # Apply alpha blending to fade the cropped image
        #box_image = cv.addWeighted(finalCrop, alpha, white_image, 1 - alpha, 0)
        box_image = finalCrop


        # Set the font, scale, and color of the text
        font = cv.FONT_HERSHEY_SIMPLEX
        original_font_scale = 1
        font_color = (200, 0, 0)  # Red color (B, G, R, A)
        original_thickness = 2
        original_white_thickness = 8
        line_spacing = 5


        x, y = originalCropLocations[i]


        #get words for each box
        words = translated_box.split()
        longest_word = max(words, key=len)

        #find longest word and set font scale so that it fits in the box
        (longestTextWidth, text_height), _ = cv.getTextSize(longest_word, font, original_font_scale, original_thickness)
        print("Longest word: " + longest_word + " text width: " + str(longestTextWidth))
        #font_scale = scale_text_to_bbox(longest_word, bbox, font, font_scale, thickness, longestTextWidth - 20)
        font_scale_factor = (boxWidth - 10) / longestTextWidth
        font_scale, thickness, white_thickness = scaleText(original_font_scale, original_thickness, original_white_thickness, font_scale_factor)
        print("font scale: " + str(font_scale))


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

        
        # Calculate the total height of the wrapped text
        print("line: " + line)
        #total_text_height = sum(cv.getTextSize(line, font, font_scale, thickness)[0][1] + line_spacing for line in lines) - line_spacing
        total_text_height = 0
        for line in lines:
            (text_width, text_height), _ = cv.getTextSize(line, font, font_scale, thickness)
            total_text_height += text_height + line_spacing
        total_text_height -= line_spacing

        # if total height is greater than box height
        while (total_text_height > (boxHeight - 20)):
            print("old font scale factor: " + str(font_scale_factor))
            print("boxheight: " + str(boxHeight) + " totalTextHeight: " + str(total_text_height))
            font_scale_factor = font_scale_factor * ((boxHeight - 20) / total_text_height)
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
        y_offset = (boxHeight - total_text_height) // 2

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



    #image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    #image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    # Save the modified image
    cv.imwrite(output_path, image)  # Convert back to BGR before saving
    cv.imshow('girl', image)
    




mocr = MangaOcr()
folderToTranslate = 'test'
imagePath = 'test/ears.png'
cropped_folder = 'cropped_images'
output_path = 'output_image.jpg'

# Create the output folder if it doesn't exist
if not os.path.exists(cropped_folder):
    os.makedirs(cropped_folder)

entries = os.listdir(folderToTranslate)
pagesList = [entry for entry in entries if os.path.isfile(os.path.join(folderToTranslate, entry))]
print(pagesList)
translate_page(imagePath)
# translate all pages in folder
#for page in pagesList:
    #translate_page(folderToTranslate + "/" + page)


cv.waitKey(0)  # Wait for a key press to close the window
cv.destroyAllWindows()  # Destroy all windows











'''
#image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)
image = cv.medianBlur(image,1)
grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)



#ret, imageBinary = cv.threshold(image, 90, 255, cv.THRESH_BINARY, cv.THRESH_OTSU) #imgf contains Binary image
#image = cv.adaptiveThreshold(grey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 30)
#th4 = cv.adaptiveThreshold(grey, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, -5)
#cv.imshow('girlBINary', image)

# read page
reader = eocr.Reader(['ja'], model_storage_directory='models', user_network_directory='models', recog_network='japanese_g2')
textList = reader.readtext(image, paragraph=True, x_ths=.05, y_ths=.05, rotation_info=[90,180,270], add_margin=0.15) #






#horizontal_list, free_list = reader.detect(image)
#textList = reader.recognize(img_cv_grey=grey, horizontal_list=horizontal_list, free_list=free_list)

f = open("output.txt", "a", encoding='utf8')
# apply text to picture
for texth in textList:
    #print(text)
    #bbox, text, score = text
    bbox, texth = texth
    #f.write(text)
    #print('bbox: ' + ' '.join(map(str, bbox)) + '\ntext: ' + text + '\nscore: ') # + str(score)
    cv.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
f.close()
'''


