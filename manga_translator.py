# Python 3.10.10
# torch-2.0.0%2Bcu118-cp310-cp310-win_amd64.whl   https://pytorch.org/                      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# opencv_python-4.7.0.72-cp37-abi3-win_amd64.whl  https://pypi.org/project/opencv-python/   pip install opencv-contrib-python-headless
# manga_ocr-0.1.8.                                https://github.com/kha-white/manga-ocr    pip3 install manga-ocr
# deepl-1.14.0-py3-none-any.whl                   https://github.com/DeepLcom/deepl-python  pip install --upgrade deepl


# not used
# easyocr-1.6.2-py3-none-any.whl                  https://pypi.org/project/easyocr/         pip install easyocr
# deep_translator-1.10.1                          https://github.com/nidhaloff/deep-translator   pip install -U deep-translator 
# https://www.jaided.ai/easyocr/documentation/
# pip install matplotlib


import os
import time
import textwrap
import numpy as np
import cv2
import easyocr as eocr
from manga_ocr import MangaOcr

import translation_helper as th
import text_writer
import font_info
import image_rw as irw
import text_box_intersections




############################################################## TRANSLATE A FULL PAGE FULL PROCESS #########################################################################################
def translate_page(image_filepath):
    _, image_filename = os.path.split(image_filepath)
    
    # identify text on image
    textboxes_list, image = rw_image.get_text_boxes(image_filepath, translator.source_lang)

    original_text, original_crops, bboxes = ([] for i in range(3))

    ############################################################## READ TEXT AND MAKE BBOXES #########################################################################################
    # Iterate through the detected text regions and get text
    for i, textbox in enumerate(textboxes_list):
        rotated_bbox, bbox, cropped_textbox, x, y, w, h = rw_image.get_bboxes(textbox, image)

        # Save the cropped image to a file
        cropped_textbox_image_file = os.path.join(cropped_folder, os.path.splitext(image_filename)[0], f'cropped_text_{i}.jpg')
        if not os.path.exists(cropped_textbox_image_file.removesuffix(f'cropped_text_{i}.jpg')):
            os.makedirs(cropped_textbox_image_file.removesuffix(f'cropped_text_{i}.jpg'))
        cv2.imwrite(cropped_textbox_image_file, cropped_textbox)

        # use manga ocr on text location
        text_from_page = mocr(cropped_textbox_image_file)
        print(text_from_page)

        # remove original text from image
        #image = rw_image.remove_text(image, rotated_bbox)

        # lists for each text box
        original_text.append(text_from_page)
        original_crops.append(cropped_textbox)
        bboxes.append(bbox)


    # check to make sure no boxes intersect
    #bboxes = text_box_intersections.check_all_intersections(bboxes)
    #for bbox in bboxes:
    #    cv2.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (255, 0, 0), 5)


    ############################################################## TRANSLATE #########################################################################################
    # if already translated do not retranslate
    translated_text_list = []
    translated_text_filename = os.path.join(output_folder, os.path.splitext(image_filename)[0] + "_" + translator.getTranslatorService() + "_translated.txt")
    if os.path.isfile(translated_text_filename) & ~retranslate:
        #translated_text_file = open(translated_text_filename,"r+")
        #translated_text = translated_text_file.read()
        #translated_text_file.close()
        with open(translated_text_filename,"r+", encoding='utf-8') as file:
            translated_text = file.read()
    
    else:
        # translate
        translated_text, translated_text_list = translator.translate(original_text)
        print(translated_text)

        # write translation to file so we dont have to retranslate
        #translated_text_file = open(translated_text_filename, "w", encoding='utf-8')
        #translated_text_file.write(translated_text)
        #translated_text_file.close()
        with open(translated_text_filename, "w", encoding='utf-8') as file:
            file.write(translated_text)



    ############################################################## APPLY TRANSLATED TEXT TO IMAGE ####################################################################
    for i, translated_box in enumerate(translated_text_list):
        #print("\ntranslated box: " + translated_box)

        bbox = bboxes[i]
        #box_height, box_width = bbox[1][0] - bbox[0][0], bbox[3][1] - bbox[0][1]
        box_height, box_width, channels = original_crops[i].shape

        translated_box = translated_box.strip()

        
        #get words for each box
        words = translated_box.split()
        #if (words):
        longest_word = max(words, key=len)        ################# do check to see if bad translate
        #else:
        #    print("WORD DIDNT GET TRANSLATED PROPERLY, SKIPPING")
        #    continue

        # Scale text where longest word takes up length of textbox
        font.scale_text_based_on_longest_word(words, longest_word, translated_box, box_width)

        # Determine where text will wrap: Add the text to the box image, wrapping it within the box
        wrapped_lines, y_offset = text_writer.wrap_text_in_box(words, font, box_width)

        # Calculate the total height of the wrapped text and adjust font_scale
        y_offset = font.rescale_font(wrapped_lines, box_height, y_offset)

        # Write the wrapped text to the image
        image = text_writer.write_text_to_page(image, bbox, wrapped_lines, font, box_width, y_offset)



    ############################################################## SAVE FINAL IMAGE ########################################################################################
    # Save the modified image
    savename = os.path.splitext(image_filename)[0] + "_" + th.TranslatorService.SUGOI.value + "_translated.jpg"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print("\n-savename: " + savename + "\n")
    cv2.imwrite(os.path.join(output_folder, savename), image)
    if debug:
        cv2.imshow(image_filename, image)
    




#---------------------------------------------- MAIN ---------------------------------------------------
start_time = time.perf_counter()
total_character_count_translated = 0

mocr = MangaOcr()

debug = True
retranslate = True

# Set the font, scale, and color of the text
font = font_info.Font_info()
#font.load_font()

# detect text on image and manipulate bboxes
rw_image = irw.image_rw()

#for single image
image_filepath = 'imageToTranslate.png'

#for multiple images
base_folder = 'folderToTranslate'
cropped_folder = os.path.join(base_folder, 'cropped_images')
output_folder = os.path.join(base_folder, 'translated')

# Create the output folder if it doesn't exist
if not os.path.exists(cropped_folder):
    os.makedirs(cropped_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# get all of the images in the base folder
entries = os.listdir(base_folder)
pages_list = [entry for entry in entries if os.path.isfile(os.path.join(base_folder, entry))]
print("Pages to translate: ")
print(pages_list)

translator = th.Translator(th.TranslatorService.SUGOI, th.Language.JAPANESE, th.Language.ENGLISH)
print("translation service: ", translator.getTranslatorService())




translate_page(image_filepath)

# translate all pages in folder
#for page in pages_list:
#    translate_page(base_folder + "/" + page)





end_time = time.perf_counter()
time_taken = end_time - start_time
print(f"Time taken: {time_taken:.6f} seconds")
print("Total characters translated: ", total_character_count_translated)

cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Destroy all windows

