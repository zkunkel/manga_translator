# manga_translator

A proof of concept manga / bulk image translator. Also has a tool (snip_to_translate.py) to let you snip screenshots on your screen directly for quick translations.

This tool can take in a single image or folder of images, dectect the text on each image and translate it and then write the translated text back on top of the image.


Uses: 
- Manga OCR for the text detection https://github.com/kha-white/manga-ocr
- open cv for the image/text manipulation https://pypi.org/project/opencv-python/
- for translation can use google translate / deepl / offline models

### Example before after:

<img width="300" src="https://github.com/user-attachments/assets/aeb8f541-0365-4924-bd5f-1f5665bcaebb">
<img width="300" src="https://github.com/user-attachments/assets/d1f6f45e-f9bc-4239-b922-a3fd00be3e6e">



### Snip to translate example: This program works the same way as windows snipping tool. Here I selected around the text on the left
![image](https://github.com/user-attachments/assets/5a1bb402-eea4-47f7-a6c9-f60cc8203e40)
