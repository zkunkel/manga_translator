import cv2
import PIL

class Font_info:
    _font = cv2.FONT_HERSHEY_SIMPLEX
    _font_color = (0, 0, 0)
    _font_outline_color = (255, 255, 255)
    _original_font_scale = 1
    _original_thickness = 2          # 2
    _original_white_thickness = 7    # 7

    _font_scale = _original_font_scale
    _thickness = _original_thickness
    _white_thickness = _original_white_thickness

    _font_scale_minimum = 1
    _font_scale_maximum = 2

    _line_spacing = 5


    def __init__(self, font=None, font_color=None, font_outline_color=None, original_font_scale=None, original_thickness=None, original_white_thickness=None, font_scale=None, thickness=None, white_thickness=None, line_spacing=None):
        self.font = font                                          if font is not None else self._font
        self.font_color = font_color                              if font_color is not None else self._font_color
        self.font_outline_color = font_outline_color              if font_outline_color is not None else self._font_outline_color
        self.original_font_scale = original_font_scale            if original_font_scale is not None else self._original_font_scale
        self.original_thickness = original_thickness              if original_thickness is not None else self._original_thickness
        self.original_white_thickness = original_white_thickness  if original_white_thickness is not None else self._original_white_thickness
        self.font_scale = font_scale                              if font_scale is not None else self._font_scale
        self.thickness = thickness                                if thickness is not None else self._thickness
        self.white_thickness = white_thickness                    if white_thickness is not None else self._white_thickness
        self.line_spacing = line_spacing                          if line_spacing is not None else self._line_spacing


    def scaleText(self, font_scale_factor):
        # maybe scale amount based in if fontScale is above or below certain value
        # self.font_scale *= (font_scale_factor)
        self.thickness = self.thickness # round(self.thickness * font_scale_factor)
        self.white_thickness = self.white_thickness # round(self.white_thickness * (font_scale_factor * 1))

        #print(self.font_scale, self.thickness, self.white_thickness)

        if self.font_scale < self._font_scale_minimum:
            self.font_scale = self._font_scale_minimum
        if self.font_scale > self._font_scale_maximum:
            self.font_scale = self._font_scale_maximum

        return
    

    def scale_text_to_bbox(self, text, bbox, max_width):
        top_left, top_right, bottom_right, bottom_left = bbox
        bbox_width = top_right[0] - top_left[0]

        # Calculate the size of the text with the initial font scale
        (text_width, text_height), _ = cv2.getTextSize(text, self.font, self.original_font_scale, self.thickness)

        # Calculate the font scale needed to fit the text within the bounding box width
        adjusted_font_scale = min(self.original_font_scale * bbox_width / text_width, max_width / text_width)

        return adjusted_font_scale
    

    def scale_text_based_on_longest_word(self, words, longest_word, translated_box, box_width):
        #find longest word and set font scale so that it fits in the box
        (longestTextWidth, text_height), _ = cv2.getTextSize(longest_word, self.font, self.original_font_scale, self.original_thickness)
        #print("Longest word: '" + longest_word + "' text width: " + str(longestTextWidth))
        #font_scale = scale_text_to_bbox(longest_word, bbox, font, font_scale, thickness, longestTextWidth - 20)
        font_scale_factor = (box_width - 10) / longestTextWidth
        self.scaleText(font_scale_factor)
        #print("font scale: " + str(font_scale))

        return
    

    def rescale_font(self, lines, boxHeight, y_offset):
        #y_offset = 0
        #total_text_height = sum(cv2.getTextSize(line, font, font_scale, thickness)[0][1] + line_spacing for line in lines) - line_spacing
        total_text_height = 0
        for line in lines:
            (text_width, text_height), _ = cv2.getTextSize(line, self.font, self.font_scale, self.thickness)
            total_text_height += text_height + self.line_spacing
        total_text_height -= self.line_spacing

        font_scale_factor = 1
        # if total height is greater than box height
        # while (total_text_height > (boxHeight)): # - 20
        #     #print("old font scale factor: " + str(font_scale_factor))
        #     #print("boxheight: " + str(boxHeight) + " totalTextHeight: " + str(total_text_height))
        #     font_scale_factor = font_scale_factor * ((boxHeight) / total_text_height) #boxHeight - 20
        #     #print("new font scale factor: " + str(font_scale_factor))
        #     self.scaleText(font_scale_factor)
        #     #print("font_scale: " + str(font_scale) + " thickness: " + str(thickness) + " white_thickness: " + str(white_thickness))

        #     # recalculate height
        #     total_text_height = 0
        #     for line in lines:
        #         (text_width, text_height), _ = cv2.getTextSize(line, self.font, self.font_scale, self.thickness)
        #         total_text_height += text_height + self.line_spacing
        #     total_text_height -= self.line_spacing

        # Calculate the initial y_offset to center the text vertically
        y_offset = (boxHeight - (total_text_height + text_height)) // 2

        return y_offset
    

    def load_custom_font(self):
        self.font = PIL.ImageFont.truetype("fonts/wildwordsroman.TTF", 16)
