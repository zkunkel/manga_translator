import cv2
import font_info


def wrap_text_in_box(words, font: font_info.Font_info, box_width):
    y_offset = 0
    line = ""
    lines = []
    for word in words:
        temp_line = line + " " + word if line else word
        #font_scale = scale_text_to_bbox(temp_line, bbox, font, font_scale, thickness, width)
        (text_width, text_height), _ = cv2.getTextSize(temp_line, font.font, font.font_scale, font.thickness)
        text_height += font.line_spacing
        
        if text_width <= box_width:
            line = temp_line
        else:
            #font_scale = scale_text_to_bbox(line, bbox, font, font_scale, thickness, width)
            (text_width, text_height), _ = cv2.getTextSize(line, font.font, font.font_scale, font.thickness)
            text_height += font.line_spacing
            x_offset = ((box_width - text_width) // 2)# + (text_width // 2)
            #white text and then black text
            lines.append(line)
            #cv2.putText(box_image, line, (x_offset, y_offset + text_height), font, font_scale, (255, 255, 255), thickness + 8, cv2.LINE_AA)
            #cv2.putText(box_image, line, (x_offset, y_offset + text_height), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            y_offset += text_height# - 5
            line = word

    if line:
        #font_scale = scale_text_to_bbox(line, bbox, font, font_scale, thickness, width)
        (text_width, text_height), _ = cv2.getTextSize(line, font.font, font.font_scale, font.thickness)
        text_height += font.line_spacing
        x_offset = ((box_width - text_width) // 2)# + (text_width // 2)
        lines.append(line)
        #cv2.putText(box_image, line, (x_offset, y_offset + text_height), font, font_scale, (255, 255, 255), thickness + 8, cv2.LINE_AA)
        #cv2.putText(box_image, line, (x_offset, y_offset + text_height), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return lines, y_offset



def write_text_to_page(image, bbox, lines, font: font_info.Font_info, box_width, y_offset):
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(line, font.font, font.font_scale, font.thickness)
        text_height += font.line_spacing

        # Calculate the x_offset for the current line
        x_offset = (box_width - text_width) // 2
        cv2.putText(image, line, (bbox[0][0] + x_offset, bbox[0][1] + y_offset + text_height), font.font, font.font_scale, font.font_outline_color, font.thickness + font.white_thickness, cv2.LINE_AA)
        cv2.putText(image, line, (bbox[0][0] + x_offset, bbox[0][1] + y_offset + text_height), font.font, font.font_scale, font.font_color, font.thickness, cv2.LINE_AA)
        y_offset += text_height + font.line_spacing

    return image