import pyautogui
from PIL import Image
from PIL import ImageGrab
from pynput import mouse
import tkinter as tk
import easyocr as eocr
from manga_ocr import MangaOcr
from deep_translator import GoogleTranslator
import os

#pip install pyinstaller

class TransparentOverlay(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master, bg="white")
        self.attributes('-alpha', 0.2)  # Set window transparency
        self.attributes('-topmost', True)  # Keep the window on top
        self.overrideredirect(True)  # Remove window borders and title bar
        self.geometry("{0}x{1}+0+0".format(self.winfo_screenwidth(), self.winfo_screenheight()))  # Fullscreen window

        self.start_x = None
        self.start_y = None
        self.rect_id = None
        
        self.canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.on_left_button_down)
        #self.canvas.bind("<B1-Motion>", self.on_left_button_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_button_up)

    def on_left_button_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        #self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red", width=2)

    #def on_left_button_move(self, event):
    #    self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_left_button_up(self, event):
        self.find_text(event.x, event.y)
        self.withdraw()
        #self.capture_screenshot(self.start_x, self.start_y, event.x, event.y)
        self.destroy()
        self.quit()

    def find_text(self, click_x, click_y):
        # take repeated screenshots until there is only 1 text box
        y1 = click_y - self.winfo_screenheight() // 4
        if y1 < 0: y1 = 0
        y2 = click_y + self.winfo_screenheight() // 4
        if y1 < self.winfo_screenheight(): y1 = self.winfo_screenheight()
        x1 = click_x - self.winfo_screenheight() // 5
        x2 = click_x + self.winfo_screenheight() // 5

        y1_percent_move = (click_y // self.winfo_screenheight()) * 100

        print("x1: ", x1, "y1: ", y1, "x2: ", x2, "y2: ", y2)

        number_of_textboxes = 2
        while number_of_textboxes > 1:
            print("x1:", x1, "y1:", y1, "x2:", x2, "y2:", y2)
            self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=3)
            print("rectangle done")
            x1 += 100
            x2 -= 100
            y1 += y1_percent_move
            y2 -= (100 - y1_percent_move)
            screenshot = self.capture_screenshot(x1, y1, x2, y2)
            print("screenshot done")
            text_boxes_list = self.process_for_text(os.path.join(self.output_folder, "japanese_text.png"), 'ja')
            print("text boxes list done")
            number_of_textboxes = len(text_boxes_list)


    def capture_screenshot(self, x1, y1, x2, y2):
        x, y = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)
        screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
        
        self.output_folder = "temp_translation"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        screenshot.save(os.path.join(self.output_folder, "japanese_text.png"))
        #screenshot.show()
        return screenshot

    def process_for_text(self, image, source_abrv):
        print("inside process for text")
        print("image to read: ", image)
        reader = eocr.Reader(['ja'], model_storage_directory='models', user_network_directory='models', recog_network='japanese_g2')
        print("reader done")
        text_boxes_list = reader.readtext(image, paragraph=True, rotation_info=[90,180,270], add_margin=0.1, canvas_size=self.winfo_screenheight()) #
        print("in function text boxes list done")
        return text_boxes_list

    def read_text(self, textbox_file):
        text_from_page = mocr(textbox_file)
        return text_from_page





def on_close():
    app.destroy()
    exit()

def on_button_click():
    
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    overlay = TransparentOverlay()
    overlay.mainloop()

    output_folder = "temp_translation"
    text_from_image = mocr(os.path.join(output_folder, "japanese_text.png"))

    translated_text = GoogleTranslator(source='ja', target='en').translate(text_from_image)
    label.config(text=f"{translated_text}")




# Create the main application window
app = tk.Tk()
app.title("clicky translaty")

# Create a label
label = tk.Label(app, text="text", font=("Arial", 25))
label.pack(padx=150, pady=100)

# Create a button
button = tk.Button(app, text="Click to translate!", command=on_button_click)
button.pack(padx=10, pady=10)

# Bind the close event to the on_close function
app.protocol("WM_DELETE_WINDOW", on_close)


mocr = MangaOcr()

# Start the main event loop
app.mainloop()








root = tk.Tk()
root.withdraw()  # Hide the root window
overlay = TransparentOverlay()
overlay.mainloop()


'''
def on_click(x, y, button, pressed):
    if pressed:
        coords.append((x, y))
    else:
        coords.append((x, y))
        return False

# Capture the mouse events
with mouse.Listener(on_click=on_click) as listener:
    listener.join()

# Calculate the coordinates and size of the region
x1, y1 = coords[0]
x2, y2 = coords[1]
x, y = min(x1, x2), min(y1, y2)
width, height = abs(x2 - x1), abs(y2 - y1)

# Take a screenshot
screenshot = pyautogui.screenshot()

# Crop the screenshot to the desired region
region = screenshot.crop((x, y, x + width, y + height))

# Save the cropped screenshot to a file
region.save("screenshot.png")

# Display the cropped screenshot
region.show()
'''