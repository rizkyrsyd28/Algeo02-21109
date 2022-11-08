from tkinter import *
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk
import tkinter.messagebox
import customtkinter
import cv2

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):

    WIDTH = 1280
    HEIGHT = 720

    def __init__(self):
        super().__init__()

        self.cap = cv2.VideoCapture(1)
        self.title("Eigenface")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        # ============ frame_left ============

        # configure grid layout (1x11)
        # self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(7, weight=1)  # empty row as spacing
        # self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        # self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="EigenFace",
                                              text_font=("SF Pro Text", 16, "bold"))  # font name and size in px
        self.label_1.grid(row=1, column=0, pady=20, padx=20)

        self.title_insert_img = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Insert Image",
                                              text_font=("SF Pro Text", 12))  # font name and size in px
        self.title_insert_img.grid(row=2, column=0, pady=10, padx=20)
        self.button_insert_img = customtkinter.CTkButton(master=self.frame_left,
                                                text="Choose Image",
                                                text_font=("SF Pro Text", 10),
                                                command=self.open_image)
        self.button_insert_img.grid(row=3, column=0, pady=10, padx=20)
        self.title_insert_fld = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Insert Folder",
                                              text_font=("SF Pro Text", 12))  # font name and size in px
        self.title_insert_fld.grid(row=4, column=0, pady=10, padx=20)

        self.button_insert_fld = customtkinter.CTkButton(master=self.frame_left,
                                                text="Choose Folder",
                                                text_font=("SF Pro Text", 10),
                                                command=self.on_cam)
        self.button_insert_fld.grid(row=5, column=0, pady=10, padx=20)

        self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="Theme :")
        self.label_mode.grid(row=9, column=0, pady=0, padx=20, sticky="w")

        # option menu 
        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                        values=["Light", "Dark", "System"],
                                                        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=10, column=0, pady=10, padx=20, sticky="w")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="", padx=20, pady=20)

        self.subframeR_1 = customtkinter.CTkFrame(master = self.frame_right, width=400, height=400,)
        self.subframeR_1.grid(row=0, column=0, sticky="nswe", padx=20, pady=20)

        self.image_input = customtkinter.CTkLabel(master = self.subframeR_1, text="Input Image", width=400, height=400)
        self.image_input.grid(row=1, column=0, pady=10, padx=10)

        self.subframeR_2 = customtkinter.CTkFrame(master = self.frame_right, width=400, height=400,)
        self.subframeR_2.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        self.image_output = customtkinter.CTkLabel(master = self.subframeR_2, text="Output Image", width=400, height=400)
        self.image_output.grid(row=1, column=0, pady=10, padx=10)

    def button_event(self):
        print("Button pressed")

    def on_cam(self):
        self.img = self.cap.read()[1]
        self.imgBGR= cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        self.cam = Image.fromarray(self.imgBGR)
        self.imgTk = ImageTk.PhotoImage(image=self.cam.resize((500,500)))
        self.image_output.configure(image=self.imgTk)
        self.image_input.after(20,self.on_cam)


    def open_image(self):
        self.filename = filedialog.askopenfile()
        # self.img = cv2.imread(self.filename.name)
        self.img = Image.open(self.filename.name)
        self.imgtk = ImageTk.PhotoImage(self.img.resize((500,500)))
        self.image_input.config(image = self.imgtk)

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_closing(self, event=0):
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()