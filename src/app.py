from tkinter import *
from tkinter import filedialog
# import tkinter as tk
from PIL import Image, ImageTk
# import tkinter.messagebox 
import customtkinter
import cv2
import utils as utl
import main
# import time

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):

    WIDTH = 1280
    HEIGHT = 720

    def __init__(self):
        super().__init__()
        
        self.imgRes = None

        self.cam = 0
        self.cap = cv2.VideoCapture(self.cam)
        self.status_cam = False
        self.dir = ""

        self.result = 'NaN'
        self.count_time = 'NaN'
        self.camera_status = 'Off'

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
        self.frame_left.grid_rowconfigure(9, weight=1)  # empty row as spacing
        # self.frame_left.grid_rowconfigure(9, minsize=20)    # empty row with minsize as spacing
        # self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="EigenFace",
                                              text_font=("SF Pro Text", 16, "bold"))  # font name and size in px
        self.label_1.grid(row=1, column=0, pady=20, padx=20)

    # Insert Image 
        self.title_insert_img = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Insert Image",
                                              text_font=("SF Pro Text", 12))  # font name and size in px
        self.title_insert_img.grid(row=2, column=0, pady=10, padx=20)

        self.button_insert_img = customtkinter.CTkButton(master=self.frame_left,
                                                text="Choose Image",
                                                text_font=("SF Pro Text", 10),
                                                command=self.open_image)
        self.button_insert_img.grid(row=3, column=0, pady=10, padx=20)

    # Insert Folder
        self.title_insert_fld = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Insert Folder",
                                              text_font=("SF Pro Text", 12))  # font name and size in px
        self.title_insert_fld.grid(row=4, column=0, pady=10, padx=20)

        self.button_insert_fld = customtkinter.CTkButton(master=self.frame_left,
                                                text="Choose Folder",
                                                text_font=("SF Pro Text", 10),
                                                command=self.open_folder)
        self.button_insert_fld.grid(row=5, column=0, pady=10, padx=20)

    
    # Oncam 
        self.title_oncam = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Camera",
                                              text_font=("SF Pro Text", 12))  # font name and size in px
        self.title_oncam.grid(row=6, column=0, pady=10, padx=20)

        self.button_oncam = customtkinter.CTkSwitch(master=self.frame_left,
                                                text=f"Camera {self.camera_status}",
                                                text_font=("SF Pro Text", 10),
                                                command=self.on_cam, 
                                                onvalue=1, 
                                                offvalue=0)
        self.button_oncam.grid(row=7, column=0, pady=10, padx=20)

    # Theme Menu
        self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="Theme :")
        self.label_mode.grid(row=12, column=0, pady=0, padx=20, sticky="w")

        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                        values=["Light", "Dark", "System"],
                                                        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=13, column=0, pady=10, padx=20, sticky="w")


        self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="Camera :")
        self.label_mode.grid(row=10, column=0, pady=0, padx=20, sticky="w")
        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                        values=['0', '1', '2'],
                                                        command=self.change_cam)
        self.optionmenu_1.grid(row=11, column=0, pady=10, padx=20, sticky="w")

    # Frame Right
        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="", padx=20, pady=20)

        self.frame_footer1 = customtkinter.CTkLabel(text=f"Result : {self.result}", master=self.frame_right)
        self.frame_footer1.grid(row=2, column=0, padx=10, sticky='w', pady=10)
        self.frame_footer2 = customtkinter.CTkLabel(text=f"Time : {self.count_time}", master=self.frame_right)
        self.frame_footer2.grid(row=2, column=1, padx=10, sticky='w', pady=10)

        self.frame_footer3 = customtkinter.CTkButton(text=f"Start", master=self.frame_right, command=self.start)
        self.frame_footer3.grid(row=3, column=0, columnspan=2, padx=10, sticky='', pady=10)

        self.title_R1 = customtkinter.CTkLabel(master=self.frame_right,
                                              text="Test Image",
                                              text_font=("SF Pro Text", 16, "bold"))  # font name and size in px
        self.title_R1.grid(row=0, column=0, pady=10, padx=20, sticky="s")

        self.subframeR_1 = customtkinter.CTkFrame(master = self.frame_right, width=400, height=400,)
        self.subframeR_1.grid(row=1, column=0, sticky="nswe", padx=20, pady=0)

        self.image_input = customtkinter.CTkLabel(master = self.subframeR_1, text="Input Image", width=400, height=400)
        self.image_input.grid(row=1, column=0, pady=10, padx=10)

        self.title_R1 = customtkinter.CTkLabel(master=self.frame_right,
                                              text="Predict Image",
                                              text_font=("SF Pro Text", 16, "bold"))  # font name and size in px
        self.title_R1.grid(row=0, column=1, pady=10, padx=20, sticky="s")

        self.subframeR_2 = customtkinter.CTkFrame(master = self.frame_right, width=400, height=400,)
        self.subframeR_2.grid(row=1, column=1, sticky="nswe", padx=20, pady=0)

        self.image_output = customtkinter.CTkLabel(master = self.subframeR_2, text="Output Image", width=400, height=400)
        self.image_output.grid(row=1, column=0, pady=10, padx=10)


    def on_cam(self):
        if not self.status_cam:
            self.cap = cv2.VideoCapture(self.cam)
            self.camera_status = "On"
            self.status_cam = True
        if self.button_oncam.get() == 1 :
            self.img = self.cap.read()[1]
            imgBGR= cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
            imgcrop = utl.crop_cam(imgBGR)
            cam = Image.fromarray(imgcrop)
            self.imgtk = ImageTk.PhotoImage(image=cam.resize((500,500)))
            self.image_input.configure(image=self.imgtk)
            self.image_input.after(20,self.on_cam)
        # i+=1
        # print(i)
        
        else:
            self.image_input.configure(image='')
            self.status_cam = False
            self.camera_status = "Off"
            self.cap.release()
            return

    def open_image(self):
        filename = filedialog.askopenfile()
        # print(filename)
        # self.img = cv2.imread(self.filename.name)
        if (filename != None):
            self.img = cv2.imread(filename.name)
            img = Image.open(filename.name)
            self.imgtk = ImageTk.PhotoImage(img.resize((400,400)))
            self.image_input.config(image = self.imgtk)
        else:
            self.image_input.config(image = '')
            

    def open_folder(self):
        utl.temp_folder()
        filename = filedialog.askdirectory()
        self.dir = filename 
        # DEBUG
        print("[DEBUG] => You Choose Directory : " + self.dir)

    def change_cam(self, choice):
        self.cam = int(choice)
        print("[DEBUG] => You Choose Camera " + choice)

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)
    
    def start(self):
        if self.imgtk != None : 
            if (self.status_cam):
                print("[DEBUG] GET FROM CAM")
                self.result, self.count_time, self.imgRes = main.predictImageIndex(self.img, self.dir+"/")
            else :
                print("[DEBUG] GET FROM IMAGE")
                self.result, self.count_time, self.imgRes = main.predictImageIndex(self.img, self.dir+"/")
            img = cv2.imread(self.imgRes, 1)
            imgBGR= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            cam = Image.fromarray(imgBGR)
            self.imgHasil = ImageTk.PhotoImage(image=cam.resize((500,500)))
            self.image_output.configure(image=self.imgHasil)
        else :
            print("[DEBUG] [WARN] => START ACTION !!!!")

    def on_closing(self, event=0):
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()