import cv2
import os
import tkinter as tk
import tkinter.font as tkFont
import tkinter
import tkinter.filedialog
import numpy as np
import tkinter.messagebox 

class App:
    def __init__(self, root):
        # Fixed window size (16:9 ratio)
        self.window_width = 1000
        self.window_height = 1000
        self.root = root
        global rgbformat
        global frame
        global heightratio
        global Widthratio
        heightratio = 1
        Widthratio = 1
        # Center the window
        display_width = root.winfo_screenwidth()
        display_height = root.winfo_screenheight()
        x_offset = (display_width - self.window_width) // 2
        y_offset = (display_height - self.window_height) // 2
        root.geometry(f"{self.window_width}x{self.window_height}+{x_offset}+{y_offset}")
        root.resizable(width=False, height=False)
        root.configure(bg="black")

        # Initialize UI components
        self.create_widgets()

    def create_widgets(self):
        # Title Label
        self.title_label = tk.Label(
            self.root, text="An AI Machine Learning Colorizer", bg="black", fg="#ffec00", font=("unispace", 15)
        )
        self.title_label.place(relx=0.5, rely=0.1, anchor="center")  # Centered at 10% height

        # Info Label
        self.info_label = tk.Label(
            self.root,
            text="Convert old black and white images to color through the power of AI!",
            wraplength=self.window_width * 0.8,
            bg="black",
            fg="#ffec00",
            font=("Terminal", 10),
            justify="center",
        )
        self.info_label.place(relx=0.5, rely=0.25, anchor="center")  # Centered at 25% height

        # Image Selection Button
        self.img_select_button = tk.Button(
            self.root,
            text="Select Image",
            bg="#550000",
            fg="#fff44f",
            activebackground="#ffffff",
            activeforeground="#000000",
            font=("System", 10),
            command=self.Colorization,
        )
        self.img_select_button.place(relx=0.5, rely=0.55, anchor="center", width=self.window_width * 0.45, height=50)

        # Image Size Label
        self.img_size_label = tk.Label(
            self.root, text="Output Image Size:", bg="black", fg="#ffec08", font=("System", 8)
        )
        self.img_size_label.place(relx=0.1, rely=0.8, anchor="w")

        # Size Buttons
        button_width = self.window_width * 0.15
        button_height = 35

        self.large_button = tk.Button(
            self.root,
            text="Large",
            bg="#500000",
            fg="#fff44f",
            activebackground="#00FF00",
            activeforeground="#cc0000",
            font=("System", 8),
            command=self.Large_command,
        )
        self.large_button.place(relx=0.75, rely=0.88, anchor="center", width=button_width, height=button_height)

        self.medium_button = tk.Button(
            self.root,
            text="Medium",
            bg="#500000",
            fg="#fff44f",
            activebackground="#0000FF",
            activeforeground="#ff0000",
            font=("System", 8),
            command=self.Medium_command,
        )
        self.medium_button.place(relx=0.5, rely=0.88, anchor="center", width=button_width, height=button_height)

        self.small_button = tk.Button(
            self.root,
            text="Small",
            bg="#500000",
            fg="#fff44f",
            activebackground="#ff0000",
            activeforeground="#fad400",
            font=("System", 8),
            command=self.Small_command,
        )
        self.small_button.place(relx=0.25, rely=0.88, anchor="center", width=button_width, height=button_height)
        
    def Colorization(self):

    # This function performs colorization of black-and-white images using a pre-trained neural network model.
    # It allows the user to select an image file, processes the image to add color using the model,
    # and displays the original and colorized images side by side. The user can choose to process more images
    # or exit the process.


    # Declare global variables to store processed frames, format, output images, and display sizes
        global frame
        global rgbformat
        global bgrout1
        global imshowSize
        global imshowSize2
        
        var = True  # A flag to control the loop for processing multiple images
        while var:  # Loop to allow repeated image colorization

            # Open a file dialog to select an image
            path = tkinter.filedialog.askopenfilename()
            
            # Break the loop if no file is selected
            if len(path) == 0:
                break

            # Read the selected image using OpenCV
            cap = cv2.VideoCapture(path)
            cv2.destroyAllWindows()  # Close any OpenCV windows

            # Set the width and height for the neural network's input
            w = 224
            h = 224

            # Load the pre-trained Caffe model and associated weights
            net = cv2.dnn.readNetFromCaffe('colorization.prototxt', 'colorization_release.caffemodel')
            
            # Load cluster centers (color data used for colorizing)
            kernel = np.load('kernel.npy')  
            kernel = kernel.transpose()  # Transpose kernel for the correct shape
            kernel = kernel.reshape(2, 313, 1, 1)  # Reshape to fit the model's expected input

            # Assign the cluster centers to specific layers of the model
            net.getLayer(net.getLayerId('class8_ab')).blobs = [kernel.astype(np.float32)]
            net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

            # Read the first frame (image) from the selected file
            ret, frame = cap.read()

            # Convert the image from BGR to RGB format and normalize pixel values to [0, 1]
            rgbformat = (frame[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)

            # Convert the image to Lab color space (L: lightness, a/b: color channels)
            labformat = cv2.cvtColor(rgbformat, cv2.COLOR_RGB2Lab)
            l = labformat[:, :, 0]  # Extract the L (lightness) channel

            # Get the original dimensions of the image
            (H_orignal, W_orignal) = rgbformat.shape[:2]

            # Resize the image to the neural network's required input size
            rs = cv2.resize(rgbformat, (w, h))
            lab_rs = cv2.cvtColor(rs, cv2.COLOR_RGB2Lab)
            lrs = lab_rs[:, :, 0]  # Extract the resized lightness channel
            lrs -= 50  # Subtract mean value (50) for normalization

            # Pass the lightness channel through the neural network
            net.setInput(cv2.dnn.blobFromImage(lrs))
            ab = net.forward()[0, :, :, :].transpose((1, 2, 0))  # Output a and b color channels

            # Get the dimensions of the neural network's output
            (H_out, W_out) = ab.shape[:2]

            # Resize the a and b channels to match the original image dimensions
            ab_us = cv2.resize(ab, (W_orignal, H_orignal))

            # Combine the original lightness channel with the predicted a and b channels
            labout1 = np.concatenate((l[:, :, np.newaxis], ab_us), axis=2)

            # Convert the Lab image back to BGR and clip pixel values to [0, 1]
            bgrout1 = np.clip(cv2.cvtColor(labout1, cv2.COLOR_Lab2BGR), 0, 1)

            # Read the original image to check its dimensions
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            inputheight = img.shape[0]
            inputwidth = img.shape[1]

            # Set the output image dimensions based on user-selected size ratios
            outputY = int(inputwidth * Widthratio)
            outputX = int(inputheight * heightratio)
            imshowSize = (outputY, outputX)  # Full-size display
            imshowSize2 = (int(outputY / 2.5), int(outputX / 2.5))  # Scaled-down display

            # Release the video capture object (image is loaded into memory)
            cap.release()

            # Call methods to display the original and colorized images
            self.imagecomparision()
            self.outputstage()

            # Ask the user if they want to colorize another image
            MsgBox = tk.messagebox.askquestion(
                'How do you want to proceed?',
                'Do you want to colorize another image?',
                icon='question'
            )
            
            # Update the loop flag based on the user's choice
            if MsgBox == 'yes':
                var = True
            else:
                var = False
    def imagecomparision(self):
        MsgBox = tk.messagebox.askquestion(
            'How do you want to proceed?',
            'Do you want to see the output result compared with the original?',
            icon='question'
        )
        
        if MsgBox == 'yes':
            global frame
            global imshowSize2
            frame_resized = cv2.resize(frame, imshowSize2)
            imgout_resized = cv2.resize(bgrout1, imshowSize2)
            original_resized = cv2.resize(rgbformat, imshowSize2)
            img_concate_hori = np.concatenate((original_resized, imgout_resized), axis=1)
            
            cv2.imshow('Comparison: Original vs Colorized', img_concate_hori)
            
            # Use a non-blocking wait and handle window closure
            while True:
                key = cv2.waitKey(1) & 0xFF
                if cv2.getWindowProperty('Comparison: Original vs Colorized', cv2.WND_PROP_VISIBLE) < 1:
                    break  # Window closed by the user
            cv2.destroyAllWindows()
            tk.messagebox.showinfo(
                'Save File',
                'In the next screen select your output folder and input a suitable name for your colorized picture'
            )
        else:
            tk.messagebox.showinfo(
                'Save File',
                'In the next screen select your output folder and input a suitable name for your colorized picture'
            )

    def outputstage(self):
        
            imagetosave = bgrout1 * 255
            outputfile = cv2.resize(imagetosave, imshowSize)
            file = tkinter.filedialog.asksaveasfile(mode='w', defaultextension=".png", filetypes=(("PNG file", "*.png"),("All Files", "*.*") ))
            if file:
             abs_path = os.path.abspath(file.name)
             cv2.imwrite(abs_path , outputfile)
    
    def Large_command(self):
         global heightratio
         global Widthratio
         Widthratio= 1.5
         heightratio=  1.5

    def Small_command(self):
        global heightratio
        global Widthratio
        Widthratio= 0.5
        heightratio= 0.5
      
    def Medium_command(self):
        global heightratio
        global Widthratio
        Widthratio= 1.2
        heightratio= 1.2
        
      
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
