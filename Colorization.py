import cv2
import os
import tkinter as tk
import tkinter.font as tkFont
import tkinter
import tkinter.filedialog
import numpy as np
import tkinter.messagebox 

class App:
    
    global rgbformat
    global frame
    global heightratio
    global Widthratio
    heightratio = 1
    Widthratio = 1     
    
    def __init__(self, root):
          
        root.title("Pico V1.0")
        #setting window size
        width=450 
        height=500
        #root.attributes('-topmost', 1)
        displaywidth = root.winfo_screenwidth()
        displayheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (displaywidth - width) / 2, (displayheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)
        root.configure(bg='black')
        
        
        class Buttons: 
            
            Large=tk.Button(root)
            Large["activebackground"] = "#00FF00"
            Large["activeforeground"] = "#cc0000"
            Large["bg"] = "#500000"
            Large["cursor"] = "star"
            ft = tkFont.Font(family='System',size=8)
            Large["font"] = ft
            Large["fg"] = "#fff44f"
            Large["justify"] = "center"
            Large["text"] = "Large"
            Large["relief"] = "sunken"
            Large.place(x=323,y=440,width=70,height=25)
            Large["command"] = self.Large_command
    
            Medium=tk.Button(root)
            Medium["activebackground"] = "#0000FF"
            Medium["activeforeground"] = "#ff0000"
            Medium["bg"] = "#500000"
            Medium["cursor"] = "star"
            ft = tkFont.Font(family='System',size=8)
            Medium["font"] = ft
            Medium["fg"] = "#fff44f"
            Medium["justify"] = "center"
            Medium["text"] = "Medium"
            Medium["relief"] = "sunken"
            Medium.place(x=193,y=440,width=70,height=25)
            Medium["command"] = self.Medium_command
    
            Small=tk.Button(root)
            Small["activebackground"] = "#ff0000"
            Small["activeforeground"] = "#fad400"
            Small["bg"] = "#500000"
            Small["cursor"] = "star"
            ft = tkFont.Font(family='System',size=8)
            Small["font"] = ft
            Small["fg"] = "#fff44f"
            Small["justify"] = "center"
            Small["text"] = "Small"
            Small["relief"] = "sunken"
            Small.place(x=70,y=440,width=70,height=25)
            Small["command"] = self.Small_command
    
            imgselect=tk.Button(root)
            imgselect["activebackground"] = "#ffffff"
            imgselect["activeforeground"] = "#000000"
            imgselect["bg"] = "#550000"
            imgselect["cursor"] = "spraycan"
            ft = tkFont.Font(family='System',size=8)
            imgselect["font"] = ft
            imgselect["fg"] = "#fff44f"
            imgselect["justify"] = "center"
            imgselect["text"] = "Select image"
            imgselect["relief"] = "sunken"
            imgselect.place(x=150,y=285,width=160,height=30)
            imgselect["command"] = self.Colorization
            
        class label: 
            
            maintitle=tk.Label(root)
            ft = tkFont.Font(family='unispace',size=15)
            maintitle["font"] = ft
            maintitle["fg"] = "#ffec00"
            maintitle["bg"] = "#000000"
            maintitle["justify"] = "center"
            maintitle["text"] = " An AI Machine Learning Colorizer "
            maintitle.place(x=0,y=25,width=450,height=79)
            
            
    
            infolabel= tk.Label(root, text ="Convert old black and white images to color through the power of AI! ", wraplength=180)
            infolabel["anchor"] = "w"
            ft = tkFont.Font(family='Terminal',size=10)
            infolabel["font"] = ft
            infolabel["fg"] = "#ffec00"
            infolabel["bg"] = "#000000"
            infolabel["justify"] = "left"
          
            infolabel["padx"] = 5
                       
            infolabel["relief"] = "flat"
            infolabel.place(x=10,y=120)
              
            
            imglabel=tk.Label(root)
            ft = tkFont.Font(family='System',size=10)
            imglabel["font"] = ft
            imglabel["fg"] = "#C8A2C8"
            imglabel["bg"] = "#000000"
            imglabel["justify"] = "center"
            imglabel["text"] = "Tap the button to get started!"
            imglabel.place(x=140,y=240,width=190,height=30)
            
            Imgsize=tk.Label(root)
            ft = tkFont.Font(family='system',size=6)
            Imgsize["font"] = ft
            Imgsize["fg"] = "#ffec08"
            Imgsize["bg"] = "#000000"
            Imgsize["justify"] = "center"
            Imgsize["text"] = "Output Image Size : "
            Imgsize.place(x=10,y=400,width=150,height=20)
        
    def Colorization(self):
        
        global frame
        global rgbformat
        global bgrout1
        global imshowSize
        
        global imshowSize2
        var = True 
        while var == True :
            
            path = 0
            path = tkinter.filedialog.askopenfilename()
            if len(path) == 0:
             break
            cap = cv2.VideoCapture(path)
            cv2.destroyAllWindows()
                   
            w = 224
            h = 224

            # Select desired mod[el
            net = cv2.dnn.readNetFromCaffe('colorization.prototxt', 'colorization_release.caffemodel')
            kernel = np.load('kernel.npy') # load cluster centers
            
            # populate cluster centers as 1x1 
            kernel = kernel.transpose()
            kernel = kernel.reshape(2, 313, 1, 1)
         
            
            net.getLayer(net.getLayerId('class8_ab')).blobs = [kernel.astype(np.float32)]
            
            net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]
                                
            ret, frame = cap.read()
    
            rgbformat = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
    
            labformat = cv2.cvtColor(rgbformat, cv2.COLOR_RGB2Lab)
            
            l = labformat[:,:,0] 
         
            (H_orignal,W_orignal) = rgbformat.shape[:2] 
            
           # resize image to input size 
            rs = cv2.resize(rgbformat, (w, h)) 
            lab_rs = cv2.cvtColor(rs, cv2.COLOR_RGB2Lab)
            lrs = lab_rs[:,:,0]
            
            # subtract 50 for mean
            lrs -= 50 
      
            # Set the input for forwarding through neural network
            net.setInput(cv2.dnn.blobFromImage(lrs))
            
            
            ab= net.forward()[0,:,:,:].transpose((1,2,0))
        
            # retreive the a and b channels
            (H_out,W_out) = ab.shape[:2]
            
            #Resize to original size
            ab_us = cv2.resize(ab, (W_orignal, H_orignal))
   
            # join togheter with original image i.e. L channel 
            labout1 = np.concatenate((l[:,:,np.newaxis],ab_us),axis=2)
            
        
            bgrout1 = np.clip(cv2.cvtColor(labout1, cv2.COLOR_Lab2BGR), 0, 1)
            
            # Checks input image dimensions.
            img = cv2.imread(path , cv2.IMREAD_UNCHANGED)
            inputheight = img.shape[0] 
            inputwidth = img.shape[1]
            
            # Sets output image size
            
            outputY = int(inputwidth * Widthratio)
            outputX = int(inputheight * heightratio)
            imshowSize = (outputY, outputX)
            imshowSize2 = (int(outputY/2.5), int(outputX/2.5))
            
            cap.release()    
            self.imagecomparision()
            self.outputstage()
            
            MsgBox = tk.messagebox.askquestion ('How do you want to proceed?','Do you want to colorize another image?',icon = 'question')
               
            if MsgBox == 'yes':
                     var = True
            else: 
                var = False
                
            
    def imagecomparision(slef):
                     
             MsgBox = tk.messagebox.askquestion ('How do you want to proceed?','Do you want to see the output result compared with the original?',icon = 'question')
             
             if MsgBox == 'yes':
                 
                global frame 
                global imshowSize
                global imshowSize2
                frame = cv2.resize(frame, imshowSize2)
                imgoutss = cv2.resize(bgrout1, imshowSize2)
                orignal = cv2.resize(rgbformat, imshowSize2)
                img_concate_Hori = np.concatenate((orignal,imgoutss),axis=1)
                cv2.imshow('concatenated',img_concate_Hori)
                cv2.waitKey(0)
                tk.messagebox.showinfo('Save File',' In the next screen select your output folder and input a suitable name for your colorized picture')
                
             else:
                 
              tk.messagebox.showinfo('Save File',' In the next screen select your output folder and input a suitable name for your colorized picture')
                 
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
