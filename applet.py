from tkinter import *
from PIL import ImageTk, Image
import pandas as pd

def forward(img_no):
    global label1
    global label2
    global button_forward
    global button_back
    global button_exit
    label1.grid_forget()
    label2.grid_forget()

    label1 = Label(image=BLSList_images[img_no-1])
    label2 = Label(image=FoldList_images[img_no-1])
    label1.grid(row=1, column=0)
    label2.grid(row=1, column=1)

    button_forward = Button(root, text="Forward",
                            command=lambda: forward(img_no+1))

    if img_no == len(BLSList_images):
        button_forward = Button(root, text="Forward", state=DISABLED)

    button_back = Button(root, text="Back",
                         command=lambda: back(img_no-1))

    button_back.grid(row=5, column=0)
    button_exit.grid(row=5, column=1)
    button_forward.grid(row=5, column=2)

def back(img_no):
    global label1
    global label2
    global button_forward
    global button_back
    global button_exit
    label1.grid_forget()
    label2.grid_forget()

    label1 = Label(image=BLSList_images[img_no - 1])
    label2 = Label(image=FoldList_images[img_no - 1])
    label1.grid(row=1, column=0)
    label2.grid(row=1, column=1)

    button_forward = Button(root, text="Forward",
                            command=lambda: forward(img_no + 1))
    button_back = Button(root, text="Back",
                         command=lambda: back(img_no - 1))

    if img_no == 1:
        button_back = Button(root, text="Back", state=DISABLED)

    button_back.grid(row=5, column=0)
    button_exit.grid(row=5, column=1)
    button_forward.grid(row=5, column=2)

def search_tic():
    tic_id = entry.get()
    try:
        img_index = tic_ids.index(int(tic_id))
        label1.grid_forget()
        label2.grid_forget()
        label1.config(image=BLSList_images[img_index])
        label2.config(image=FoldList_images[img_index])
        label1.grid(row=1, column=0)
        label2.grid(row=1, column=1)
    except ValueError:
        print("TIC ID not found.")

root = Tk()
root.title("Image Viewer")
root.geometry("800x500")

df = pd.read_csv('candidates3.csv', on_bad_lines='skip', header=0)
tic_ids = df['Target ID'].tolist()
BLSList_images = [ImageTk.PhotoImage((Image.open(f"/Users/aavikwadivkar/Documents/Exoplanets/Research/Pure_WD_LCs/Rp = {id}_lc.png")).resize((400, 300))) for id in range(1, 9)]
FoldList_images = [ImageTk.PhotoImage((Image.open(f"/Users/aavikwadivkar/Documents/Exoplanets/Research/Pure_WD_LCs/Rp = {id}_blsplot.png")).resize((600, 300))) for id in range(1, 9)]
label1 = Label(image=BLSList_images[0])
label2 = Label(image=FoldList_images[0]) 
label1.grid(row=1, column=0)
label2.grid(row=1, column=1)

button_back = Button(root, text="Back", command=lambda: back(1), state=DISABLED)
button_exit = Button(root, text="Exit", command=root.quit)
button_forward = Button(root, text="Forward", command=lambda: forward(2))

button_back.grid(row=5, column=0)
button_exit.grid(row=5, column=1)
button_forward.grid(row=5, column=2)

entry = Entry(root, width=20)
entry.grid(row=6, column=0, columnspan=2)
search_button = Button(root, text="Search TIC ID", command=search_tic)
search_button.grid(row=6, column=2)

root.mainloop()
