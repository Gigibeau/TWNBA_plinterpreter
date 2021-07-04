from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import TWNBA_plinterpreter_ocv
import masks

root = Tk()

# Parameters
global list_of_files
global pl_image
var_tilt = IntVar(value=1)
var_crop = IntVar(value=1)
var_visual = IntVar(value=1)
var_img = IntVar(value=1)
list_of_masks = []
for mask in masks.mask_dict:
    list_of_masks.append(mask)

button_open = Button(root, text="open", command=lambda: open_files())
button_open.grid(row=0, column=0)

checkbutton_tilt = Checkbutton(root, text='tilt correction?', variable=var_tilt)
checkbutton_tilt.grid(row=0, column=1)
checkbutton_crop = Checkbutton(root, text='crop?', variable=var_crop)
checkbutton_crop.grid(row=0, column=2)
checkbutton_visual = Checkbutton(root, text='visual?', variable=var_visual)
checkbutton_visual.grid(row=0, column=3)
checkbutton_visual = Checkbutton(root, text='save images?', variable=var_img)
checkbutton_visual.grid(row=0, column=4)

combo_mask = ttk.Combobox(root, values=list_of_masks)
combo_mask.set('Which mask?')
combo_mask.grid(row=1, column=1, columnspan=2)

button_analyse = Button(root, text='Auto Analyse', command=lambda: auto_analyse(var_tilt.get(),
                                                                                var_crop.get(),
                                                                                var_visual.get(),
                                                                                var_img.get(),
                                                                                combo_mask.get()))
button_analyse.grid(row=1, column=0)

button_mask_builder = Button(root, text='Mask Builder', command=lambda: mask_builder())
button_mask_builder.grid(row=1, column=3, columnspan=2)

# Placement of the log
text = Text(root, width=40, height=20)
text.grid(row=2, column=0, columnspan=5)


def open_files():
    global list_of_files
    list_of_files = filedialog.askopenfilenames()

    text.delete('1.0', END)
    for file in list_of_files:
        text.insert('end', file.split('/')[-1] + '\n')
    text.see('end')


def auto_analyse(tilt, crop, visual, img, mask_choice):
    global pl_image
    for file in list_of_files:
        pl_image = TWNBA_plinterpreter_ocv.PlImage(file)
        if tilt == 1:
            pl_image.tilt_correction()

        if crop == 1:
            pl_image.crop()

        if mask_choice == 'Which mask?':
            if visual == 1:
                pl_image.analyse('default')
                pl_image.show_img()
            else:
                pl_image.analyse('default')
        else:
            if visual == 1:
                pl_image.analyse(mask_choice)
                pl_image.show_img()
            else:
                pl_image.analyse(mask_choice)

        if img == 1:
            pl_image.save_img()

        pl_image.save_data()
    pl_image.close_windows()


def mask_builder():
    pl_image_to_mask = TWNBA_plinterpreter_ocv.PlImage(list_of_files[0])
    new_mask = pl_image_to_mask.manual_analyse()
    print(new_mask)


root.mainloop()
