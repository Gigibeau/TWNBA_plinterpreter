from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox
import pickle
import pandas as pd
import TWNBA_plinterpreter

root = Tk()

# Parameters
global list_of_files
global pl_image
global list_of_masks
var_tilt = IntVar(value=1)
var_crop = IntVar(value=1)
var_visual = IntVar(value=1)
var_img = IntVar(value=1)

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

label_threshold = Label(root, text='threshold:')
label_threshold.grid(row=1, column=0)
entry_threshold = Entry(root, width=5)
entry_threshold.insert(END, 0.95)
entry_threshold.grid(row=1, column=1)

combo_mask = ttk.Combobox(root, values=[])
combo_mask.set('Which mask?')
combo_mask.grid(row=2, column=1, columnspan=2)


def update_mask_list():
    global list_of_masks
    with open('masks.pickle', 'rb') as pickle_file:
        mask_dict_conv = pickle.load(pickle_file)
    list_of_masks = []
    for mask in mask_dict_conv:
        list_of_masks.append(mask)
    combo_mask.config(values=list_of_masks)


update_mask_list()

button_analyse = Button(root, text='Auto Analyse', command=lambda: auto_analyse(var_tilt.get(),
                                                                                var_crop.get(),
                                                                                var_visual.get(),
                                                                                var_img.get(),
                                                                                combo_mask.get(),
                                                                                float(entry_threshold.get())))
button_analyse.grid(row=2, column=0)

button_mask_builder = Button(root, text='Mask Builder', command=lambda: mask_builder(var_tilt.get(),
                                                                                     var_crop.get(),
                                                                                     float(entry_threshold.get())))
button_mask_builder.grid(row=2, column=3, columnspan=2)

button_rapid_mask = Button(root, text='Rapid Mask', command=lambda: rapid_mask(var_tilt.get(),
                                                                               var_crop.get(),
                                                                               float(entry_threshold.get())))
button_rapid_mask.grid(row=1, column=3, columnspan=2)

# Placement of the log
text = Text(root, width=40, height=20)
text.grid(row=3, column=0, columnspan=5)

button_merge = Button(root, text="merge .csv files", command=lambda: merge_csv())
button_merge.grid(row=4, column=0)


def open_files():
    global list_of_files
    list_of_files = filedialog.askopenfilenames()

    text.delete('1.0', END)
    for file in list_of_files:
        text.insert('end', file.split('/')[-1] + '\n')
    text.see('end')


def auto_analyse(tilt, crop, visual, img, mask_choice, threshold):
    global pl_image

    try:
        for file in list_of_files:
            pl_image = TWNBA_plinterpreter.PlImage(file)
            if tilt == 1:
                pl_image.tilt_correction(threshold)

            if crop == 1:
                pl_image.crop(threshold)

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
        TWNBA_plinterpreter.close_windows()

    except NameError:
        messagebox.showerror('Error', 'You forgot to open a file. Open a file(s) first!')


def mask_builder(tilt, crop, threshold):
    with open('masks.pickle', 'rb') as pickle_mask_file:
        new_mask_dict = pickle.load(pickle_mask_file)

    input_title = simpledialog.askstring('Input', 'What should be the title of the mask', parent=root)
    input_fields_str = simpledialog.askstring('Input', 'What are the names of the fields (, sep (like .csv))',
                                              parent=root)
    input_fields_list = input_fields_str.split(',')
    new_mask_dict[input_title] = {field: [] for field in input_fields_list}
    try:
        pl_image_to_mask = TWNBA_plinterpreter.PlImage(list_of_files[0])

        if tilt == 1:
            pl_image_to_mask.tilt_correction(threshold)

        if crop == 1:
            pl_image_to_mask.crop(threshold)
        new_mask = pl_image_to_mask.manual_analyse()

        if len(input_fields_list) != len(new_mask):
            messagebox.showerror('Error', 'Number of fields doesnt match the number of names. Try again!')
        else:
            for field in range(len(input_fields_list)):
                new_mask_dict[input_title][input_fields_list[field]] = [(new_mask[field][0], new_mask[field][1]),
                                                                        (new_mask[field][0] + new_mask[field][2],
                                                                         new_mask[field][1] + new_mask[field][3])]

            with open('masks.pickle', 'wb') as fp:
                pickle.dump(new_mask_dict, fp)

            update_mask_list()
            TWNBA_plinterpreter.close_windows()

    except NameError:
        messagebox.showerror('Error', 'You forgot to open a file. Open a file to build a mask!')


def rapid_mask(tilt, crop, threshold):
    with open('masks.pickle', 'rb') as pickle_mask_file:
        new_mask_dict = pickle.load(pickle_mask_file)

    input_title = simpledialog.askstring('Input', 'What should be the title of the mask', parent=root)

    try:
        pl_image_to_mask = TWNBA_plinterpreter.PlImage(list_of_files[0])

        if tilt == 1:
            pl_image_to_mask.tilt_correction(threshold)

        if crop == 1:
            pl_image_to_mask.crop(threshold)

        new_mask = pl_image_to_mask.rapid_analyse()
        num_of_points = new_mask.shape[0]
        new_mask_dict[input_title] = {str(field): [] for field in range(num_of_points)}

        for field in range(num_of_points):
            new_mask_dict[input_title][str(field)] = [(new_mask[field][0], new_mask[field][1]),
                                                      (new_mask[field][0] + new_mask[field][2],
                                                       new_mask[field][1] + new_mask[field][3])]

        with open('masks.pickle', 'wb') as fp:
            pickle.dump(new_mask_dict, fp)

        update_mask_list()

    except NameError:
        messagebox.showerror('Error', 'You forgot to open a file. Open a file to build a mask!')


def merge_csv():
    df_csvs = pd.DataFrame()
    list_of_csvs = filedialog.askopenfilenames()
    for csv in list_of_csvs:
        df_to_add = pd.read_csv(csv)
        df_csvs = pd.concat([df_csvs, df_to_add])
    df_csvs.to_csv('summary.csv')


root.mainloop()
