import numpy as np
import pandas as pd
import cv2 as cv
import pickle
from scipy import stats

pos_list = []


class PlImage:
    def __init__(self, file_name):
        self.filename = file_name.split('.')[0]
        self.img_original = cv.imread(file_name, flags=(cv.IMREAD_LOAD_GDAL | cv.IMREAD_ANYDEPTH))
        self.mean = cv.mean(self.img_original)[0]
        self.min_value = self.img_original.min()
        self.img_corrected = self.img_original
        self.data = pd.DataFrame()

        self.threshold = None
        self.upper_point = None
        self.lower_point = None
        self.left_point = None
        self.top_point = None
        self.img_norm = None
        self.img_corrected_norm = None
        self.tilt = None
        self.radian = None
        self.degree = None

    def show_img(self):
        self.img_norm = cv.normalize(self.img_original, None, alpha=0, beta=1,
                                     norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        self.img_corrected_norm = cv.normalize(self.img_corrected, None, alpha=0, beta=1,
                                               norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)



        cv.imshow('corrected image', self.img_corrected_norm)
        cv.moveWindow('corrected image', 0, 0)
        win_position = cv.getWindowImageRect('corrected image')
        cv.imshow('original image', self.img_norm)
        cv.moveWindow('original image', win_position[2], 0)
        cv.waitKey(1000)

    def save_img(self):
        cv.imwrite(self.filename + '_final.tif', self.img_corrected)

    def save_data(self):
        self.data.to_csv(self.filename + '.csv')

    def tilt_correction(self, threshold):
        self.threshold = self.mean * threshold

        x_values = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        y_values = []

        for x_value in x_values:
            y_values.append(np.argmax(self.img_original[x_value, :] > self.threshold))

        slope, _, __, ___, ____ = stats.linregress(x_values, y_values)

        self.tilt = abs(slope)
        self.radian = np.arctan(self.tilt / 800)
        self.degree = self.radian * (180 / np.pi)

        image_center = tuple(np.array(self.img_corrected.shape[1::-1]) / 2)

        if slope <= 0:
            rot_mat = cv.getRotationMatrix2D(image_center, self.degree, 1.0)
        else:
            rot_mat = cv.getRotationMatrix2D(image_center, - self.degree, 1.0)

        self.img_corrected = cv.warpAffine(self.img_corrected, rot_mat,
                                           self.img_corrected.shape[1::-1], flags=cv.INTER_LINEAR)

        self.img_corrected[self.img_corrected < self.min_value] = self.min_value


    def crop(self, threshold):
        self.threshold = self.mean * threshold

        x_values = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        y_values_left = []
        y_values_top = []

        for x_value in x_values:
            y_values_left.append(np.argmax(self.img_original[x_value, :] > self.threshold))

        self.left_point = int(np.mean(y_values_left))

        for x_value in x_values:
            y_values_top.append(np.argmax(self.img_original[:, x_value] > self.threshold))

        self.top_point = int(np.mean(y_values_top))

        self.img_corrected = self.img_corrected[self.top_point:, self.left_point:]

    def analyse(self, mask_option):
        font_scale = 0.8
        thickness = 2
        font = cv.FONT_HERSHEY_SIMPLEX

        with open('masks.pickle', 'rb') as pickle_file:
            mask_dict = pickle.load(pickle_file)

        for key in mask_dict[mask_option]:
            self.data.loc[self.filename.split('/')[-1], key] = cv.mean(
                self.img_corrected[mask_dict[mask_option][key][0][1]:mask_dict[mask_option][key][1][1],
                                   mask_dict[mask_option][key][0][0]:mask_dict[mask_option][key][1][0]])[0]

            cv.rectangle(self.img_corrected, mask_dict[mask_option][key][0],
                         mask_dict[mask_option][key][1], int(self.min_value), 2)

            cv.putText(self.img_corrected, key,
                       tuple(map(lambda x, y: x + y, mask_dict[mask_option][key][0], (2, 30))),
                       font, font_scale, int(self.min_value), thickness, cv.LINE_AA)

            cv.putText(self.img_corrected, str(round(self.data.loc[self.filename.split('/')[-1], key], 2)),
                       tuple(map(lambda x, y: x + y, mask_dict[mask_option][key][0], (2, 60))),
                       font, font_scale, int(self.min_value), thickness, cv.LINE_AA)

    def manual_analyse(self):
        self.img_corrected_norm = cv.normalize(self.img_corrected, None, alpha=0, beta=1,
                                               norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        array_of_rois = cv.selectROIs('Manual Analyse', self.img_corrected_norm, showCrosshair=True)
        return array_of_rois

    def rapid_analyse(self):
        global pos_list
        pos_list = []
        self.img_corrected_norm = cv.normalize(self.img_corrected, None, alpha=0, beta=1,
                                               norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        
        controls_rapid = cv.imread('controls_rapid_mask.jpg')
        cv.imshow('Controls', controls_rapid)
        cv.moveWindow('Controls', 0, 0)
        roi = cv.selectROI('ROI Selection', self.img_corrected_norm, showCrosshair=True)
        
        cv.destroyWindow('ROI Selection')
        cv.imshow('Rapid Analyse', self.img_corrected_norm)
        cv.setMouseCallback('Rapid Analyse', on_click)
        while True:
            k = cv.waitKey(1000)
            if k == 32:
                cv.destroyAllWindows()
                break

        list_of_rois = []
        for position in pos_list:
            roi_position = [position[0] - (roi[2] / 2),
                            position[1] - (roi[3] / 2),
                            roi[2],
                            roi[3]]
            list_of_rois.append(roi_position)

        array_of_rois = np.array(list_of_rois)
        return array_of_rois.astype(int)


def on_click(event, x, y, _, __):
    global pos_list
    if event == cv.EVENT_LBUTTONDOWN:
        pos_list.append((x, y))


def close_windows():
    cv.destroyAllWindows()
