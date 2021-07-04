import numpy as np
import pandas as pd
import cv2 as cv
import masks


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
        # cv.imshow('test', np.concatenate((self.img_norm, self.img_corrected_norm), axis=1))
        cv.imshow('original image', self.img_norm)
        cv.imshow('corrected image', self.img_corrected_norm)
        cv.waitKey(1000)

    def save_img(self):
        cv.imwrite(self.filename + '_final.tif', self.img_corrected)

    def save_data(self):
        self.data.to_csv(self.filename + '.csv')

    def tilt_correction(self, threshold):
        self.threshold = self.mean * threshold
        self.upper_point = np.argmax(self.img_original[100, :] > self.threshold)
        self.lower_point = np.argmax(self.img_original[900, :] > self.threshold)
        self.left_point = np.argmax(self.img_original[512, :] > self.threshold)
        self.top_point = np.argmax(self.img_original[:, 512] > self.threshold)

        self.tilt = abs(self.upper_point - self.lower_point)
        self.radian = np.arctan(self.tilt / 800)
        self.degree = self.radian * (180 / np.pi)

        image_center = tuple(np.array(self.img_corrected.shape[1::-1]) / 2)

        if self.upper_point > self.lower_point:
            rot_mat = cv.getRotationMatrix2D(image_center, self.degree, 1.0)
        else:
            rot_mat = cv.getRotationMatrix2D(image_center, - self.degree, 1.0)

        self.img_corrected = cv.warpAffine(self.img_corrected, rot_mat,
                                           self.img_corrected.shape[1::-1], flags=cv.INTER_LINEAR)

        self.img_corrected[self.img_corrected < self.min_value] = self.min_value

    def crop(self, threshold):
        self.threshold = self.mean * threshold
        self.upper_point = np.argmax(self.img_original[100, :] > self.threshold)
        self.lower_point = np.argmax(self.img_original[900, :] > self.threshold)
        self.left_point = np.argmax(self.img_original[512, :] > self.threshold)
        self.top_point = np.argmax(self.img_original[:, 512] > self.threshold)

        self.img_corrected = self.img_corrected[self.top_point:, self.left_point:]

    def analyse(self, mask_option):
        font_scale = 0.8
        thickness = 2
        font = cv.FONT_HERSHEY_SIMPLEX

        for key in masks.mask_dict[mask_option]:
            self.data.loc[self.filename, key] = cv.mean(
                self.img_corrected[masks.mask_dict[mask_option][key][0][1]:masks.mask_dict[mask_option][key][1][1],
                                   masks.mask_dict[mask_option][key][0][0]:masks.mask_dict[mask_option][key][1][0]])[0]

            cv.rectangle(self.img_corrected, masks.mask_dict[mask_option][key][0],
                         masks.mask_dict[mask_option][key][1], int(self.min_value), 2)

            cv.putText(self.img_corrected, key,
                       tuple(map(lambda x, y: x + y, masks.mask_dict[mask_option][key][0], (2, 30))),
                       font, font_scale, int(self.min_value), thickness, cv.LINE_AA)

            cv.putText(self.img_corrected, str(round(self.data.loc[self.filename, key], 2)),
                       tuple(map(lambda x, y: x + y, masks.mask_dict[mask_option][key][0], (2, 60))),
                       font, font_scale, int(self.min_value), thickness, cv.LINE_AA)

    def manual_analyse(self):
        self.img_corrected_norm = cv.normalize(self.img_corrected, None, alpha=0, beta=1,
                                               norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        list_of_rois = cv.selectROIs('Manual Analyse', self.img_corrected_norm, showCrosshair=True)
        return list_of_rois

    def close_windows(self):
        cv.destroyAllWindows()


'''
test = PlImage('test2.tif')
test.tilt_correction()
test.crop()
test.manual_analyse()


list_of_files = ['test1.tif', 'test2.tif', 'test3.tif', 'test4.tif']

for file in list_of_files:
    test = PlImage(file)
    test.tilt_correction()
    test.crop()
    test.analyse('default')
    test.save_img()
    test.save_data()
'''
