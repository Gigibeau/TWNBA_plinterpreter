from PIL import Image
import numpy as np



class PlImage:
    def __init__(self, file_name, threshold):
        self.img_array = np.array(Image.open(file_name))
        self.img_original = Image.open(file_name)
        self.img_corrected = None
        self.img_corrected_array = None
        self.upper_point = np.argmax(self.img_array[100, :] > threshold)
        self.lower_point = np.argmax(self.img_array[900, :] > threshold)
        self.tilt = abs(self.upper_point - self.lower_point)
        self.radian = np.arctan(self.tilt / 800)
        self.degree = self.radian * (180 / np.pi)

    def tilt_correction(self):
        if self.upper_point > self.lower_point:
            self.img_corrected = self.img_original.rotate(self.degree)
        else:
            self.img_corrected = self.img_original.rotate(- self.degree)

        self.img_corrected_array = np.array(self.img_corrected)
        print(np.amax(self.img_array))
        print(np.amax(self.img_corrected_array))



    def save_img(self):
        self.img_corrected.save('rotated.tiff')

    def change_color(self):
        self.img_corrected = self.img_corrected.convert('RGBA')
        self.img_corrected_array = np.array(self.img_corrected)
        im2 = Image.fromarray(self.img_corrected_array)
        im2.show()


imgtif = PlImage('test2.tif', 680)
imgtif.tilt_correction()
print(imgtif.upper_point)
print(imgtif.lower_point)
print(imgtif.tilt)
print(imgtif.degree)
imgtif.save_img()


