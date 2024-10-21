import os
from PIL import Image
import numpy as np

class ImgData():

    def __init__(self, input_file, output_file):

        self.cdir_path = os.path.dirname(os.path.realpath(__file__))

        try:
            self.input_path = self.cdir_path + "/../data/img/" + input_file
            self.input_pil = Image.open(self.input_path)
        except:
            print(f"Image file not found at {self.input_path}.")
            exit(1)

        self.input_res = self.input_pil.size
        self.input_tensor = np.asarray(self.input_pil)[:,:,:3].reshape((self.input_res[0]*self.input_res[1], 3)).astype(float)
        self.unique_col_input_tensor, self.unique_col_input_counts = np.unique(self.input_tensor, axis=0, return_counts=True)
        self.unique_col_input_num = self.unique_col_input_tensor.shape[0]

        self.output_path = self.cdir_path + "/../data/img/" + output_file
        self.output_tensor = None
        self.output_pil = None
        self.unique_col_output_tensor = None
        self.unique_col_output_num = None

    def set_output(self, data):
        try:
            self.output_tensor = data.reshape(self.input_res[1], self.input_res[0], 3)
        except:
            print("Output image data does not match input image resolution.")
            exit(1)
        self.output_pil = Image.fromarray(self.output_tensor, "RGB")
        self.unique_col_output_tensor = np.unique(self.output_tensor, axis=0)
        self.unique_col_output_num = self.unique_col_output_tensor.shape[0]

    def save_output(self):
        self.output_pil.save(self.output_path)

    def get_pil(self, input=True):
        if input:
            return self.input_pil
        else:
            return self.output_pil
    
    def get_res(self):
        return self.input_res

    def get_tensor(self, input=True):
        if input:
            return self.input_tensor
        else:
            return self.output_tensor
    
    def get_unique_col_tensor(self, input=True):
        if input:
            return self.unique_col_input_tensor
        else:
            return self.unique_col_output_tensor
        
    def get_unique_col_counts(self):
        return self.unique_col_input_counts
    
    def get_unique_col_num(self, input=True):
        if input:
            return self.unique_col_input_num
        else:
            return self.unique_col_output_num