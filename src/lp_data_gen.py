import os
import sys
from simplifier import Simplifier, Threshold_Generator
import numpy as np
from utils import ImgData

def create_data_file(img, k):

    # file management
    cdir_path = os.path.dirname(os.path.realpath(__file__))
    output_file_path = cdir_path + "/../lp/data.dat"
    file = open(output_file_path, "w")
    centers_file_path = cdir_path + "/../data/params/centers.csv"

    # fetching necessary data
    n = img.get_unique_col_num()

    try:
        centers = np.genfromtxt(centers_file_path, delimiter=',')
    except:
        print(f"Missing centers file at {centers_file_path}.")
        exit(1)
    m = centers.shape[0]

    counts = img.get_unique_col_counts()

    unique_col_tensor = img.get_unique_col_tensor()
    power = np.full(3, 2)
    dist = np.zeros((n, m), dtype=int)
    for j, center in enumerate(centers):
        dist[:, j] = np.sum((unique_col_tensor - center) ** power, axis=1)

    # header
    file.write("data;\n\n")

    # points and clusters
    file.write(f"param N := {n};\n\n")
    file.write(f"param M := {m};\n\n")
    file.write(f"param K := {k};\n\n")

    # counts
    file.write(f"param c := {' '.join([str(i+1) + ' ' + str(count) for i, count in enumerate(counts)])};\n\n")

    # distances
    file.write(f"param d : {'  '.join([str(i+1) for i in range(m)])} := \n")
    for i in range(n):
        if i < n - 1:
            file.write(f"{i+1}   {'  '.join([str(el) for el in dist[i, :]])}\n")
        else:
            file.write(f"{i+1}   {'  '.join([str(el) for el in dist[i, :]])};\n\n")

    # end
    file.write("end;\n")

    file.close()

if __name__ == "__main__":
    input_file = sys.argv[1]
    k = int(sys.argv[2])

    img = ImgData(input_file, input_file[:-4] + "output.png")

    thresh_gen = Threshold_Generator(img, 8)
    thresh_gen.run()

    simplifier = Simplifier(img, 8, thresh_gen.get_threshold())   
    simplifier.run()

    create_data_file(img, k)
 


