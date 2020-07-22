import numpy as np
import os
import pdb


class DICOMObject:
    def __init__(self, DICOM_txt_file):
        DICOM_file = open(DICOM_txt_file, "r")

        patient_line = DICOM_file.readline()
        patient_values = patient_line.split(",")
        self.patient_name = patient_values[1].rstrip('\n')

        series_line = DICOM_file.readline()
        series_values = series_line.split(",")
        self.series = int(series_values[1])

        instance_line = DICOM_file.readline()
        instance_values = instance_line.split(",")
        self.instance = int(instance_values[1])

        num_rows_line = DICOM_file.readline()
        num_rows_values = num_rows_line.split(",")
        self.num_rows = int(num_rows_values[1])

        num_cols_line = DICOM_file.readline()
        num_cols_values = num_cols_line.split(",")
        self.num_cols = int(num_cols_values[1])

        #         print("Patient name: %s" % self.patient_name)
        #         print("Series: %d" % self.series)
        #         print("Instance: %d" % self.instance)
        #         print("Number of rows: %d" % self.num_rows)
        #         print("Number of columns: %d" % self.num_cols)

        self.data = np.zeros((self.num_rows, self.num_cols))

        for i in range(0, self.num_rows):

            data_row = DICOM_file.readline()
            data_row_values = data_row.split(",")

            for j in range(0, self.num_cols):
                self.data[i][j] = data_row_values[j]

        # pdb.set_trace()


base_folder = 'D:\\disertatie\\materiale-radu\\Medical SuperRes\\data\\3d_txt'
folders = os.listdir(base_folder)

for folder in folders:
    files = os.listdir(os.path.join(base_folder, folder))
    for file in files:
        obj = DICOMObject(os.path.join(base_folder, folder, file))
        np.save(os.path.join(base_folder, folder, file[:-4] + ".npy"), obj.data)
