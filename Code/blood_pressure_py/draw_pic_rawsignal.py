import matplotlib.pyplot as plt
import h5py
import numpy as np

if __name__ == '__main__':
    file_name = r"F:\Project-342B\血压预测\BloodPressure_Prediction\data_11.05\1.mat"
    signal = h5py.File(file_name, 'r')
    signal_np = np.transpose(signal['signal'][:])
    reg = signal_np[1, :]
    plt.figure(1)
    plt.plot(reg)


