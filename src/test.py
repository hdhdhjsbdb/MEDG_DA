import config
import numpy as np
test_x = R"E:\研一\实验\MEDG_DA\data\MAFAULDA\test_info.npy"
testnp = np.load(test_x)
print(np.unique(testnp[:,0], return_counts=True))  