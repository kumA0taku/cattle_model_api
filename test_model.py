# import tensorflow as tf
# from tensorflow import keras
# import pandas as pd

# new_model = tf.keras.models.load_model('c2_N35_cell2.h5', compile=False)
# new_model.compile()
# new_model.summary()

# dataframes = []
# df = pd.read_csv("test_New.csv",  on_bad_lines='skip')
# dataframes.append(df)
# r_sort = df.sort_values(by='_time')

# r = pd.concat(dataframes, ignore_index=True) # รวมตาราง
# r_sort = r.sort_values(by='_time')
# r_sort = r_sort.replace('Stationary-Or-Eating', 0)
# r_sort = r_sort.replace('sleeping', 1)
# r_sort
# r_sort = r_sort.drop(columns=['_time'])
# y  = r_sort['behavior']
# X = r_sort.drop(columns=['behavior'])
# y.to_numpy()
# X = X.to_numpy()
# y = y.to_numpy()

# y = y.reshape((-1, 1))
# import numpy as np

# # สร้างฟังก์ชันสำหรับแปลงข้อมูลให้เหมาะสำหรับ LSTM
# def create_dataset(dataset_input, dataset_result, n_steps):
#     dataX, dataY = [], []
#     for i in range(len(dataset_input) - n_steps + 1):
#         a = dataset_input[i:(i + n_steps), :]
#         dataX.append(a)
#         dataY.append(dataset_result[i + n_steps - 1, :])
#     return np.array(dataX), np.array(dataY)

# # แปลงข้อมูลเป็นรูปแบบที่เหมาะสำหรับ LSTM
# dataX, dataY = create_dataset(X, y, 35)
# dataX, dataY

# class_name =["walking","stationary","sleep"]
# predictions = new_model.predict(dataX[2230:2251])
# prediction_classes =  np.argmax(predictions, axis=1)
# prediction_classes