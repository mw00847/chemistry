

# import the builtin time module
import time

# Grab Currrent Time Before Running the Code
start = time.time()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np


races=pd.read_csv("crayford_data.csv",delimiter=",")



x = (races[['Race_ID','Trap','Odds','BSP','Public_Estimate','Last_Run','Distance_All','Finish_All','Distance_Places_All','Races_All','Distance_Recent','Finish_Recent','Odds_Recent','Early_Recent','Races_380','Wins_380','Finish_380','Odds_380','Early_380','Grade_380','Time_380','Early_Time_380','Stay_380','Favourite','Finished','Wide_380','Dist_By']])


#x = (races[['Trap','Odds','Public_Estimate','Last_Run','Distance_All','Finish_All','Distance_Places_All','Races_All','Distance_Recent','Finish_Recent']])


y=(races['Winner'])
#print(races.iloc[5])

print(x)
print(y)

ss = preprocessing.StandardScaler()
x = pd.DataFrame(ss.fit_transform(x),columns = x.columns)

#print(X)

# split data into train and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=1)


print(X_train)
print(y_train)

print(X_test)
print(y_test)


#iris model
#from tensorflow.keras import Sequential
#model = Sequential()
#model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
#model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
#model.add(Dense(3, activation='softmax'))


#model = tf.keras.Sequential([
#    tf.keras.layers.Dense(30, activation='relu', input_shape=(27,)),
#    tf.keras.layers.Dense(1, activation='softmax')
#])


model=Sequential()
model.add(Dense(500, activation=tf.nn.relu, input_dim=(27)))
model.add(Dense(1000, activation=tf.nn.relu))
model.add(Dense(200, activation=tf.nn.relu))
model.add(Dense(1, activation='softmax'))


model.compile(optimizer=tf.keras.optimizers.Adam(5e-04),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
print(model.summary())




#dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
#train_dataset = dataset.shuffle(len(X_train)).batch(500)
#dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
#validation_dataset = dataset.shuffle(len(X_test)).batch(500)



print("Start training..\n")
#history = model.fit(train_dataset, epochs=200, validation_data=validation_dataset)
history=model.fit(X_train,y_train,epochs=50)
print("Done.")

#precision = history.history['precision']
#val_precision = history.history['val_precision']
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#epochs = range(1, len(precision) + 1)






loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.2f' % acc)

pd.DataFrame(history.history).plot(figsize=(10,6))
plt.grid(True)
plt.gca().set_ylim(0,1)

plt.show()


#plt.plot(epochs, precision, 'b', label='Training precision')
#plt.plot(epochs, val_precision, 'r', label='Validation precision')
#plt.title('Training and validation precision')
#plt.legend()
#plt.figure()

#plt.plot(epochs, loss, 'b', label='Training loss')
#plt.plot(epochs, val_loss, 'r', label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
#plt.show()



















# Grab Currrent Time After Running the Code
end = time.time()

#Subtract Start Time from The End Time
total_time = end - start
print("completed in : " + str(total_time),"s")
