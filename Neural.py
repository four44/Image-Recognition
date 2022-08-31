
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


#split the data into train and test sets and load the minst data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)  #(60000, 28, 28)  60000 rows, 28-28 pixels
print(x_test.shape)   #(10000, 28, 28)

print(x_train[0]) #=>>>  first image in the training data set
print(y_train[0]) #=>>> the image label
plt.imshow(x_train[0])
plt.show()

#Reshape the data to fit the model
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#Encode target data;
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
print(y_train_one_hot[0])


                     #      --- CNN MODEL ----
model = Sequential()
#model layers

#Concolutional layer1
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28,28,1)))

#max pooling
model.add(MaxPooling2D((2,2), padding="same"))

#convolutional layer2
model.add(Conv2D(32, kernel_size=3, activation="relu"))

#flattening layer
model.add(Flatten())

#fully connected layer with output fuction "softmax"
model.add(Dense(10, activation="softmax"))

#compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


#train the model
hist = model.fit(x_train,y_train_one_hot, validation_data=(x_test,y_test_one_hot), epochs=3)

#visualize the model accuracy
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.title("Accuracy of the Model")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Val"], loc="upper left")
plt.show()


#Show predictions as probabilities  for tho first 10  images in the data set
predictions = model.predict(x_test[:10])
print(predictions)

#print the predictions  as number labels for the first 10 images in the data set
print(np.argmax(predictions, axis=1))

#print the actual labels
print(y_test[:10])

#show the first  10 images as pictures
for i in range(0,10):
    image = x_test[i]
    image = np.array(image, dtype="float")
    pixels = image.reshape((28,28))
    plt.imshow(pixels, cmap="gray")
    plt.show()



