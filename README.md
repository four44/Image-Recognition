# Image-Recognition
Image Recognition and Processing with Artificial Neural Networks

  The project is mainly about writing a CNN (Convolutional Neural Network) which will recognize (classify) handwritten digit numbers (0,1, 2, 3, 4, 5, 6, 7, 8, 9) in an image. The training dataset (and test data set) is the MNIST data set.

  We have 2 tuples. One tuple is going to contain train data set and the other tuple is going to contain the test data set. X_train will contain the training images and y_train will contain the training labels. X_test will contain the images to test on, and y_test will contain the labels to test on.
  
  The label gives the number itself. But we need to encode it to input into neural networks. The program basicly convert the labels into a set of the numbers to input into the neural network.
  
  In this Project, multiple layers used to create a Convolutional neural network. Such as Convolutional layer , pooling layer, flattening layer, fully connected layer and with “softmax” output function.
  
  Convolutional layer :
In deep learning, CNN is a class of deep neural networks, most commonly applied to analyzing visual imagery. Convolutional layer is generally the first layer of a CNN. It calculates the element wise product of the image matrix, and a filter. A convolution operation takes place between the image and the filter and the convolved feature is generated. In this Project, 2 convolutional layers used.
 
  Kernel size is an integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions. Activasion is an activation function to use. If we don't specify anything, no activation is applied. 64 and 32 are the filters. Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
  
  Pooling layer :
Pooling layers are generally added after a convolutional layer, to reduce the dimensions of our data. A pooling window size is selected, and all the values in there will be replaced by the maximum, average or some other value. Max Pooling is the one of the most common pooling techniques.

  Flattening layer :
Flattens the input. Does not affect the batch size.

  Connected layer with softmax function:
implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True)

  Compiling the model:
“Adam” is Optimizer that implements the Adam algorithm. Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
Loss ; The purpose of loss functions is to compute the quantity that a model should seek to minimize during training. 
Categorical crossenrophy; Computes the crossentropy loss between the labels and predictions.
Accuracy metrics; Calculates how often predictions equal labels. This metric creates two local variables, total and count that are used to compute the frequency with which y_pred matches y_true. This frequency is ultimately returned as binary accuracy: an idempotent operation that simply divides total by count.
  
  Training the model:
Fit method; Trains the model for a fixed number of epochs (iterations on a dataset).
Epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. In this Project the iteration (epoch) is 3.
Validation data; Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data.

  Result:
The Program managed to classify the handwritten numbers correctly with using Convolutional Neural Network. As a result it can be seen as the output for first ten 
images in the dataset that matches with the prediction. The classification is accurate.

