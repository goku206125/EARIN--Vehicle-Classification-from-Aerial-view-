# EARIN Project


##EARIN Project Preliminary Documentation
#Aayush Gupta
#Description of Project

In this project, a data set consisting of aerial images of ten different vehicles will be used
to train a neural network to identify vehicles. A second (untested and unlabeled) data set will
then be run through the neural network to see how well it classifies vehicles in pictures it has
not seen yet.
The two main difficulties that will be faced are the fact that the training set is heavily
imbalanced and that, while the validation set is balanced, it is unlabeled. This will not only make
the model more difficult to reliably predict, but it will also make it much more difficult to
quantitatively analyze the effectiveness of the neural network’s predictions of the validation set.
This project will make use of a convolution neural network along with the mini batch
gradient descent method to optimize the weights and biases of each layer. After each epoch, the
validation set will be run through the neural network to see how much the outputs have
changed since the previous epoch. The program will stop optimizing once the validation set’s
outputs have converged or after a set number of epochs have passed, whichever comes first. At
that point, relevant information about the final states of both the training set and validation set
will be printed.

#Description of Data Set
The data set consists of ten sets of black and white 32 by 32 pixel images of different
vehicles. There is a significant difference in the number of images between the different sets.
Half of the vehicles have fewer than 1000 images, whereas Sedans have 234,209 images.
Furthermore, most of the vehicles that have fewer images have rather insignificant variations in
their images. In fact, in many cases, there are only three or four different vehicles pictured in
the entire set with only slight variations in the images. The fact that that many of the smaller
data sets have unique images numbering in the single digits will make creating a reliable
predictive model significantly more challenging.
Below are 38 images from the data set showing flatbed trucks with trailers. While the
data set consists of 633 different images, all 633 images are actually slight variations of the
same two vehicles.
With such tiny variations in many of the data sets, any model created will be prone to
significant overfitting of the training set. The model is likely find specific patterns that very
strongly correlate a set of images of some vehicle type in the training set and assume that all
vehicles of that type must always have those specific patterns. As an example, if all images of
busses were of the same bus parked at the same 45 degree angle, the AI model might conclude
that all busses are long rectangles occupying a 45 degree angle in the image. Of course, the AI
technically wouldn’t be wrong—every single image of a bus it had seen up until that point
satisfied that description. Even the perfect AI model is only as good as the training set that was
used to develop it.

A potential solution to this problem is to artificially expand the training sets of the most
poorly represented classes. This can be done by taking the existing images and altering them in
one or more ways. Some common image manipulations are rotating, rescaling, changing
brightness, and adding random noise. These changes not only make the gradient descent
method more effective at minimization (as a result of the larger data set), but they can also
reduce the chances that improper correlations are made. In this case, the goal is to ensure that
predictions are based on the shape of the vehicle and not the orientation, background, or size (a
picture can be taken close or far away).
#General Algorithms to be Used

The neural network will consist of an input layer (a matrix of the pixel values of one
image), some number of hidden layers (of some width), and an output layer (the neural
network’s prediction). Apart from the input layer, each layer will have a series of weights and
biases as well as an activation function, which will all contribute to the values of the final
output. Each layer (apart from the input) performs a convolution and, optionally, max pooling.
The output layer will also have a cost function that will be used to quantify how well the current
weights and biases fit the data.
Each hidden layer will have one or more nodes, a filter size (for the convolution), a stride
length, an activation function, and possibly max pooling (which will have its own pool size and
stride length). Each node will have a bias and set of weights arranged in an n by n matrix, where
n is the filter size. Initially, the weights and biases are randomized from -10 to 10.
Generally, the raw output of each hidden node, which is equal to the inputs multiplied
by the node weights with the bias added at the end, will be passed through an activation
function, which modifies, filters, and/or normalizes the output. Some activation functions that
will be tested are ReLU, the sigmoid function, and softmax. The former two will generally be
used in hidden layers. Softmax, which normalizes the outputs so that their values are equal to
the probabilities of them being the correct prediction for the given input, will be used for the
output layer.

When a convolution is performed, the n by n matrix (the one with the weights) travels
along the input matrix performing a dot product with the values in the input matrix that it
overlaps with. After the dot product is performed, the bias is added, and the number is passed
into the activation function. The result of the activation function is saved in a new matrix. The
weights matrix moves along the input matrix in steps according to the stride length until it has
traversed the entire matrix. If there are multiple input matrices, dot products are performed for
each matrix with its associated weights matrix, and the results of all the dot products are
summed.

Afterwards, if max pooling is present, a second matrix (with dimensions equal to the
pooling size) traverses along the new matrix in steps according to the stride length. For every
step that the pooling matrix takes, the highest value among the values in the overlapping area is
taken and placed into the output (for that particular node) matrix.
Both the convolution and max pooling reduce the matrix size. In the case of convolution,
patterns within the input image (represented by the weights) are being searched for, and the
convolution returns the areas where those patterns can be found. This, more or less, focuses
the matrix on points of interest (as opposed to the whole image), reducing its size. The max
pooling reduces the resolution of the matrix, while preserving the most relevant pieces of
information. This should reduce the complexity (and computation time) of subsequent
calculations without significantly affecting the final outcome.
The convolution on the output layer is always performed with a filter size equal to the
dimensions of the incoming matrices. As a result, the output of the convolution for each output
node is a single value (which is a 1 by 1 matrix).
Since the training set is labeled, the true identities of the outputted values are known
and can be compared against the predicted values. The output layer will have a cost function
associated with it that will be used to numerically determine the effectiveness of the current
weights and biases values at correctly predicting subsequent data points. The two cost functions
that will be tested will be the mean squared error and cross entropy.
The mini batch gradient descent method will be used to optimize the values of the
weights and biases. This optimization will be achieved by finding the values of weights and
biases that minimize the cost function. A key feature of the mini batch gradient descent method
is that it does not wait for the entire data set to be passed through the network between
optimization steps. Instead, it reoptimizes after just a few data points (relative to the overall
data set) have passed through. While this does increase the noise of the optimization, it greatly
reduces the computation time and makes it less likely that the optimization will get stuck in a
local minimum.

Some additional steps will be taken to further speed up the optimization. In order to
avoid having to recalculate the weights and biases for every data point, the back propagation
algorithm will be applied at the start of every new batch. Additionally, the gradient descent
method will be enhanced by adding a momentum hyperparameter, which will allow it to be
influenced by previous gradients (and not just the current gradient). By having a momentum,
the gradient descent method should be less prone to being slowed down by large oscillations
and sudden flat regions (which have a small gradient).
Since the training set is heavily imbalanced, even a batch size of 100 will never have at
least one member of every class, assuming every batch contains an equal proportion of each
class relative to the total class samples. This means that each mini batch will optimize only some
of the classes, potentially resulting in a huge discrepancy in the qualities of the fit. Furthermore,
even when the less numerous classes are represented, individually they only represent, at most,
7 percent of the mini batch. If the cost function is consistently low for the class with the most
samples, but very high for some of the other classes, the overall cost will remain relatively low,
which will make it appear to the neural network as if it’s doing well for all the classes. As a
result, it may not update its weights and biases correctly to improve the fit for the poorly
performing classes.

The first solution will involve artificially increasing the representation of classes with
fewer samples within each batch. While this will result in some data sets being cycled through
faster than others, it will ensure that each class contributes to every optimization step. However,
even with increased representation, the classes with smaller data sets will still be greatly
outnumbered by those that have larger data sets.
The second solution will address this issue by applying weights to the costs of the less
represented classes so that, for each batch, the impact of each class’s costs on the total cost is
equal to 10 percent. By doing so, whenever a class performs poorly, regardless of the number of
samples it has, the neural network will much more readily adjust its weights and biases to fix
the issue.

#General Plan of Tests/Experiments

Preliminary testing will involve optimizing the gradient descent and the neural network
to best fit the training data set. A number of different parameters will be analyzed to determine
which values result in the lowest cost while maintaining a reasonable computation time.
Once a favorable set of parameters is found, the training set will temporarily be split to
create a test validation set. The test validation set will have the same number of samples and
distribution as the real validation set, and its samples will be chosen randomly. The neural
network will then use the new training set to try and correctly predict the samples in the test
validation set (which were removed from the training set).
After the neural network has made its predictions, they will be analyzed using the
confusion matrix, which shows the number of true positives, true negatives, false positives, and
false negatives for every class. Two matrices will be created and populated. The first will be filled
in with the neural network’s best guesses for each sample, while the second will be filled in
using the probabilities that the neural network estimated for each sample. The most relevant
pieces of information that the two confusion matrices will show will be the number of mistakes
the neural network made and the number of mistakes it thinks it made.
If the neural network is found to have made too many errors on the test set, the first
step will be repeated until it improves. Otherwise, the second step will be repeated, this time
with a completely new test validation set (again, randomly selected). The same analysis will be
performed to see how consistent the results are with different data sets.
Once the behavior of the neural network is reasonably well understood, the training data
set will be returned to its original state, and the true validation set will be tested. Two sets of
values will be recorded for the validation set’s predictions. The first will be the total number of
predictions made of each class, and the second will be a confusion matrix filled in with the
probabilities that the neural network estimated for each sample. Since the validation set is
unlabeled, a confusion matrix comparing the predicted values with the actual values will not be
possible to create.

First, the total number of predictions made for each class will be compared with the
actual number of samples of each class (77), and a total offset from all classes (total difference
between the predicted and actual number of samples from each class) will be calculated. This
number will be compared against a table showing the average total offset per number of errors
to obtain a rough estimate of the accuracy of the predictions.
Then, using what was learned about the two confusion matrices from the test validation
set, the true validation set’s (sum of probabilities) confusion matrix will be analyzed. The goal
will be to try and find any patterns or similarities that indicate obvious false positives, false
negatives, true positives, or true negatives. If enough identifiable errors are found, a different
set of neural network parameters can be tested to see if the predictions improve.
The third method of analysis of the unlabeled validation set will be through the use of
tdistributed stochastic neighbor embedding on the classification layer. This will enable the
classification layer to be visualized in two dimensions, which can be used to show the general
locations of the validation set samples relative to the training data set. The locations of the
validation set samples relative to their prediction groups can potentially reveal useful
information or even trends about correct and incorrect predictions. As an example, samples that
are deep inside of their predicted group are much more likely to be true positives than those
near the edges.

Statistically, the lower the predictions’ total offset from all classes, the lower the overall
error of the predictions. As such, a reliable goal will be to get that number as low as possible
without worsening the fit of the training set. and, ideally, without worsening the error of a
subsequent test validation set.
Methods of Result Visualization
A series of graphs and tables will be used to show the results obtained. While optimizing
the gradient descent and neural network parameters, the plots of the loss value, accuracy on
train set, and accuracy on validation set (test set only) will be printed. Additionally, the total
computation time will also be shown. These will be used to quickly analyze the effectiveness of
every set of parameters tested.

Additionally, the TSNE method will be used to visualize the different samples as they are
grouped in the classification layer. This method flattens a higher dimension layer into two
dimensions while preserving the grouping and separation of groups of the original layer.
When the test validation sets are run, two confusion matrices will be displayed for each
set. The first will contain the predicted values, and the second will contain the sum of all
probabilities estimated by the neural network. In both cases, relevant errors (for each class and
a grand total) will be calculated and displayed.

For the true validation set, only one confusion matrix will be printed (sum of
probabilities). The relevant errors (for each class and a grand total) will be calculated and
displayed. Additionally, the total number of predictions of each class will also be printed, as well
as the total class offsets (total difference between the predicted and actual number of samples
from each class).

Finally, a simulation will be run, where elements in a set of arrays representing correct
and incorrect predictions will be moved randomly one at a time from correct to incorrect
positions. The simulation will be run at least 1000 times simultaneously and calculate the
average total class offsets (total difference between the predicted and actual number of
samples from each class) for each number of incorrect predictions made. The standard
deviation for each value will also be found. 



