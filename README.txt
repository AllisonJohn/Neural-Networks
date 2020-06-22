The format for the input file for the network is below. "e.txt" is an example of an input file. Only lines 2-5 and 15 must be integers.

Line 1: the error bound (training will stop when the error is less than this number)
Line 2: the maximum number of iterations in training
Line 3: the number of iterations before re-randomizing the weights
Line 4: the number of input layer nodes
Line 5: the number of nodes in the first hidden layer
Line 6: the number of nodes in the second hidden layer
Line 7: the number of output layer nodes
Line 8: the starting learning factor value
Line 9: the learning factor adjustment the lambda will be multiplied or divided by in training, can be set to 1.0 if adaptive learning is not wanted.
Line 10: the maximum value that the learning factor is allowed to reach
Line 11: the minimum value for the random value of a weight
Line 12: the maximum value for the random value of a weight
Line 13: (0 or 1) 0 if weights should be randomized, 1 if they should be set manually by the user. Even if the user enters the weights manually to start, the weights could be randomized if the network doesn't converge before the number of iterations specified in Line 3.
Line 14: (0 or 1) 0 if training should use weight rollback, 1 if no weight rollback
Line 15: the number of training sets
Line 15: "Data:" if entering all data into this file, "File:" if providing a file name for a large amount of data (as from a single image), and "Hand Data:" if for training on five hand images
Line 16-end: the name of the data file or all the training data with spaces between, for example:
I1 I2
O1 O2 O3
I1 I2
O1 O2 O3
.
.
.
