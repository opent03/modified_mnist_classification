# modified_mnist_classification
mnist but with more numbers in a single picture

We first train a vanilla convnet (with augmentation) on MNIST.
Then we use openCV to draw bounding boxes around modified MNIST samples, crop out the biggest instance, and
build a test set just from the cropped images.
Resizing, reshaping, and normalizing the cropped samples, we feed them into our convnet to get a prediction
