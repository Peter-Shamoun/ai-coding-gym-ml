import numpy as np


def train_digit_classifier(X_train, y_train):
    """
    Train a digit classifier on MNIST data.

    Args:
        X_train: numpy array of shape (N, 28, 28), uint8, values 0-255
                 Grayscale images of handwritten digits
        y_train: numpy array of shape (N,), uint8, values 0-9
                 True labels

    Returns:
        predict_fn: A callable that takes X_test and returns predictions
                   predict_fn(X_test) -> numpy array of predicted labels

    Example:
        >>> predict_fn = train_digit_classifier(X_train, y_train)
        >>> predictions = predict_fn(X_test)
        >>> len(predictions) == len(X_test)
        True
    """

    # TODO: Implement your digit classifier here

    # Hint: Start by exploring the data
    # print("Training set shape:", X_train.shape)
    # print("Label range:", y_train.min(), "-", y_train.max())

    # Your training code here...

    def predict(X_test):
        """Predict digits for test images."""
        # Your prediction code here...
        pass

    return predict


# You can test your code locally by uncommenting below:
# if __name__ == "__main__":
#     # Load training data
#     X_train = np.load('mnist_train_images.npy')
#     y_train = np.load('mnist_train_labels.npy')
#
#     # Train
#     predict_fn = train_digit_classifier(X_train, y_train)
#
#     # Test on a few samples
#     test_images = X_train[:10]
#     predictions = predict_fn(test_images)
#     print("Predictions:", predictions)
#     print("True labels:", y_train[:10])
