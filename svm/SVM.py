
# Import
from classifiers.Classifier import Classifier


# Support Vector Machine Model
class SVM(Classifier):
    """
    Support Vector Machine Model
    """

    # Constructor
    def __init__(self, classes):
        """
        Constructor
        :param classes:
        """
        super(SVM, self).__init__(classes)
        pass
    # end __init__

    # Add an example
    def add(self, c, x):
        """
        Add an example
        :param c:
        :param x:
        :return:
        """
        pass
    # end add

    # Train the model
    def train(self):
        """
        Train the model
        :return:
        """
        pass
    # end train

    # Predict
    def predict(self, x):
        """
        Predict
        :param x:
        :return:
        """
        pass
    # end predict

# end EchoTextClassifier
