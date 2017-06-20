
# Import
from classifiers.Classifier import Classifier


# Two Grams Statistical Language Model
class StatisticalLanguageModelTwoGrams(Classifier):
    """
    Two Grams Statistical Language Model
    """

    # Constructor
    def __init__(self, classes):
        """
        Constructor
        :param classes:
        """
        super(StatisticalLanguageModelTwoGrams, self).__init__(classes)
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
