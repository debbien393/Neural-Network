""" The NNData class provides a means of storing feature and label
data for Neural Network training.  Methods are provided to randomly
split data between test and train sets, to ensure that all data is
served during each epoch, and to allow data to be shuffled or not
between epochs.
"""
from collections import deque
from enum import Enum
import numpy as np
import random


class DataMismatchError(Exception):
    """ Label and example lists have different lengths """


class NNData:
    """ Maintain and dispense examples for use by a Neural
    Network Application """

    class Order(Enum):
        """ Indicate whether data will be shuffled for each new epoch """
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        """ Indicate which set should be accessed or manipulated """
        TRAIN = 0
        TEST = 1

    @staticmethod
    def percentage_limiter(percentage: float):
        """ Ensure that percentage is bounded between 0 and 1 """
        return min(1, max(percentage, 0))

    def __init__(self, features=None, labels=None, train_factor=.9):
        self._train_factor = NNData.percentage_limiter(train_factor)
        if features is None:
            features = []
        if labels is None:
            labels = []
        self._features = None
        self._labels = None
        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque()
        self._test_pool = deque()
        self.load_data(features, labels)

    def load_data(self, features: list = None, labels: list = None):
        """ Load feature and label data, with some checks to ensure
        that data is valid
        """
        if features is None or labels is None:
            self._features = None
            self._labels = None
            return
        if len(features) != len(labels):
            self._features = None
            self._labels = None
            self.split_set()
            raise DataMismatchError("Label and example lists have "
                                    "different lengths")
        if len(features) > 0:
            if not (isinstance(features[0], list)
                    and isinstance(labels[0], list)):
                self._features = None
                self._labels = None
                self.split_set()
                raise ValueError("Label and example lists must be "
                                 "homogeneous numeric lists of lists")
        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._features = None
            self._labels = None
            self.split_set()
            raise ValueError("Label and example lists must be homogeneous "
                             "and numeric lists of lists")
        self.split_set()

    def split_set(self, new_train_factor=None):
        """ Split indices between training set and testing set based on
        new train factor calculation
        """
        if new_train_factor is not None:
            self._train_factor = \
                NNData.percentage_limiter(new_train_factor)
        if self._features is None:
            return
        num_samples = list(range(len(self._features)))
        random.shuffle(num_samples)
        num_train = round(len(num_samples) * self._train_factor)
        self._train_indices = num_samples[:num_train]
        self._test_indices = num_samples[num_train:]
        random.shuffle(self._train_indices)
        random.shuffle(self._test_indices)
        self.prime_data()

    def prime_data(self, target_set=None, order=None):
        """Load one or both deques to be used as indirect indices """
        self._train_pool.clear()
        self._test_pool.clear()
        self._train_pool = deque(self._train_indices)
        self._test_pool = deque(self._test_indices)
        if target_set is NNData.Set.TRAIN:
            return self._train_pool
        elif target_set is NNData.Set.TEST:
            return self._test_pool
        if order is NNData.Order.RANDOM:
            random.shuffle(self._train_pool)
            random.shuffle(self._test_pool)
        if order is None or order is NNData.Order.SEQUENTIAL:
            return self._train_pool, self._test_pool

    def number_of_samples(self, target_set=None):
        """ Return total number of samples"""
        if target_set is NNData.Set.TRAIN:
            return len(self._train_pool)
        elif target_set is NNData.Set.TEST:
            return len(self._test_pool)
        else:
            return len(self._train_pool) + len(self._test_pool)

    def pool_is_empty(self, target_set=None):
        """ This method returns True if the target_set deque (self._train_pool
        or self._test_pool) is empty, or False otherwise.  If target_set is
        None, use the train pool.
        """
        if target_set == NNData.Set.TRAIN or target_set is None:
            return not bool(self._train_pool)
        elif target_set == NNData.Set.TEST:
            return not bool(self._test_pool)
        else:
            return None

    def get_one_item(self, target_set=None):
        """ Return exactly one feature/label pair as a tuple """
        if target_set == NNData.Set.TRAIN or target_set is None:
            pool = self._train_pool
        elif target_set == NNData.Set.TEST:
            pool = self._test_pool
        else:
            return None
        if not pool:
            return None
        index = pool.popleft()
        return self._features[index], self._labels[index]


def load_xor():
    """ Load the complete population of XOR examples.  Note that the
    nature of this set requires 100% to be placed in training.
    """
    xor_x = [[0, 0], [1, 0], [0, 1], [1, 1]]
    xor_y = [[0], [1], [1], [0]]
    xor_array = NNData(xor_x, xor_y, 1)


def unit_test():
    errors = False
    try:
        # Create a valid small and large dataset to be used later
        x = [[i] for i in range(10)]
        y = x
        our_data_0 = NNData(x, y)
        x = [[i] for i in range(100)]
        y = x
        our_big_data = NNData(x, y, .5)

        # Try loading lists of different sizes
        y = [[1]]
        try:
            our_bad_data = NNData()
            our_bad_data.load_data(x, y)
            raise Exception
        except DataMismatchError:
            pass
        except:
            raise Exception

        # Create a dataset that can be used to make sure the
        # features and labels are not confused
        x = [[1.0], [2.0], [3.0], [4.0]]
        y = [[.1], [.2], [.3], [.4]]
        our_data_1 = NNData(x, y, .5)

    except:
        print("There are errors that likely come from __init__ or a "
              "method called by __init__")
        errors = True

    # Test split_set to make sure the correct number of examples are in
    # each set, and that the indices do not overlap.
    try:
        our_data_0.split_set(.3)
        print(f"Train Indices:{our_data_0._train_indices}")
        print(f"Test Indices:{our_data_0._test_indices}")

        assert len(our_data_0._train_indices) == 3
        assert len(our_data_0._test_indices) == 7
        assert (list(set(our_data_0._train_indices +
                         our_data_0._test_indices))) == list(range(10))
    except:
        print("There are errors that likely come from split_set")
        errors = True

    # Make sure prime_data sets up the deques correctly, whether
    # sequential or random.
    try:
        assert our_data_0.number_of_samples(NNData.Set.TEST) == 7
        assert our_data_0.number_of_samples(NNData.Set.TRAIN) == 3
        assert our_data_0.number_of_samples() == 10
    except:
        print("Return value of number_of_samples does not match the "
              "expected value.")
    try:
        our_data_0.prime_data(order=NNData.Order.SEQUENTIAL)

        assert len(our_data_0._train_pool) == 3
        assert len(our_data_0._test_pool) == 7
        assert our_data_0._train_indices == list(our_data_0._train_pool)
        assert our_data_0._test_indices == list(our_data_0._test_pool)
        our_big_data.prime_data(order=NNData.Order.RANDOM)
        assert our_big_data._train_indices != list(our_big_data._train_pool)
        assert our_big_data._test_indices != list(our_big_data._test_pool)
    except:
        print("There are errors that likely come from prime_data")
        errors = True

    # Make sure get_one_item is returning the correct values, and
    # that pool_is_empty functions correctly.
    our_data_1.get_one_item()
    try:
        our_data_1.prime_data(order=NNData.Order.SEQUENTIAL)
        my_x_list = []
        my_y_list = []
        while not our_data_1.pool_is_empty():
            example = our_data_1.get_one_item()
            my_x_list.append(list(example[0]))
            my_y_list.append(list(example[1]))
        assert len(my_x_list) == 2
        assert my_x_list != my_y_list
        my_matched_x_list = [i[0] * 10 for i in my_y_list]
        assert my_matched_x_list == my_x_list
        while not our_data_1.pool_is_empty(our_data_1.Set.TEST):
            example = our_data_1.get_one_item(our_data_1.Set.TEST)
            my_x_list.append(list(example[0]))
            my_y_list.append(list(example[1]))
        assert my_x_list != my_y_list
        my_matched_x_list = [i[0] * 10 for i in my_y_list]
        assert my_matched_x_list == my_x_list
        assert set(i[0] for i in my_x_list) == set(i[0] for i in x)
        assert set(i[0] for i in my_y_list) == set(i[0] for i in y)
    except:
        print("There are errors that may come from prime_data, but could "
              "be from another method")
        errors = True

    # Summary
    if errors:
        print("You have one or more errors.  Please fix them before "
              "submitting")
    else:
        print("No errors were identified by the unit test.")
        print("You should still double check that your code meets spec.")
        print("You should also check that PyCharm does not identify any "
              "PEP-8 issues.")


if __name__ == "__main__":
    unit_test()

