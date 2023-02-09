""" The NNData class provides a means of storing feature and label
data for Neural Network training.  Methods are provided to randomly
split data between test and train sets, to ensure that all data is
served during each epoch, and to allow data to be shuffled or not
between epochs.
"""
from abc import ABC, abstractmethod
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
        self._reporting_nodes = dict()
        try:
            self.load_data(features, labels)
        except (ValueError, DataMismatchError):
            pass

    def _clear_data(self):
        """ Reset features and labels, and make sure all
        indices are reset as well
        """
        self._features = None
        self._labels = None
        self.split_set()

    def load_data(self, features: list = None, labels: list = None):
        """ Load feature and label data, with some checks to ensure
        that data is valid
        """
        if features is None or labels is None:
            self._clear_data()
            return
        if len(features) != len(labels):
            self._clear_data()
            raise DataMismatchError("Label and example lists have "
                                    "different lengths")
        if len(features) > 0:
            if not (isinstance(features[0], list)
                    and isinstance(labels[0], list)):
                self._clear_data()
                raise ValueError("Label and example lists must be "
                                 "homogeneous numeric lists of lists")
        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._clear_data()
            raise ValueError("Label and example lists must be homogeneous "
                             "and numeric lists of lists")
        self.split_set()

    def split_set(self, new_train_factor=None):
        """ Split indices between training set and testing set based on
        new train factor calculation
        """
        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)
        if self._features is None or len(self._features) == 0:
            self._train_indices = []
            self._test_indices = []
            return
        num_samples = list(range(len(self._features)))
        random.shuffle(num_samples)
        num_train = round(len(num_samples) * self._train_factor)
        self._train_indices = num_samples[:num_train]
        self._test_indices = num_samples[num_train:]
        random.shuffle(self._train_indices)
        random.shuffle(self._test_indices)
        self.prime_data()

    def get_one_item(self, target_set=None):
        """ Return exactly one feature/label pair as a tuple """
        try:
            if target_set == NNData.Set.TEST:
                index = self._test_pool.popleft()
            else:
                index = self._train_pool.popleft()
            return self._features[index], self._labels[index]
        except IndexError:
            return None

    def number_of_samples(self, target_set=None):
        """ Return total number of samples"""
        if target_set is NNData.Set.TRAIN:
            return len(self._train_pool)
        elif target_set is NNData.Set.TEST:
            return len(self._test_pool)
        else:
            return len(self._features)

    def pool_is_empty(self, target_set=None):
        """ This method returns True if the target_set deque (self._train_pool
        or self._test_pool) is empty, or False otherwise.  If target_set is
        None, use the train pool.
        """
        if target_set is NNData.Set.TEST:
            return len(self._test_pool) == 0
        else:
            return len(self._train_pool) == 0

    def prime_data(self, target_set=None, order=None):
        """Load one or both deques to be used as indirect indices """
        if order is None:
            order = NNData.Order.SEQUENTIAL
        if target_set is not NNData.Set.TRAIN:
            # this means we need to prime test
            test_indices_temp = list(self._test_indices)
            if order == NNData.Order.RANDOM:
                random.shuffle(test_indices_temp)
            self._test_pool = deque(test_indices_temp)
        if target_set is not NNData.Set.TEST:
            train_indices_temp = list(self._train_indices)
            if order == NNData.Order.RANDOM:
                random.shuffle(train_indices_temp)
            self._train_pool = deque(train_indices_temp)


class LayerType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class MultiLinkNode(ABC):
    """This is an abstract base class that will be the starting point
    for our eventual FFBPNeurode class.
    """

    class Side(Enum):
        """ Use these terms to identify relationships between neurodes"""
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        self._reporting_nodes = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}
        self._reference_value = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}
        self._neighbors = {MultiLinkNode.Side.UPSTREAM: [],
                           MultiLinkNode.Side.DOWNSTREAM: []}

    def __str__(self):
        ret_str = "-->Node " + str(id(self)) + "\n"
        ret_str = ret_str + "   Input Nodes:\n"
        for key in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            ret_str = ret_str + "   " + str(id(key)) + "\n"
        ret_str = ret_str + "   Output Nodes\n"
        for key in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            ret_str = ret_str + "   " + str(id(key)) + "\n"
        return ret_str


    @abstractmethod
    def _process_new_neighbor(self, node, side):
        """ This is an abstract method that takes a node and a Side enum
        as parameters.  We will implement it in the Neurode class below.
        """
        raise NotImplementedError("This method must be implemented "
                                  "by a subclass")

    def reset_neighbors(self, nodes, side):
        """ It will reset (or set) the nodes that link into this node
        either upstream or downstream.
        """
        self._neighbors[side] = nodes.copy()
        for node in nodes:
            self._process_new_neighbor(node, side)
        self._reference_value[side] = (1 << len(nodes)) - 1
        self._reporting_nodes[side] = 0


class Neurode(MultiLinkNode):
    """ This class is inherited from and implements MultiLinkNode """

    def __init__(self, node_type, learning_rate=.05):
        super().__init__()
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = dict()

    def _process_new_neighbor(self, node, side):
        """ This method will be called when any new neighbors are added
        """
        if side == MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = random.uniform(0, 1)

    @property
    def value(self):
        return self._value

    @property
    def node_type(self):
        return self._node_type

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    def _check_in(self, node, side):
        """ This method will be called whenever the node learns that a
        neighboring node has information available.
        """
        node_index = self._neighbors[side].index(node)
        self._reporting_nodes[side] |= (1 << node_index)
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        else:
            return False

    def get_weight(self, node):
        """ The upstream node will pass in a reference to itself, and our
        current node will look up the upstream node's associated weight
        in the self._weights dictionary, and return that value
        """
        return self._weights[node]


class FFNeurode(Neurode):
    """ This class is inherited from Neurode """

    def __init__(self, my_type):
        self.my_type = my_type
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value):
        """ This should be a static method, and should return the result
        of the sigmoid function at value
        """
        return 1 / (1 + np.exp(-value))

    def _calculate_value(self):
        """ Calculate the weighted sum of the upstream nodes' values.
        Pass the result through self._sigmoid() and store the returned
        value into self._value
        """
        weighted_sum = 0
        for neighbor in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            weighted_sum += neighbor.value * self._weights[neighbor]
        self._value = self._sigmoid(weighted_sum)

    def _fire_downstream(self):
        for downstream_neighbor in \
                self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            downstream_neighbor.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        if self._check_in(node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value):
        self._value = input_value
        for downstream_neighbor in \
                self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            downstream_neighbor.data_ready_upstream(self)


def check_point_two_test():
    inodes = []
    hnodes = []
    onodes = []
    for k in range(2):
        inodes.append(FFNeurode(LayerType.INPUT))
    for k in range(2):
        hnodes.append(FFNeurode(LayerType.HIDDEN))
    onodes.append(FFNeurode(LayerType.OUTPUT))
    for node in inodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in hnodes:
        node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
        node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in onodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
    try:
        inodes[1].set_input(1)
        assert onodes[0].value == 0
    except:
        print("Error: Neurodes may be firing before receiving all input")
    inodes[0].set_input(0)

    # Since input node 0 has value of 0 and input node 1 has value of
    # one, the value of the hidden layers should be the sigmoid of the
    # weight out of input node 1.

    value_0 = (1 / (1 + np.exp(-hnodes[0]._weights[inodes[1]])))
    value_1 = (1 / (1 + np.exp(-hnodes[1]._weights[inodes[1]])))
    inter = onodes[0]._weights[hnodes[0]] * value_0 + \
            onodes[0]._weights[hnodes[1]] * value_1
    final = (1 / (1 + np.exp(-inter)))
    try:
        print(final, onodes[0].value)
        assert final == onodes[0].value
        assert 0 < final < 1
    except:
        print("Error: Calculation of neurode value may be incorrect")


if __name__ == "__main__":
    check_point_two_test()

"""
--- sample run #1 ---
0.7146970662339076 0.7146970662339076
"""
