class DecisionTree:
  class Node:
    def __init__(self):
      self.children = {}  # Key: feature value, Value: child node
      self.feature = None # Feature used for splitting at this node
      self.label = None   # if the node is a leaf, this stores the predicted class label