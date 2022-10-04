from typing import List
import tensorflow as tf

class ColumnInfo:
    """ Info about a CSV column """

    def __init__(self, name: str , labels: List[str], embeddable_dimension: int, shared_labels_name: str):
        """
            name: Column name
            labels: Column labels. None if it has shared labels
            embeddable_dimension: zero if column is not embeddable. If > 0, the
            column will be embedded, with this given dimension size. None if it has shared labels
            shared_labels_name: Name for the shared labels definition. None if it has no shared labels
        """
        self.name = name
        self.labels = labels
        self.embeddable_dimension = embeddable_dimension
        self.shared_labels_name = shared_labels_name