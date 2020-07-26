from typing import List

class ColumnInfo:
    """ Info about a CSV column """

    def __init__(self, name: str , labels: List[str], embeddable_dimension: int):
        """
            name: Column name
            labels: Column labels
            embeddable_dimension: zero if column is not embeddable. If > 0, the
            column will be embedded, with this given dimension size
        """
        self.name = name
        #print("Column name:", name)
        self.labels = labels
        #print("Column labels:", labels)
        self.embeddable_dimension = embeddable_dimension