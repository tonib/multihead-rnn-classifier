from typing import List

class ColumnInfo:
    """ Info about a CSV column """

    def __init__(self, name: str , labels: List[str] , index: int ):
        self.name = name
        #print("Column name:", name)
        self.labels = labels
        #print("Column labels:", labels)
        self.index = index
        #print("Column index:", index)
