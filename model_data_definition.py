import os
import json
from column_info import ColumnInfo

class ModelDataDefinition:

    def __init__(self, data_directory : str):

        self.data_directory = data_directory

        metadata_file_path = os.path.join( data_directory , 'data_info.json' )
        print("Reading data structure info from " , metadata_file_path)

        self.columns = []
        
        with open( metadata_file_path , 'r' , encoding='utf-8' )  as file:
            json_text = file.read()
            json_metadata = json.loads(json_text)

            for json_column in json_metadata['ColumnsInfo']:
                self.columns.append( ColumnInfo( json_column['Name'] , json_column['Labels'] ) )