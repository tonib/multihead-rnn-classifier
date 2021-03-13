import tensorflow as tf
from tensorflow.data import Dataset


sequence_length = 8 # This length includes the EOS symbol, so the real length is -1
sequence_feature_names = [ "wordType", "keywordIdx", "isCollection" ]
context_feature_names = [ "partType" ]
output_features_names = [ "outputTypeIdx" , "isCollection" ]
eos = -1

# Unique column names to use
feature_column_names = sequence_feature_names + context_feature_names + output_features_names
feature_column_names = list(dict.fromkeys(feature_column_names))

# CSV files to train
file_paths = ["data/TCalles.csv", "data/TArtAlEAcc.csv", "data/AlbaranNet.csv"]
csv_separator = ";"

# Tricky things: To get right sequences we must separate CSV contents, and seems not supporte by TF CSV hight level helpers
# Soooo, guess the the CSV structure:
with open(file_paths[0]) as f:
    first_line = f.readline()
    csv_column_names = first_line.split(csv_separator)
    csv_column_names_to_indices = { name:index for index, name in enumerate(csv_column_names) }
    # select_cols parm in tf.data.experimental.CsvDataset must be ordered by column index. So, reorder feature_column_names
    # to follow that order
    feature_column_names.sort( key=lambda x: csv_column_names_to_indices[x] )
    feature_column_indices = [ csv_column_names_to_indices[feature_column_name] for feature_column_name in feature_column_names ]

# Column types: All int32
#default_csv_values = tf.zeros( len(feature_column_names) , tf.int32 )
default_csv_values = [ tf.int32 ] * len(feature_column_names)
print(default_csv_values)

@tf.function
def load_csv(file_path):
    csv_ds = tf.data.experimental.CsvDataset(
        file_path, default_csv_values, 
        header=True,
        field_delim=";",
        use_quote_delim=False,
        select_cols=feature_column_indices
    )
    full_csv_data = tf.data.experimental.get_single_element( csv_ds.batch(1000000 ) )

    full_csv_dict = {}
    for feature_column_name, csv_column_values in zip(feature_column_names, full_csv_data):
        full_csv_dict[feature_column_name] = csv_column_values
    return full_csv_dict

ds = Dataset.list_files(file_paths)
ds = ds.map(load_csv)

for example in ds.take(50):
    print(example)