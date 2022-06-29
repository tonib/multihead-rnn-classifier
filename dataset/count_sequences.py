"""
    Script to count number of sequences train sequences in dataset.
    Use:
        python -m dataset.count_sequences
"""

from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition

data_definition = ModelDataDefinition.from_file()

def count_file_sequences(file_path: str) -> int:
    non_blank_line_count = 0
    with open(file_path) as infp:
        for line in infp:
            if line.strip():
                non_blank_line_count += 1
    
    # Remove csv header line
    non_blank_line_count -= 1

    # Count n. train sequences. data_definition.sequence_length is the input seq len. We take one more to predict
    train_seq_len = data_definition.sequence_length + 1
    if non_blank_line_count <= train_seq_len:
        return 1
    else:
        return non_blank_line_count - train_seq_len + 1

def count_dataset_sequences(dataset: DataDirectory) -> int:
    count = 0
    for file_path in dataset.file_paths:
        count += count_file_sequences(file_path)
    return count

train_files, eval_files = DataDirectory.get_train_and_validation_sets(data_definition)

print(f"N. train files: {len(train_files.file_paths)}, N. eval files: {len(eval_files.file_paths)}")

print(f"N. train sequences: {count_dataset_sequences(train_files)}")
print(f"N. eval sequences: {count_dataset_sequences(eval_files)}")
