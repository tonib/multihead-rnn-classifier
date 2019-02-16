from data_directory import DataDirectory

data_dir = DataDirectory('data')

print("Testing data set")
for row in data_dir.traverse_sequences( [None] , 3 ):
    print(row)
