import numpy as np
import pandas as pd

file = pd.read_csv( '../data/PAlcTasRec.csv' , sep=';')
#print(file)

#print( file['wordType'][0:10] )


# print( file['wordType'][0:10] )

# start_idx = -3
# idx = 4
# print( file.loc[start_idx:idx, 'wordType'].to_numpy() )

# print( file.at[ 3 , 'wordType'] )

# values = file.loc[0:4, 'wordType'].to_numpy()
# values = np.pad( values , (2,0) , 'constant' )
# print( values )

#print( file.index[ file['trainable'] == 1].tolist() )
#print( file.index[ file['trainable'] == 1] )

#print(file.shape[0] )

print( len(file[ file['trainable'] == 1]) )
