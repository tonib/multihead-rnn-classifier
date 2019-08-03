
This is a RNN classifer, to create models to make predictions for programming source code edition.
It can:

* Generate a prediction model for the next word  from a set of CSV files
* Export a production model
* Serve the model: It can create a process that read sequences from the standard input, and return the prediction, both in JSON format.
  Other process can launch this model server process, and, with a pipe, write sequences and get predictions


This project has been used to implement a predictor for the Genexus code editor. See the project [here](http://lsigxextensions.sourceforge.net/prediccion.shtml). It could be used to make predictions in other programming languages, or for other uses.

In this documentation I will use the term "token" for a "word" on a source code, with this mean: If a code line is "if(function(variable) > 0)", the tokens are "if", "(", "function", "(", "variable" , ">"...

## Model inputs and outputs

As is defined by the RNNEstimator, the inputs are:
* The "sequence": A list of tokens previous to the cursor position, up to a maximum number. The information to send to the model for each
  token in the sequence is an array. As example, it could be an array [ TOKENTYPE, VARIABLETYPE ], where TOKENTYPE could be "if", "+",
  "variable", ... , and VARIABLETYPE "none", "char", "List[int]",...
* The "context" is an array with information about the current cursor edition position. It could be, as example, if the cursor is on a call
  ( "object.method( |" ), the expected parameter type on that position, and the file type we are editing (ex. ".java", ".cs"). This is, an
  array [ PARMTYPE , FILETYPE ]

The output is an array with information about the token that will be written. It could be, as example [ TOKENTYPE, VARIABLETYPE ].

Only integer values are supported for array values. These integers could be indices to other array with the labels. As example, 0 could mean
"if", 1 = "+",... Or they could be indices to a values ranges. If you need to code a integer value, you can define a int value for each range. Ex. 0="number between 0 and 10", 1="number between 11 and 100",... This is up to you.

## Input data set

The expected input data set are a set of CSV files. Each CSV should correspond with a source code file (ex. a ".java" file). Each row of the
CSV should contain information of each code token on that file for "sequence", "context" and "output".

The CSV should always contain a first row with column titles.

In the example, the CSV should contain columns for TOKENTYPE, VARIABLETYPE, PARMTYPE, FILETYPE. Note that TOKENTYPE and VARIABLETYPE will
be used both for input and output.

The CSV can contain other columns not referenced by the the model. They will not be used to train the model, and they can they could contain, 
as example, debug information.

## Model definition

The model is defined on a file called "data_info.json":

```json
    {
    "ColumnDefinitions":[
        {
            "Labels":["Padding","DecimalConstant","if","VARIABLE","(" , "etc..." ],
            "Name":"TOKENTYPE"
        },
        {
            "Labels":["char", "int", "long", "etc..."],
            "Name":"VARIABLETYPE"
        },
        {
            "Labels":["char", "int", "long", "etc..."],
            "Name":"PARMTYPE"
        },
        {
            "Labels":[".cs",".java"],
            "Name":"FILETYPE"
        },
    ],

    "SequenceColumns":[ "TOKENTYPE" , "VARIABLETYPE" ],
    "OutputColumns": [ "TOKENTYPE" , "VARIABLETYPE" ],
    "ContextColumns": [ "PARMTYPE" , "FILETYPE" ],
    "TrainableColumn":"trainable", 
    "NNetworkElements" : 64,
    "SequenceLength":128,
    "CustomEstimator": true,

    "PercentageEvaluation": 10.0,
    "LearningRate": 0.001,

    "MaxEpochs":32,
    "MaxTrainSeconds":18000,
    "MinLossPercentage":0.1
    
}
```

Definitions:
* "ColumnDefinitions" is an array with the definition of each column on CSV files used by the model. Each array item is an object with
  a "Name" field, which is the CSV column name, and a "Labels" field with the descriptions for each integer value on the CSV column.
  Ex. if the "TOKENTYPE" CSV column contains a 0, it means "Padding", 1 = "DecimalConstant", etc.
* "SequenceColumns" are column names that will be feed as "sequence".
* "OutputColumns" are column names that will be feed as "output".
* "ContextColumns" are column names that will be feed as "context".
* "NNetworkElements" is the number of cells on the "recurrent" layer.
* "SequenceLength" is the number of tokens that are feeded to the model for prediction/training.
* If false, the canned [RNNEstimator](https://www.tensorflow.org/api_docs/python/tf/contrib/estimator/RNNEstimator) is used create the
  model. If true, a custom model is created.

Training settings:
* "PercentageEvaluation" is the percentage of the CSV files that will be used to train to the model, but just to get an accuracy evaluation.
  It's a "100%" based number.
* "LearningRate" is the learning rate for the ADAM optimizer. Default value is 0.001

Other numbers are used to define when to stop train the model:
* "MaxEpochs" is the number of epochs to train. Zero means train indefinitely.
* "MaxTrainSeconds" is the number of seconds to train. Zero means train indefinitely. This number will be checked after ending a epoch train,
  so an entire epoch will not be interrupted.
* "MinLossPercentage" is a definition to stop training when the loss stopped decreasing. When an epoch has been trained, the loss decrease 
  respect the previous to the last trained epoch is evaluated. If the loss decrease is lower to this value, the train will be stopped.
  Its a "100%" based number. Zero means train indefinitely. It can be a negative value.

Model uses "GRU" cells as RNN layer.

## Model train

Generate your CSV files and your "data_info.json", in a same directory. 


Then train your model running:

```bash
python train.py [--datadir=DIRECTORY] [--notfwarnings]
```

Where "DIRECTORY" it the directory with the CSV files. Default directory is "data". "--notfwarnings" disables the buch of TF deprecation
warnings.

Train can be interrupted and continued later with the same command line.

### Export production model

To export the model to use in production:

```bash
python export.py [--datadir=DIRECTORY] [--notfwarnings]
```

### Production predictions

To start the model server process, run:

```bash
python model_server.py [--datadir=DIRECTORY] [--notfwarnings]
```

This will run a process that will expect an standard input JSON text with the "sequence" and "context" input on a single text line.
When this input is feed, the process with write to the standard output an JSON single text line with the output prediction.

Example:
```
(venv) foo@bar:~/multihead-rnn-classifier$ python model_server.py --datadir=data --notfwarnings
Data directory: data
TF warning messages disabled
Reading data structure info from  data/data_info.json
# Reading latest exported model
Using export from data/exportedmodels/1564821495
2019-08-03 10:58:49.074870: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-08-03 10:58:49.079799: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3397920000 Hz
2019-08-03 10:58:49.080133: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x4a00580 executing computations on platform Host. Devices:
2019-08-03 10:58:49.080159: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
2019-08-03 10:58:49.484199: W tensorflow/compiler/jit/mark_for_compilation_pass.cc:1412] (One-time warning): Not using XLA:CPU for cluster because envvar TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want XLA:CPU, either set that envvar, or use experimental_jit_scope to enable XLA:CPU.  To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a proper command-line flag, not via TF_XLA_FLAGS) or set the envvar XLA_FLAGS=--xla_hlo_profile.
# Sample: {"wordType": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "keywordIdx": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "kbObjectTypeIdx": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "dataTypeIdx": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "isCollection": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "lengthBucket": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "decimalsBucket": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "textHash0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "textHash1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "textHash2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "textHash3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "controlType": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "ctxParmType": 0, "ctxParmLength": 0, "ctxParmDecimals": 0, "ctxParmCollection": 0, "ctxParmAccess": 0, "ctxIsVariable": 0, "objectType": 0, "partType": 0}
READY TO SERVE
```

Here you write the input JSON in the standard input, with a final end of line character. Example:
```
{"wordType": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "keywordIdx": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "kbObjectTypeIdx": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "dataTypeIdx": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "isCollection": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "lengthBucket": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "decimalsBucket": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "textHash0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "textHash1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "textHash2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "textHash3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "controlType": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "ctxParmType": 0, "ctxParmLength": 0, "ctxParmDecimals": 0, "ctxParmCollection": 0, "ctxParmAccess": 0, "ctxIsVariable": 0, "objectType": 0, "partType": 0}
```

Then the prediction is written to the standard output, with a final end of line character:
```
{"isCollection": {"class_prediction": 2, "probabilities": [0.0002461548720020801, 0.0005606021732091904, 0.9991932511329651]}, "lengthBucket": {"class_prediction": 18, "probabilities": [0.0010655540972948074, 6.278051841945853e-06, 2.040944127656985e-05, 2.1956082491669804e-05, 5.242620318313129e-05, 5.138143023941666e-06, 1.9032989939660183e-06, 9.466194569540676e-06, 5.238441644905834e-07, 2.4164110072888434e-05, 1.5395822629216127e-05, 7.907227882242296e-06, 1.785033077794651e-06, 5.4153206292539835e-05, 3.957937587983906e-05, 1.7479058442404494e-05, 4.2499825212871656e-05, 3.9709866541670635e-06, 0.9986094236373901]}, "decimalsBucket": {"class_prediction": 10, "probabilities": [0.00022998398344498128, 0.0006413692608475685, 2.260218707306194e-06, 6.701557140331715e-05, 0.0001081798764062114, 7.402428309433162e-05, 0.00013427966041490436, 0.00014097776147536933, 5.150819106347626e-06, 0.00044264504685997963, 0.9981541037559509]}, "outputTypeIdx": {"class_prediction": 0, "probabilities": [0.8450528979301453, 0.056459758430719376, 0.0007242700085043907,...] } }
```


## Padding

Sequences feeded to the model for training and prediction must to have a fixed length, defined by the "SequenceLength" field in 
"data_info.json" file. Let's suppose it's 128 tokens. To predict the first
word on a .java file, (ex. "import"), you must to provide to the model a 128 tokens previous to that one as sequece. 
Obviously, they don't exist.

In this case, sequences must be padded. For training, padding elements are automatically padded with an element with all "sequence" and "context" array values set to zero. 

This is important, because in prediction time the inputs are NOT automatically padded. You should send the sequence padded up to the
sequence length. The padding elements MUST to be all "sequence" and "context" zeros.

### Licensing

The function custom_rnn_estimator.py > CustomRnnEstimator._concatenate_context_input contains code of the Tensorflow
source code. Tensorflow is under [Apache license](http://www.apache.org/licenses/LICENSE-2.0).
