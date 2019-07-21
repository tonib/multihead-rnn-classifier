
This is a simple wrapper for Tensorflow [RNNEstimator](https://www.tensorflow.org/api_docs/python/tf/contrib/estimator/RNNEstimator), to
create models to make predictions for programming source code edition.
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
    "PercentageEvaluation": 10.0,

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
* "PercentageEvaluation" is the percentage of the CSV files that will be used to train to the model, but just to get an accuracy evaluation.
  It's a "100%" based number.

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
python train.py [--datadir=DIRECTORY]
```

Where "DIRECTORY" it the directory with the CSV files. Default directory is "data".

Train can be interrupted and continued later with the same command line.

### Export production model

To export the model to use in production:

```bash
python export.py [--datadir=DIRECTORY]
```

### Production predictions

To start the model server process, run:

```bash
python model_server.py [--datadir=DIRECTORY]
```

This will run a process that will expect an standard input JSON text with the "sequence" and "context" input on a single text line.
When this input is feed, the process with write to the standard output an JSON single text line with the output prediction.

## Padding

Sequences feeded to the model for training and prediction must to have a fixed length, defined by the "SequenceLength" field in 
"data_info.json" file. Let's suppose it's 128 tokens. To predict the first
word on a .java file, (ex. "import"), you must to provide to the model a 128 tokens previous to that one as sequece. 
Obviously, they don't exist.

In this case, sequences must be padded. For training, padding elements are automatically padded with an element with all "sequence" and "context" array values set to zero. 

This is important, because in prediction time the inputs are NOT automatically padded. You should send the sequence padded up to the
sequence length. The padding elements MUST to be all "sequence" and "context" zeros.

