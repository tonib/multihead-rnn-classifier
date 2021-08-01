
This is an experiment to create a language model framework for programming source code autocompletion. It tries to predict
the next word that will be typed in the IDE, based on what are tye previous words typed.
It can train, export and (somewhat) serve the model.

This is NOT a model for any specific IDE and/or language, just a framework to train a generic programming language. Is
up to you tokenize and define/give the features for each token and process the predictions.

Model is trained from a set of CSV files, one for each source file module. What a module is depends on the programming
language. In Java it could be a .java file, in C it could be a .h/.c file.

Model is trained with Tensorflow 2.4.1 or 2.5, with a RNN or a GPT-like model ([minGPT-TF](https://github.com/kamalkraj/minGPT-TF)).

Model is exported as a a Tensorflow model and a Tensorflow Lite model. You can use the standard ways to serve it. Besides this,
there is a Python script to serve the model from the command line (stdin/stdout), with JSON inputs/outputs, that could be piped to your 
IDE.

This project has been used to implement an automoplete tool for the Genexus code editor. See the project [here](http://lsigxextensions.sourceforge.net/prediccion.shtml).

## Install

TODO: Pending

# Defining the language

Definitions:

## Token 
A word in a program. It's defined by one or more features. In a language model a word (or subword) usually is defined by its text.
In a strongly typed language, a variable, as example, has more information associated than its textual name: It has a type. 
So, it could be useful feed this to the model. In this code:
```c
  int i=0, j=0;
  j = my_function(i);
```
We could define three features for a word: Its token type, text and type. So, the second line could be formally defined as:

| TokenType | Text | DataType |
| --------- | ---- | -------- |
| variable | j | int |
| equal_op | none | none |
| function_call | my_function | int (function returned type) |
| parenthesis_open | none | none |
| variable | i | int |
| parenthesis_close | none | none |
| semicolon | none | none |

What features to use is up to you.

## Token context
Token context is hat you now about a token BEFORE this token is typed, and can be useful to predict predict the next token.
Define a context is optional, so it could be zero or more features.

As example, if we were defining C, and we can get information about previously declared functions, it would be interesting 
to feed what is the expected parameter type for that function. This will make easier to the model guess what will be typed
as parameter.

Also, again in C, it would be useful to know if the code is in a .h or a .c file, as predictions would be very different.
So, the previous example with context would be like this:

| TokenType | Text | DataType | CtxExpectedParmType | CtxFileType | 
| --------- | ---- | -------- | ------------------- | ----------- |
| variable | j | int | none | c_file |
| equal_op | none | none | none | c_file |
| function_call | my_function | int (here is function returned type) | none | c_file |
| parenthesis_open | none | none | none | c_file |
| variable | i | int | int | c_file |
| parenthesis_close | none | none | none | c_file |
| semicolon | none | none | none | c_file |

## data_info.json

Language is defined in a file called data_info.json. For the previous example it could be like this:

```json
{
  "ColumnDefinitions":[
    { "Name":"TokenType" , "Labels": ["variable", "equal_op","function_call", "parenthesis_open", "variable", "parenthesis_close", "semicolon"] },
    { "Name":"Text" , "Labels": ["none", "i", "j", "my_function"] },
    { "Name":"DataType" , "Labels": ["none", "int"] },
    { "Name":"CtxExpectedParmType" , "Labels": ["none", "int"] },
    { "Name":"CtxFileType" , "Labels": ["h_file", "c_file"] },
  ],
  "SequenceColumns":["TokenType", "Text", "DataType"],
  "ContextColumns":["CtxExpectedParmType", "CtxFileType"],
  "OutputColumns":["TokenType", "Text", "DataType"]
}
```

"ColumnDefinitions" is the declaration of all features, for tokens or context. Each feature is declared with a name, and the set of 
labels for that feature (only categorical features are supported).

"SequenceColumns" is the list of token features that will be used as input to the model. "ContextColumns" is the list of context features that will be used as input to the model. 

A feature can be only in "ColumnDefinitions" or "ContextColumns", not in both (it has no sense).

"OutputColumns" if the list of features that will predict the model for the next word. There can be more than one, and a feature can be in 
"SequenceColumns" and "OutputColumns" at same time. 

## Text features

In previous example we have declared "Text" feature:
```json
{ "Name":"Text" , "Labels": ["none", "i", "j", "my_function"] }
```
This will not scale, as in the train set there can be thousands of different names for variables and functions. Even more, if somebody
types a variable name that is not in the list, it cannot be used as input.
You can use a hash function to the names, or use BPE, or anything else. Hashing is simple and, if the hash range is large, it work
well in practice. If your hash function returns a number range of 0..3, you should define a label for each value. Example:

```json
{ "Name":"Text" , "Labels": ["none", "#0", "#1", "#2", "#3"] }
```

Label text here is no relevant, it could be "0", or "#0", or whatever.

## Output features and predictions

In a language model usually there is a single feature to predict: The next word (or subword) text. This is supported, as you can define a single 
output feature, with the token text. In a autocomplete function, this can be troublesome.

For the model, it will be much easier to predict the token type/data type (ex. "here will come a variable with integer type") than it's name, 
so it can be useful define multiple features to predict, and combine them to get a probability for a token as the next word. 

# Dataset 
The train dataset is a set of CSV files. There should be CSV file for each module in the source code to train. What a "module" is depends of
the programming language. In C it should be a .c / .h file, in Java a .java file, etc. Do not join multiple modules in a single CSV file, as 
this is important for the train process.

File fields separator must to be ";". It must contain a column for each feature declared in
"ColumnDefinitions" in data_info.json. The column title must to be the "Name" feature. Values must to be numeric INDICES of labels 
in the labels list declared for the feature in data_info.json (not the label itself).

CSV files can contain other columns. It's useful for debugging add columns with the label value.

Columns order in CSV is not important, but all CSV must have the same ordering.

Example:

| TokenType_debug | Text_debug | DataType_debug | CtxExpectedParmType_debug | CtxFileType_debug | TokenType | Text | DataType | CtxExpectedParmType | CtxFileType |
| --------- | ---- | -------- | ------------------- | ----------- | --------- | ---- | -------- | ------------------- | ----------- |
| variable | j | int | none | c_file | 0 | 2 | 1 | 0 | 1 |
| equal_op | none | none | none | c_file | 1 | 0 | 0 | 0 | 1 |
| function_call | my_function | int | none | c_file | 2 | 3 | 1 | 0 | 1 |

And so on. Dataset should be in it's own directory. data_info.json must to be in the directory root. CSV can be anywhere inside the directory.
Example:

```
  dataset_dir
    data_info.json
    module1.csv
    module2.csv
    subdirectory
      module3.csv
      module4.csv
```

# Defining the model

Model is configured in data_info.json:

```json
{
  ...
  "SequenceLength":32,
  "ModelType":"gpt"
}
```
"SequenceLength" is the maximum number of tokens, previous to the cursor position in the IDE, to feed to the model as input.
As greater is this number, slower and bigger will be the model.

"ModelType" defines what model type to use. There are 3 model flavours:
* "gpt": A GPT-like model (token context feeded for all timesteps)
* "exp": A RNN, with token context feeded for all timesteps.
* "rnn": A RNN, with token context ONLY for the token to predict

## Model inputs

"gpt" and "exp" model interface for inputs and outputs is the same. You feed tokens and tokens context for all timesteps, and 
you get a prediction for output features. As example, imagine we have typed this:

```c
  j = my_function( // Cursor is here
```

We have, from the module start (here a .C file), four typed tokens. But we have token context for *FIVE* tokens: four typed tokens, 
and the token to predict. Remember that token context is what we know before a token is typed. For the non typed yet token,
we know that is a parameter that expects an integer value.

So, we will feed the following input for each feature:

```javascript
{
  // Typed token features: We feed four:
  "TokenType": [ "variable", "equal_op", "function_call", "parenthesis_open" ],
  "Text": [ "j", "none", "my_function", "none" ],
  "DataType": [ "int" , "none", "int", "none" ],
  // Context token features: We feed FIVE:
  "CtxExpectedParmType": [ "none" , "none" , "none" , "none" , "int" ],
  "CtxFileType": [ "c_file" , "c_file" , "c_file" , "c_file" , "c_file" ]
}
```

"rnn" model has different inputs: It expects token context ONLY for the future token, this is, the context for the cursor
position. So, the input will be:

```javascript
{
  // Typed token features: We feed four:
  "TokenType": [ "variable", "equal_op", "function_call", "parenthesis_open" ],
  "Text": [ "j", "none", "my_function", "none" ],
  "DataType": [ "int" , "none", "int", "none" ],
  // Context token ony for the token that will be typed:
  "CtxExpectedParmType": "int",
  "CtxFileType": "c_file"
}
```

Recommended model is "gpt". 

In these examples, here we have show inputs with the labels, but the model expects, as in CSV files,
INDICES to labels in list declared in the data_info.json list, not the labels itself.

## GPT-like model configuration

This is an adaptation of the [minGPT-TF](https://github.com/kamalkraj/minGPT-TF) for multiples inputs/outputs. 
Configuration is defined in data_info.json:

```javascript
{
  ...
  "GptAttentionDropout":0.1,
  "GptEmbeddingDropout":0.1,
  "GptEmbeddingSize":128,
  "GptNHeads":2,
  "GptNLayers":2,
  "GptResidualDropout":0.1
}
```

## RNN models configuration

Configuration for RNN models is defined in data_info.json:

```javascript
{
  "CellType":"gru", // This can be "gru" or "lstm"
  "Dropout":0.1,
  "NNetworkElements":128,
}
```

## Model outputs

In all models outputs are the same: A probability for each label in output features, in same order as labels are declared 
in data_info.json. Example:

```javascript
{
  "TokenType": [ 
    0.3,  // Probability for "variable" label
    0.2,  // Probability for "equal_op" label
    ...
  ], 
  "Text": [
    0.9,  // Probability for "none"
    0.06,  // Probability for "i"
    ...
  ],
  "DataType": [ ... ]
}
```

# Training

TODO: Add configuration for training

The language, and the features to feed/predict to the model are defined in a file called "data_info.json". Each word is defined by
one or more features.

# *OLD*

# Defining the language

## Language declaration (basics)

The model is agnostic about the programming language, so, what features extract of each "word" in the language must to be defined. 
This is done in a file called "data_info.json". Lets define a subset of C. Example:
```c
  int i = 0;
  int j = i + 1;
```

Here, language keywords are "int", "=", "+", ";". Variables identifiers, "i" and "j", are not language keywords. 
How to define them? Is up to you, and there are a lot of options to choose (BPE, text hash, ...). Lets assume the easiest:
You do a dictionary with all the variable names in your code, and all your code are these two lines.

What about "0" and "1"?. Lets assume you extract them with a generic keyword of "IntegerConstant".

So, your language definition is a list of keywords: "int", "=", "+", ";", "i", "j", "IntegerConstant". The "data_info.json" file is defined like this:

```json
{
  "ColumnDefinitions":[
    { "Name":"Keyword", "Labels": [ "int", "=", "+", ";", "i", "j", "IntegerConstant" ] }
  ],
  "SequenceColumns":["Keyword"],
  "OutputColumns":["Keyword"]
}
```

"ColumnDefinitions" are the set of features of each token in language to extract.
"SequenceColumns" is the set of input features to feed to the model, and "OutputColumns" if the set of features the model
will try to predict. 

In this case each token has a single dimension, and the language is defined as a dictionary, and it will work very similar to a typical language model.

## Language declaration (multidimensional)




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
    "CellType": "gru",

    "PercentageEvaluation": 10.0,
    "LearningRate": 0.001,
    "Dropout": 0.2,

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
* "CustomEstimator": If false, the canned [RNNEstimator](https://www.tensorflow.org/api_docs/python/tf/contrib/estimator/RNNEstimator) is 
  used to create the model. If true, a custom model is created. Default value is false.
* "CellType" is the RNN cell type to use. it can be 'gru' or 'lstm'. Default value is 'gru'. If CustomEstimator = false, this parameter is ignored

Training settings:
* "PercentageEvaluation" is the percentage of the CSV files that will be used to train to the model, but just to get an accuracy evaluation.
  It's a "100%" based number.
* "LearningRate" is the learning rate for the ADAM optimizer. Default value is 0.001. If CustomEstimator = false, this parameter is ignored
* "Dropout": If > 0, a Dropout layer will be added after the RNN layer. Values is the fraction on RNN outputs that will be dropped.
  Default value is 0. If CustomEstimator = false, this parameter is ignored

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
