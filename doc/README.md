
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
Token context is what you now about a token BEFORE this token is typed, and can be useful to predict predict the next token.
Define a context is optional, so it could be zero or more features.

As example, if we were defining C, and we can get information about previously declared functions, it would be interesting 
to feed what is the expected parameters types for that function. This will make easier to the model guess what will be typed
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
You can use a hash function to encode the names, or use BPE, or anything else. Hashing is simple and, if the hash range is large, it work
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
  "NNetworkElements":128
}
```

## Model outputs

In all models outputs are the same: A probability for each label in output features, in same order as labels are declared 
in data_info.json. Example:

```javascript
{
  "TokenType": { "probabilities": [
    0.3,  // Probability for "variable" label
    0.2,  // Probability for "equal_op" label
    ...
  ]},
  "Text": { "probabilities" : [
    0.9,  // Probability for "none"
    0.06,  // Probability for "i"
    ...
  ]},
  "DataType": { "probabilities": [ ... ] }
}
```

# Training

Script to train the model is:

```bash
export TF_CPP_MIN_LOG_LEVEL=1 # Disable info messages (Optional)
# --datadir -> Directory path with data / data_info.json. Default="./data"
# --notfwarnings -> Disable warning Tensorflow C++ messages 
python train.py [--datadir DATADIR] [--notfwarnings]
```

It uses a standard Tensorflow Adam optimizer, with default parameters, except learning rate.

All model stuff will be saved in a "DATADIR/model" directory. It generates Tensorboard logs 
(DATADIR/model/tensorboard_logs) and stores a checkpoint after each epoch train end (DATADIR/model/checkpoints). 

Training can be stopped (Ctrl-C) and continued later running again the same command line from the last saved checkpoint.

Configuration for training is defined (again) in data_info.json:

```javascript
{
  ...
  "BatchSize":64,               // Train batch size. Defaults to 64
  "LearningRate": 0.001,        // Learning rate. Defaults to 0.001
  "PercentageEvaluation":10.0,  // Percentage (0-100) of module CSV files to use for model evaluation. Defaults to 15
  "LogEachBatches":100,         // Prints a log to sdtout each X number of trained batches. Just to check train performance
                                // If == 0, no log is printed. Defaults to zero.
  "MaxBatchesPerEpoch":500,     // For big datasets only. If != 0, an epoch will be limited to this number of batches.
                                // Defaults to zero
  "DatasetCache": false,        // If true, dataset will be cached in "model/cache" directory. It will give bette performance
                                // for small datasets. Cannot be set if "MaxBatchesPerEpoch" is set. Defaults to false
  "MaxEpochs":32                // Maximum number of epochs to train. Defaults to 10.
}
```

# Exporting model

After training, you must to export model with the following command line:

```bash
# --datadir -> Directory path with data / data_info.json. Default="./data"
# --checkpoint -> What checkpoint to export (numeric, 1 == first one). Defaults to the last trained checkpoint
# --notfwarnings -> Disable warning Tensorflow C++ messages 
python train.py [--datadir DATADIR] [--checkpoint CHECKPOINT] [--notfwarnings]
```

This will generate a "SavedModel" Tensorflow model in "DATADIR/model/exported_model". 

To generate a Tensorflow Lite model, export a "SavedModel" and then run this:

```bash
# --datadir -> Directory path with data / data_info.json. Default="./data"
# --notfwarnings -> Disable warning Tensorflow C++ messages 
python -m tflite.convert [--datadir DATADIR] [--notfwarnings]
```

This will generate a file "DATADIR/model/model.tflite"

# Inference

After this you have a Tensorflow / Tensorflow Lite model. You can run the model in standard ways. Remember model expects
as inputs indices to labels, not labels itself.

In data_info.json you have specified a maximum sequence length. If the tokens number in input is lower than this maximum, model
assumes that we are at a module start. If the tokens number in input is greater or equal to the maximum, it will assume we
are predicting in the middle of a module, and it will discard older tokens, up to the maximum number. Better performance 
will get if you feed up to the maximum sequence number.

Besides standard ways, we provide a simple script to run inferences from the command line. This script can be started and piped from
the IDE, and it will get inputs and return predictions in JSON. Inputs and outputs are expected to be in a single line. 
When it gets an input followed with a line return, it will run the inference and emit the prediction in JSON.

You can run it like this:

```bash
# --datadir -> Directory path with data / data_info.json. Default="./data"
# --notfwarnings -> Disable warning Tensorflow C++ messages 
python model_server.py [--datadir DATADIR] [--notfwarnings]
```

This script will check if Tensorflow Lite model ("DATADIR/model/model.tflite") exists. In this case, TF Lite model will preferred.
Otherwise, it will serve the full model. 

This script could emit an arbitrary amount of text lines. You can start to feed inputs after a "READY TO SERVE" line is generated by 
the script:

```
Data directory: data
Reading data structure info from  data/data_info.json
# Reading TF lite file from data/model/model.tflite
Loading prediction module from data/model/model.tflite
# Sample: {"Type": [], "DataType": [], "Collection": [], "Length": [], "Decimals": [], "NameHash0": [], "NameHash1": [], "NameHash2": [], "ControlType": [], "CtxParmType": [0], "CtxParmLength": [0], "CtxParmDecimals": [0], "CtxParmCollection": [0], "CtxParmAccess": [0], "CtxIsVariable": [0], "CtxObjectType": [0], "CtxPartType": [0]}
READY TO SERVE
```

Here you write the input JSON in the standard input, with a final end of line character:

```
{"Type": [], "DataType": [], "Collection": [], "Length": [], "Decimals": [], "NameHash0": [], "NameHash1": [], "NameHash2": [], "ControlType": [], "CtxParmType": [0], "CtxParmLength": [0], "CtxParmDecimals": [0], "CtxParmCollection": [0], "CtxParmAccess": [0], "CtxIsVariable": [0], "CtxObjectType": [0], "CtxPartType": [0]}
```

In standard output you will get the prediction, in JSON format:

```
{"Collection": {"probabilities": [6.503040640382096e-05, 1.768148194969399e-06, 0.9999332427978516]}, "DataType": {"probabilities": [0.9998089671134949, 2.2306089419998898e-07, 1.8228998669655994e-05, ... ]}, "Decimals": {"probabilities": [8.654110570205376e-05, 2.2994543087406782e-06, 1.5911791706457734e-06, ...]}, ... }
```

