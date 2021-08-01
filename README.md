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

This project has been used to implement an autocomplete tool for Genexus code editor. See the project [here](http://lsigxextensions.sourceforge.net/prediccion.shtml).

## Install

Requires Tensorflow 2.4.1 or 2.5. See https://www.tensorflow.org/install

## Use

See [documentation](doc/README.md)

## Licensing

This repo includes a modified version of the [minGPT-TF](https://github.com/kamalkraj/minGPT-TF) 
(see [model/mingpt](model/mingpt) for original and modified version, and the minGPT-TF license)
