# Generation of artificial knowledge graphs

Implementation of synthesis model from the paper "Synthesizing Knowledge Graphs for Link and Type Prediction Benchmarking" submitted to ESWC2017.


## Knowledge graph models

The six models described in the paper are supported:

- M1: Includes distribution of types over entities plus joint distribuiton of relations, subject and object types
- M2: M1 plus non-reflexiveness, functionality and inverse functionality of relations
- M3: M2 plus horn rules
- EM*i*: M*i* plus bias to selection of entities

## Setup
- Install requirements:
```
pip install -r requirements.txt
```
- If you use conda on macOS you also need to install the following and use `pythonw` instead of `python` to execute the scripts. (This is needed since matplotlib needs a python framework build as dependency.)
```
conda install python.app
```
- The knowledge graph model M3 requires a text file containing horn rules learned with
[`AMIE`](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/amie/).

## Usage

For this example we use a small knowledge base from the Semantic Web dog food about the conference
[`ESWC2015`](http://data.dws.informatik.uni-mannheim.de/hmctp/kbgen/eswc2015.n3), whose horn rules are available [`here`](http://data.dws.informatik.uni-mannheim.de/hmctp/kbgen/eswc2015-AmieRules.txt).

- First load the knowledge base dump into a tensor:

 ```
 python load_tensor.py eswc2015.n3
 ```

 this will create the file ```eswc2015.npz``` with a tensor representation of the knowledge base

- Then Learn the models:

 ```
 python learn_model.py eswc2015.ext -m M1
 python learn_model.py eswc2015.ext -m M2
 python learn_model.py eswc2015.ext -m M3 -r eswc2015-AmieRules.txt
 python learn_model.py eswc2015.ext -m e -sm M1 M2 M3
 ```

 The commands need to be executed in the order above because one model is an extension of the other.
 The commands will create generate the models and save them in the pickle files:

 - ```eswc2015-M1.pkl```
 - ```eswc2015-M2.pkl```
 - ```eswc2015-M3.pkl```
 - ```eswc2015-eM1.pkl```, ```eswc2015-eM2.pkl``` and ```eswc2015-eM3.pkl```

- From the learned models the knowledge base can be synthesized

 ```
 python synthesize.py eswc2015-M1.pkl eswc2015-replica-M1.n3 -size 0.1
 ```

 This will synthesize a replica of the dataset with 10% of the original size and dump it into ```eswc2015-replica-M1.n3```

All commands together:
```
python load_tensor.py eswc2015.n3
python learn_model.py eswc2015.npz -m M1
python learn_model.py eswc2015.npz -m M2
python learn_model.py eswc2015.npz -m M3 -r eswc2015-AmieRules.txt
python learn_model.py eswc2015.npz -m e -sm M1 M2 M3
python synthesize.py eswc2015-M3.pkl eswc2015-replica-M3.n3 --size 0.1
```

And for macOS users:
```
pythonw load_tensor.py eswc2015.n3
pythonw learn_model.py eswc2015.npz -m M1
pythonw learn_model.py eswc2015.npz -m M2
pythonw learn_model.py eswc2015.npz -m M3 -r eswc2015-AmieRules.txt
pythonw learn_model.py eswc2015.npz -m e -sm M1 M2 M3
pythonw synthesize.py eswc2015-M3.pkl eswc2015-replica-M3.n3 --size 0.1
```
