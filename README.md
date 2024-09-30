# EvidenceMap

The EvidenceMap project aims to facilitate the ingestion of PubMed articles, specifically Randomized Control Trials (RCTs). It parses out essential data from these articles and builds a PICO map visualizing the RCT outcomes. The project consists of several modules under development. Each has its specific function towards achieving the overall objective of the project.

For detailed usage of the modules including the inputs of the functions, please see the individual README files in the module's directory.

## Installation
EvidenceMap needs a working SOLR instance to work.  Prebuilt core's ready to be loaded into SOLR are available in the link to the pre-trained models below.  Follow the instructions on the INSTALLATION file found in that link.

You may use the provided models in the link below, or use the preset defaults in the pipeline to have the application rebuild/retrain the models.  When a model is not found in Negation detection and Sentence Classification, they are automatically created during the initial loading of the pipeline.  Models will need to be explicitly trained.  Follow the READMEs on the specific models section and modules for more details instructions on how to do so.

## Usage
The easiest way to use this is use the utility scripts provided in the root.

### Scripts
Refer to the help doc for each script for usage.
- acquire_pubmed.py: Acquire PubMed articles from the official PubMed database, parse them, and upload the JSON documents to the SOLR database.
- webapp.py: Starts the website, no args needed.
- ValidateEnvironment.py: Quickly establishes the environment capabilities for handling TensorFlow and PyTorch models.
- TrainSpert.py: A training sandbox for training a base model in Models.

### Defining the configuration
EvidenceConfig.py allows you to change the passable configuration values for any component of the pipeline for every driver that exists.  A pre-made EvidenceConfig.py is supplied.  You can refer to it to understand the formatting requirements.

EvidenceConfig.py is applied to every component of the EvidenceMap application/scripts

### Using the pipeline
While the pipeline is most easiest used with the utility scripts, if you wanted to use it for development, you can most easily import the pipeline with the following import statement.

```python
from Website.pipeline import SentenceClassification, Model, NegationDriver, PropositionDriver, EvidenceMapDriver
```

The pipeline is imported based on what is configured in EvidenceConfig.py

However, if you seek more flexibility, you can manage each pipeline component individually by following the Modules section below.

## Modules
### [SentenceClassification](./SentenceClassification)

The SentenceClassification module serves to tag each sentence in an abstract based on what section they are a part of.

To import the module, use the following code:
```python
from SentenceClassification.EvidenceBaseSentenceClassificationDriver import EvidenceBaseSentenceClassificationDriver
SentenceClassification = EvidenceBaseSentenceClassificationDriver.load_driver(<identifier>, <configuration_values>)
```

Please refer to the [SentenceClassification README](./SentenceClassification/README.md) for more detailed usage.

### [Models](./Models)

The Models module stores various implementations of NLP models used for parsing PubMed articles and performing Named Entity Recognition (NER) and Relation Extraction (RE).

To import the Model module, use the following code:

```python
from Models.EvidenceBaseModelDriver import EvidenceBaseModelDriver
Model = EvidenceBaseModelDriver.load_driver(<identifier>, <model_name>, <configuration_values>)
```

Please refer to the [Models README](./Models/README.md) for more detailed usage.

### [Negations](./Negations)

The Negations module provides different implementations of negation detection algorithms.

To import the Negations module, use the following code:

```python
from Negations.EvidenceBaseNegationDriver import EvidenceBaseNegationDriver
Negation = EvidenceBaseNegationDriver.load_driver(<identifier>, Model, <configuration_values>)
```

Please refer to the [Negations README](./Negations/README.md) for more instruction.

### [Propositions](./Propositions)

The Propositions module contains different implementations of proposition formalizers. They take the output of the entities and their relations and formulate propositions.

The module is still under development and its class methods are subject to change.

To import the Propositions module, use the following code:

```python
from Propositions.EvidenceBasePropositionDriver import EvidenceBasePropositionDriver
Propositions = EvidenceBasePropositionDriver.load_driver(<identifier>, Model, Negation, <configuration_values>)
```

Please refer to the [Propositions README](./Propositions/README.md) for more information.

### [EvidenceMap](./EvidenceMap)

The EvidenceMap module provides various implementations of semantic-based clustering techniques. These techniques construct a network graph of the propositions by clustering similar entities together. They then build a PICO map of the clustered formalizations, effectively visualizing the RCT results between the study arms in a simple manner.

This module also builds and deconstructs the JSON documents needed for serving the parsed data to the website component

To import EvidenceMap module, use the following code:

```python
from EvidenceMap.EvidenceMapDriver import EvidenceMapDriver
MapDriver = EvidenceMapDriver.load_driver(<identifier>, Propositions, <configuration_values>)
```

Please refer to the [EvidenceMap README](./EvidenceMap/README.md) for more details.

### Pretrained Models and SOLR
Fine-tuned models and SOLR data can be found [here](https://nextcloud.cybercloudhub.org/s/PFsDa2b2WrRf9Sq).  The password is "EvidenceMap4TW".  Simply merge the folder structure with the project root to install them.
