# Propositions Module

The Propositions serves to formulate medical evidence propositions from the relations of entities outputted from the
previous stages of the pipeline (Models and Negations modules).

Given the relations between different medical entities, the Propositions module will generate propositions that
represent the medical evidence present in the text.

To import the Propositions module, use the following code:

```python
from Propositions.EvidenceBasePropositionDriver import EvidenceBasePropositionDriver
Propositions = EvidenceBasePropositionDriver.load_driver(<identifier>, Model, Negation, <configuration_values>)
```

**Note:** An instance of a Model object is required from the Models module and an instance of a Negation object is
required from the Negations module.

## Usage

The main method of the Propositions module is `buildPropositions`.

After initialization, the `buildPropositions` method can be used as follows:

```python
Propositions.buildPropositions(<model_and_negations_output>)
```

This function requires one argument, which is the output of the Models and Negations predictions. If the output is from
the `detectNegations()` method in Negations module, it needs to be passed as is.

The function `buildPropositions()` adds a 'propositions' field to the predictions output that comes in the form of a
list. The method takes in the preexisting 'entities' and 'relations' and produces a list of proposition dictionaries,
each representing a separate medical proposition.

Please note that as this module is still under development, class methods and names may change. More details will be
added as the development progresses.