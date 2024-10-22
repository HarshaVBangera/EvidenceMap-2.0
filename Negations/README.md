# Negations Module

The Negations module provides various implementations of negation detection algorithms using the evidence from the Entity and Relation predictions of the Models module.

Negations are often present in clinical narratives, and failure to correctly identify and interpret them can lead to flipping meaning of the statement. Therefore, it's crucial to consider them in the downstream NLP tasks.

To import the Negations module, use the following code:

```python
from Negations.EvidenceBaseNegationDriver import EvidenceBaseNegationDriver
Negation = EvidenceBaseNegationDriver.load_driver(<identifier>, Model, <configuration_values>)
```

**Note:** An instance of a Model object is required from the Models module.

## Usage

The primary method of the Negations module is the `detectNegations()`.

```python
prediction_output = Negation.detectNegations(<prediction_output_from_Models>)
```
This function requires one argument which is the output of the Models predictions. If the output is from the `predict()` function in Models module, it needs to be put into a list before being passed in.

Possible negation status are: 'affirmed', 'negated', and 'possible' if applicable.

Given below is an example of Models prediction output before and after the negation detection:

```json
//Before negation detection
{
    'sentence': 'The quick brown fox jumps over the lazy dog.',
    'tokens': ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.'],
    'entities': [
        {
            'type': 'Intervention',
            'start': 4,
            'end': 19,
            'text': 'quick brown fox',
            'spacy': <Spacy Doc object>
        },
        {
            'type': 'Outcome',
            'start': 35,
            'end': 43,
            'text': 'lazy dog',
            'spacy': <Spacy Doc object>
        }
    ],
    'relations': [
        {
            'type': 'POSITIVE',
            'head': 0,
            'tail': 1
        }
    ],
    'spacy': <Spacy Doc object>
}
```

```json
//After negation detection
{
    'sentence': 'The quick brown fox jumps over the lazy dog.',
    'tokens': ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.'],
    'entities': [
        {
            'type': 'Intervention',
            'start': 4,
            'end': 19,
            'text': 'quick brown fox',
            'spacy': <Spacy Doc object>,
            'negation_status': 'affirmed'
        },
        {
            'type': 'Outcome',
            'start': 35,
            'end': 43,
            'text': 'lazy dog',
            'spacy': <Spacy Doc object>,
            'negation_status': 'affirmed'
        }
    ],
    'relations': [
        {
            'type': 'POSITIVE',
            'head': 0,
            'tail': 1
        }
    ],
    'spacy': <Spacy Doc object>
}
```
