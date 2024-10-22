The Models module houses various NLP model implementations used to parse PubMed articles, performing tasks such as Named
Entity Recognition (NER) and Relationship Extraction (RE). Users should refer to the [Models README](./Models/README.md)
for specific instructions and documentation on each model driver. Each implementation, or "driver," has its specific
options and requirements.

Available drivers are listed in the `available_drivers` dictionary within the loaded module object. The recommended
driver for current use is `V1`.

When the module is loaded, one method becomes available:

- `classifySentences(['<sentence>', '<sentence>', ...])`: Classifies each sentence and attempts to predict which
  abstract section it belongs to.

Example:

```python
docs, tags = SentenceClassification.classifySentences(docs)
```

Ideally, SentenceClassification should be passed into the Model when running a prediction as it will bundle and handle
doc pre-processing automatically.