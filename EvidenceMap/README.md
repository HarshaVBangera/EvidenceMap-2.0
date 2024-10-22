# EvidenceMap Module

The EvidenceMap module, is the final stage in a pipeline which takes the output from the previous components and builds
a visual network graph summarizing the abstract.
This module also constructs the final JSON document needed for the Website component.

To import the EvidenceMap module, use the following code:

```python
from EvidenceMap.EvidenceMapDriver import EvidenceMapDriver

EvidenceMapDriver = EvidenceMapDriver.load_driver( < identifier >, PropositionDriver, < configuration_values >)
```

**Note:** An instance of a PropositionDriver object is required from the PropositionDriver module.

## Usage

When the module is loaded, several methods become available:

- `fit_propositions(<output from PropositionDriver>)`: Deduplicates similar entities, ie Diabetes and T2DM, so they are
  not presented multiple times. Returns a tuple, with the model predictions, and proposed study arms.
- `build_map(<prediction from fit_proposition>, <proposed_arms from fit_propositions>)`: Generates an enrollment, study
  design di-graph, and a study results di-graph and returns a tuple containing a list of digraph root nodes for each
  category, and an EvidenceMap NodeSpace storing all active nodes.
- `draw_map(<a Root from build_map>, [<optional file_name>, <optional_directory_path>])`: Draws an SVG of the supplied
  digraph and stores it to the file system.
- `build_json(<Root node of participant from build_map>, <EvidenceMap from build_map>, <list of metadata from PubMed abstract>, <prediction output from fit_propositions()>, <PubMed ID>, <Abstract Text>)`:
  Builds JSON document to be uploaded to SOLR for the website.
- `deconstruct_json(<JSON Decoded from SOLR>)`: Reconstructs the model prediction format needed for the pipeline to
  function.
    - Returns a tuple consisting of:
        - PubMed metadata
        - model_predictions, formatted to be used in the pipeline
        - PMID
        - Abstract text
        - A boolean indicating successful restoration of entity information
        - A boolean indicating successful restoration of negation information
        - A boolean indicating successful restoration of proposition information
        - A boolean indicating successful restoration of sentence tags/classes
        - A boolean indicating successful restoration of the Evidence Map
    - The booleans serve as indicators as to which part of the pipeline the parsing process must start at to ensure data
      integrity.
    - At this time, build_json and deconstruct_json are not bidirectional.
        - The output of a build_json can be fed into a deconstruct_json without issue.
        - The output of a deconstruct_json CANNOT be fed into a build_json as the built JSON lacks vital metadata that
          must be recomputed from the pipeline first.
        - In other words, <Final Pipeline Data>->build_json()->deconstruct_json()-><Lossy/incomplete Pipeline Data>