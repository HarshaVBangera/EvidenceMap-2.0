### V1 EvidenceMap Configuration Options

The V1 EvidenceMap driver uses an original implementation design.

Below is a list of the configuration options for this module:

- `particpant_eps`: Specifies the clustering epsilon default for Participant entities. Value 0<eps<1.
    - Default: 0.25

- `intervention_eps`: Specifies the clustering epsilon default for Intervention entities. Value 0<eps<1.
    - Default: 0.29

- `outcome_eps`: Specifies the clustering epsilon default for Outcome entities. Value 0<eps<1.
    - Default: 0.25

- `count_eps`: Specifies the clustering epsilon default for Count entities. Value 0<eps<1.
    - Default: 0.01

- `observation_eps`: Specifies the clustering epsilon default for Observation entities. Value 0<eps<1.
    - Default: 0.15

- `metric`: Sets a clustering distance computation method. Above value ranges can be impacted depending on metric used.
    - Default: 'cosine'

- `abbreviation_cluster_threshold`: For handling entities that are identified as abbreviations. Sets the ratio of how
  dominant the usage of the abbreviation is to be determined to be a significant cluster. The higher the number, the
  more dominant the abbreviated entity needs to be merged into the non-abbreviated cluster.
    - Default: 0.5

- `ambiguouity_strictness_threshold`: Some entities indirectly refer to other entities, ie group a, group b. A lower
  number makes it likelier an entity mentioning a group is merged into another entity.
    - Default: 0.8