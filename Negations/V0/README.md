# V0 Negations Driver

The V0 Negations driver is a simple port from older code from the pre-existing code-base. It's been adapted to work
natively in this new pipeline.

While the V0 Negations driver does not add much functionality beyond basic negation detection, it does offer a feature
to tag possible negations by considering the contextual usage of a phrase.

## Configuration

This version of the Negations driver has one configuration option:

- `tag_possible_phrases`: This configuration specifies whether the Negations driver should mark entities where negation
  might not be explicit but is possible due to the context. Entities, where negation is possible, would be tagged with
  a 'negation_status' of 'possible'.
    - Default: True
