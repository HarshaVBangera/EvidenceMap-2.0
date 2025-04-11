import os
import sys

from itertools import product

modules = {key: value for key, value in sys.modules.items() if 'EvidenceBasePropositionDriver' in key}

if modules:
    module = list(modules.values())[0]
    EvidenceBasePropositionDriver = getattr(module, 'EvidenceBasePropositionDriver')
else:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from EvidenceBasePropositionDriver import EvidenceBasePropositionDriver


# This driver is adapted from the original negation code in the parsing pipeline.  It has been cleaned up and simplified for slightly better readability and maintainability
class V0PropositionDriver(EvidenceBasePropositionDriver):
    _identifier = "V0"

    def __init__(self, model_driver, negation_driver, config):
        super().__init__(model_driver, negation_driver, config)

    def buildPropositions(self, model_predictions):
        comp_tags = ["JJS", "JJR", "RBS", "RBR"]
        comp_terms = ["as", "difference", "different", "same", "similar", "differences", "more"]

        for prediction in model_predictions:
            prediction['elements'] = {"Participant": [], "Intervention": [], "Outcome": [], "Observation": [],
                                      "Count": []}
            propositions = []
            special_entities = []
            prediction['propositions'] = []

            for eid, entity in enumerate(prediction['entities']):
                if entity['type'] not in ["Outcome", "Intervention", "Participant"]:
                    special_entities.append((eid, entity['type']))

                element = {
                    "term": entity['text'],
                    "negation": entity['negation_status'],
                    "UMLS": {},
                    "start": entity['start'],
                    "end": entity['end']
                }
                if entity['negation_status'] == "negated":
                    element['negation_position'] = entity['negation_position']
                    element['negation_phrase'] = entity['negation_phrase']
                prediction['elements'][entity['type']].append(element)

            for ob_id, ob_type in special_entities:
                observation_entity = prediction['entities'][ob_id]
                ob_tags = [token.tag_ for token in self.nlp(observation_entity['text'])]

                flag = False
                for comp in comp_tags:
                    if comp in ob_tags:
                        flag = True
                for t in observation_entity['text'].split(" "):
                    if t.lower() in comp_terms:
                        flag = True

                I = []
                O = []
                for relation in prediction['relations']:
                    if ob_id == relation['head']:
                        other_id = relation['tail']
                    elif ob_id == relation['tail']:
                        other_id = relation['head']
                    else:
                        continue

                    other_entity = prediction['entities'][other_id]

                    if other_entity['type'] == "Outcome":
                        O.append(other_entity)
                    elif other_entity['type'] == "Intervention":
                        I.append(other_entity)

                if not flag:
                    output = list(product(I, O))
                    for o in output:
                        mep = list(o)
                        mep.insert(1, observation_entity)
                        propositions.append((mep, observation_entity['negation_status'], ob_type))
                else:
                    if len(O) == 0:
                        propositions.append(
                            ([I, observation_entity, None], observation_entity['negation_status'], ob_type)
                        )
                    for outcome_entity in O:
                        propositions.append(
                            ([I, observation_entity, outcome_entity], observation_entity['negation_status'], ob_type)
                        )

            for proposition, prop_negation, prop_type in propositions:
                intervention_entities = proposition[0]

                special_entity = proposition[2]

                out_prop = {
                    "Intervention": intervention_entities,
                    "Observation": None,
                    "Outcome": special_entity,
                    "Count": None,
                    "negation": prop_negation
                }

                out_prop[prop_type] = proposition[1]

                prediction['propositions'].append(out_prop)

        return model_predictions
