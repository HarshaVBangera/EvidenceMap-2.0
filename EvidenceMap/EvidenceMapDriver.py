import base64
import importlib
import json
import lzma
import os
import sys
from abc import ABC, abstractmethod

import graphviz
from EvidenceMap.MapNodes import NodeSpace

modules = {key: value for key, value in sys.modules.items() if 'EvidenceBasePropositionDriver' in key}

if modules:
    module = list(modules.values())[0]
    EvidenceBasePropositionDriver = getattr(module, 'EvidenceBasePropositionDriver')
else:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Propositions')))
    from EvidenceBasePropositionDriver import EvidenceBasePropositionDriver


class EvidenceMapDriver(ABC):
    available_drivers = {}
    default_driver = "V1"
    multigroup_rules = ['every', 'groups', 'both', 'each', 'all', 'either']

    """
    Abstract class that defines the methods that must be implemented by a driver for a specific evidence base model.
    """

    @abstractmethod
    def __init__(self, proposition_driver: EvidenceBasePropositionDriver, config):
        self.driver_config = config
        self.driver_defaults = self.driver_config.copy()

        # we'll use this spacy model to split the input into sentences throughout all the drivers
        self.nlp = proposition_driver.nlp

    # this is triggered when a subclass is created
    # it's used to register the driver in the available_drivers dictionary
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "_identifier" in cls.__dict__:
            EvidenceMapDriver.available_drivers[cls._identifier] = cls
        else:
            raise ValueError(f"Class {cls.__name__} does not define an identifier")

    # autoloads all the drivers in the Models directory
    @classmethod
    def import_subclasses(cls):
        parent_directory = os.path.abspath(os.path.dirname(__file__))
        for child_dir_name in os.listdir(parent_directory):
            if child_dir_name == "__pycache__":
                continue
            child_dir_path = os.path.join(parent_directory, child_dir_name)
            if os.path.isdir(child_dir_path):
                module_path = f"EvidenceMap.{child_dir_name}.MapDriver"
                try:
                    importlib.import_module(module_path)
                except (ModuleNotFoundError, AttributeError) as e:
                    print(f"[WARN]: Failed to import module {module_path}: {e}")
                    continue

    def deconstruct_json(self, doc):
        predictions = []
        has_complete_entities = True
        has_complete_negations = True
        has_complete_propositions = False
        has_complete_tags = True
        has_complete_map = False

        pipeline_stages = ['sent_classification', 'entity_recognition', 'negation_detection', 'proposition_extraction',
                           'map_construction']
        minimum_stage_required = 4

        for sentence in doc['Sentence-level breakdown']:
            try:
                pred_sentence = {
                    'tag': sentence['Section'].strip(),
                    'sentence': sentence['Text'].strip(),
                    'spacy': self.nlp(sentence['Text']),
                    'tokens': [token.text for token in self.nlp(sentence['Text'])],
                    'entities': [],
                    'relations': [],
                    'propositions': [],
                    'elements': sentence['Evidence Elements']
                }
            except:
                has_complete_tags = False
                minimum_stage_required = min(minimum_stage_required, 0)
                break
            try:
                for entity_type, entity_group in sentence['Evidence Elements'].items():
                    for entity in entity_group:
                        recontstructed_entity = {
                            'type': entity_type.strip(),
                            'text': entity['term'].strip(),
                            'start': entity['start'],
                            'end': entity['end'],
                            'negation_status': entity['negation'].strip(),
                            'spacy': pred_sentence['spacy']
                        }
                        try:
                            if entity['negation'] == 'negated':
                                recontstructed_entity['negation_position'] = entity['negation_position'].strip()
                                recontstructed_entity['negation_phrase'] = entity['negation_phrase'].strip()
                        except:
                            has_complete_negations = False
                            minimum_stage_required = min(minimum_stage_required, 2)
                        pred_sentence['entities'].append(recontstructed_entity)
            except:
                has_complete_entities = False
                minimum_stage_required = min(minimum_stage_required, 1)
            try:
                for proposition in sentence['Evidence Propositions']:
                    has_complete_propositions = True
                    recontstructed_proposition = {
                        'Intervention': None,
                        'Observation': None,
                        'Outcome': None,
                        'Count': None,
                        'negation': proposition['negation'].strip()
                    }
                    try:
                        if proposition['Intervention']:
                            for entity in pred_sentence['entities']:
                                if isinstance(proposition['Intervention'], list):
                                    if recontstructed_proposition['Intervention'] is None:
                                        recontstructed_proposition['Intervention'] = []
                                    for intervention in proposition['Intervention_start']:
                                        if entity['start'] == intervention:
                                            recontstructed_proposition['Intervention'].append(entity)
                                            break
                                elif entity['start'] == proposition['Intervention_start']:
                                    recontstructed_proposition['Intervention'] = entity
                                    break
                        if proposition['Observation']:
                            for entity in pred_sentence['entities']:
                                if entity['start'] == proposition['Observation_start']:
                                    recontstructed_proposition['Observation'] = entity
                                    break
                        if proposition['Outcome']:
                            for entity in pred_sentence['entities']:
                                if entity['start'] == proposition['Outcome_start']:
                                    recontstructed_proposition['Outcome'] = entity
                                    break
                        if proposition['Count']:
                            for entity in pred_sentence['entities']:
                                if entity['start'] == proposition['Count_start']:
                                    recontstructed_proposition['Count'] = entity
                                    break
                        recontstructed_proposition['negation'] = proposition['negation'].strip()

                    except:
                        has_complete_propositions = False
                        minimum_stage_required = min(minimum_stage_required, 3)
                    pred_sentence['propositions'].append(recontstructed_proposition)
            except:
                has_complete_propositions = False
                minimum_stage_required = min(minimum_stage_required, 3)

            predictions.append(pred_sentence)

        return doc['publication metadata'], predictions, doc['doc_id'], doc[
            'abstract'], has_complete_entities, has_complete_negations, has_complete_propositions, has_complete_tags, has_complete_map

    def build_json(self, participants, EvidenceMap, meta_data, predictions, pmid, doc):
        json_root = {
            "doc_id": pmid,
            "type of study": "Therapy",
            "title": meta_data['title'].strip(),
            "abstract": doc,
            "publication metadata": meta_data,
            "Sentence-level breakdown": [],
            "level3": {
                "data": {
                    "proposed_arms": [],
                    "participants": [],
                    "study_design": [],
                    "study_results": []
                },
                "data_provider": {
                    "name": self.__class__.__name__,
                    "added_elements": self._get_additional_elements(),
                }
            }
        }

        for sentence in predictions:
            json_sentence = {
                "Section": sentence['tag'].strip(),
                "Text": sentence['sentence'].strip(),
                "Evidence Elements": sentence['elements'],
                "Evidence Propositions": []
            }
            for proposition in sentence['propositions']:
                if proposition['Outcome'] is None:
                    continue
                if proposition['Intervention'] is None:
                    continue
                if isinstance(proposition['Intervention'], list):
                    for intervention in proposition['Intervention']:
                        json_proposition = {
                            "Intervention": intervention['text'].strip(),
                            "Intervention_start": intervention['start'],
                            "Observation": proposition['Observation']['text'].strip() if proposition[
                                                                                             'Observation'] is not None else "",
                            "Observation_start": proposition['Observation']['start'] if proposition[
                                                                                            'Observation'] is not None else "",
                            "Outcome": proposition['Outcome']['text'].strip(),
                            "Outcome_start": proposition['Outcome']['start'],
                            "Count": proposition['Count']['text'].strip() if proposition['Count'] is not None else "",
                            "Count_start": proposition['Count']['start'] if proposition['Count'] is not None else "",
                            "negation": proposition['negation'].strip(),
                            # "negation_phrase":
                        }
                        json_sentence['Evidence Propositions'].append(json_proposition)
                else:
                    json_proposition = {
                        "Intervention": proposition['Intervention']['text'].strip(),
                        "Intervention_start": proposition['Intervention']['start'],
                        "Observation": proposition['Observation']['text'].strip() if proposition[
                                                                                         'Observation'] is not None else "",
                        "Observation_start": proposition['Observation']['start'] if proposition[
                                                                                        'Observation'] is not None else "",
                        "Outcome": proposition['Outcome']['text'].strip(),
                        "Outcome_start": proposition['Outcome']['start'],
                        "Count": proposition['Count']['text'].strip() if proposition['Count'] is not None else "",
                        "Count_start": proposition['Count']['start'] if proposition['Count'] is not None else "",
                        "negation": proposition['negation'].strip(),
                    }
                    json_sentence['Evidence Propositions'].append(json_proposition)
            json_root['Sentence-level breakdown'].append(json_sentence)

        traversed_nodes = set()
        for participant in participants.children:
            json_root['level3']['data']['participants'].append(
                {"term": str(participant), "umls": {}, "negation": participant.negation})
            traversed_nodes.add(participant)

        study_design = []
        study_results = []

        for node in EvidenceMap.node_dict.values():
            if node in traversed_nodes:
                continue
            if str(node) == "Root":
                continue
            json_node = {
                "term": str(node),
                "umls": {},
                "connects_from": [str(parent.cluster_label) for parent in node.parents],
                "connects_to": [str(child.cluster_label) for child in node.children],
                "label": str(node.cluster_label),
                "type": str(node.entity_type).replace("_D", "")
            }

            traversed_nodes.add(node)

            if [item for item in (json_node['connects_to'] + json_node['connects_from']) if item != "Root"]:
                if node.entity_type.endswith('_D'):
                    study_design.append(json_node)
                else:
                    study_results.append(json_node)

        json_root['level3']['data']['study_design'] = study_design
        json_root['level3']['data']['study_results'] = study_results

        json_root['payload'] = encode_json_payload(minify_json(json_root))

        return json_root

    def build_map(self, all_model_predictions, proposed_arms=None, print_output=False):
        all_evidence_maps = []

        for model_predictions in all_model_predictions:
            EvidenceMap = NodeSpace()
            Root_nodes = [
                EvidenceMap.get_node_by_label("Participants", "Root", "Root"),
                EvidenceMap.get_node_by_label("Design", "Root", "Root"),
                EvidenceMap.get_node_by_label("Results", "Root", "Root"),
            ]

            # Initialize variables for design arms
            Observation_Arms = []
            Outcome_Arms = []
            Design_Chains = []
            NodeLabels = {
                "Participant": ["Participant", "Participant_D"],
                "Intervention": ["Intervention", "Intervention_D"],
                "Observation": ["Observation", "Observation_D"],
                "Outcome": ["Outcome", "Outcome_D"],
                "Count": ["Count", "Count_D"],
            }

            results_sections = [
                'RESULTS',
                'CONCLUSIONS',
                'CONCLUSION',
                'FINDINGS',
                'INTERPRETATION',
            ]

            if print_output:
                print("Propositions found:")

            for prediction in model_predictions:
                NodeLabelFlag = int(prediction['tag'] not in results_sections)
                for entity in prediction['entities']:
                    if entity['type'] == 'Participant':
                        Participant_node = EvidenceMap.get_node_by_label("Participant", entity['text'], entity['cluster'])
                        if entity['negation_status'] == 'negated':
                            Participant_node.negation = True
                        Root_nodes[0].add_child(Participant_node)

                    if proposed_arms is None and entity['type'] == 'Intervention':
                        if prediction['tag'] not in results_sections:
                            Intervention_nodes = EvidenceMap.get_nodes_by_label("Intervention_D", entity['text'], entity['cluster'])
                            for Intervention_node in Intervention_nodes:
                                Root_nodes[1].add_child(Intervention_node)

                    if entity['type'] == 'Observation':
                        if prediction['tag'] not in results_sections:
                            Observation_Arms.append(
                                EvidenceMap.get_nodes_by_label("Observation_D", entity['text'], entity['cluster'])[0]
                            )

                    if entity['type'] == 'Outcome':
                        if prediction['tag'] not in results_sections:
                            Outcome_Arms.append(
                                EvidenceMap.get_nodes_by_label("Outcome_D", entity['text'], entity['cluster'])[0]
                            )

                for proposition in prediction['propositions']:
                    if proposition['Count'] is not None:
                        middle = proposition['Count']['cluster_text']
                    else:
                        middle = proposition['Observation']['cluster_text']

                    if isinstance(proposition['Intervention'], list):
                        for intervention in proposition['Intervention']:
                            if print_output:
                                print(f"{intervention['cluster_text']}→{middle}→{proposition['Outcome']['cluster_text']}")
                    else:
                        if print_output:
                            try:
                                print(f"{proposition['Intervention']['cluster_text']}→{middle}→{proposition['Outcome']['cluster_text']}")
                            except KeyError:
                                pass  # Handle specific key errors

                    # Process nodes
                    Observation_node = EvidenceMap.get_nodes_by_label(
                        NodeLabels['Observation'][NodeLabelFlag],
                        proposition['Observation']['cluster_text'],
                        proposition['Observation']['cluster']
                    )[0] if proposition['Observation'] is not None else None
                    
                    Count_node = EvidenceMap.get_nodes_by_label(
                        NodeLabels['Count'][NodeLabelFlag], 
                        proposition['Count']['text'],
                        proposition['Count']['cluster']
                    )[0] if proposition['Count'] is not None else None
                    
                    Outcome_node = EvidenceMap.get_nodes_by_label(
                        NodeLabels['Outcome'][NodeLabelFlag], 
                        proposition['Outcome']['text'],
                        proposition['Outcome']['cluster']
                    )[0] if proposition['Outcome'] is not None else None

                    middle_node = None
                    if Observation_node and Outcome_node:
                        Outcome_node.add_parent(Observation_node)
                        middle_node = Observation_node
                    elif Count_node and Outcome_node:
                        Outcome_node.add_parent(Count_node)
                        middle_node = Count_node

                    # Process interventions
                    if isinstance(proposition['Intervention'], list):
                        for intervention in proposition['Intervention']:
                            I_nodes = EvidenceMap.get_nodes_by_label("Intervention", intervention['text'], intervention['cluster'])
                            for I_node in I_nodes:
                                Root_nodes[2].add_child(I_node)
                                if prediction['tag'] in results_sections and middle_node:
                                    middle_node.add_parent(I_node)
                                else:
                                    Design_Chains.append(middle_node)
                    else:
                        I_nodes = EvidenceMap.get_nodes_by_label("Intervention", proposition['Intervention']['text'], proposition['Intervention']['cluster'])
                        for I_node in I_nodes:
                            Root_nodes[2].add_child(I_node)
                            if prediction['tag'] in results_sections and middle_node:
                                middle_node.add_parent(I_node)
                            else:
                                Design_Chains.append(middle_node)

            # Process Design Chains and finalize the Evidence Map
            if Design_Chains:
                for chain in Design_Chains:
                    for child in Root_nodes[1].children:
                        child.add_child(chain)
            else:
                if Outcome_Arms and not Observation_Arms:
                    Observation_Arm = EvidenceMap.get_node_by_label("Observation_D", "Compare", "O")
                    for Outcome_Arm in Outcome_Arms:
                        Outcome_Arm.add_parent(Observation_Arm)
                    for child in Root_nodes[1].children:
                        child.add_child(Observation_Arm)
                if Observation_Arms and Outcome_Arms:
                    for Observation_Arm in Observation_Arms:
                        for Outcome_Arm in Outcome_Arms:
                            for child in Root_nodes[1].children:
                                child.add_child(Observation_Arm)
                                Observation_Arm.add_child(Outcome_Arm)
                else:
                    Root_nodes[1].children = set()

            # Remove any intervention that doesn't have any children
            for child in list(Root_nodes[2].children):
                if not child.children:
                    Root_nodes[2].children.discard(child)

            # Store the evidence map for the current document
            all_evidence_maps.append((Root_nodes, EvidenceMap))

        return all_evidence_maps


    def _get_additional_elements(self):
        return {}

    def draw_map(self, map_roots, map_title="Evidence Map", output_dir=None):
    # Ensure map_roots is a list for multiple document handling
        if not isinstance(map_roots, list):
            map_roots = [map_roots]  # Wrap single map_root in a list

        for index, map_root in enumerate(map_roots):
            dot = graphviz.Digraph(comment=f"{map_title} - Document {index + 1}", format='svg')
            dot.attr(rankdir='TB')

            node_colors = {
                'Intervention': self.driver_config.get('intervention_color', '#C5E0B5'),
                'Intervention_D': self.driver_config.get('intervention_color', '#C5E0B5'),
                'Count': self.driver_config.get('count_color', '#D9D9D9'),
                'Observation': self.driver_config.get('observation_color', '#D9D9D9'),
                'Observation_D': self.driver_config.get('observation_color', '#D9D9D9'),
                'Outcome': self.driver_config.get('outcome_color', '#FFD962'),
                'Outcome_D': self.driver_config.get('outcome_color', '#FFD962'),
                'Participant': self.driver_config.get('participant_color', '#A1C2E1'),
            }
            edges = set()

            def add_node(node):
                color = node_colors.get(node.entity_type, '#A1C2E1')
                dot.node(str(hash(node)), str(node), shape='rectangle', style='filled', fillcolor=color, color='none')

            def draw_edges(node, recurse=False):
                if str(node) != "Root" and not recurse:
                    add_node(node)
                for child in node.children:
                    add_node(child)
                    edge = (node, child)
                    if str(node) != "Root" and recurse and edge not in edges:
                        dot.edge(str(hash(node)), str(hash(child)))
                        edges.add(edge)
                    draw_edges(child, True)

            draw_edges(map_root)

            # Create a unique output file path for each document's evidence map
            output_file_path = os.path.join(output_dir, f"{map_title.replace(' ', '_')}_Document_{index + 1}")
            dot.render(output_file_path, format='svg', cleanup=True)

    @abstractmethod
    def fit_propositions(self, model_predictions, print_output=False):
        pass

    # loads a driver by its identifier, and initializes it with the model and config
    @classmethod
    def load_driver(cls, identifier, model, config):
        driver_class = cls.available_drivers.get(identifier)
        if driver_class:
            return driver_class(model, config)
        else:
            raise ValueError(f"No driver found with identifier {identifier}")


EvidenceMapDriver.import_subclasses()


def minify_json(json_obj):
    return json.dumps(json_obj, separators=(',', ':'))


# Function to encode minified JSON in base62
def encode_json_payload(minified_json):
    return base64.b64encode(lzma.compress(minified_json.encode())).decode()


def decode_json_payload(encoded_json):
    return lzma.decompress(base64.b64decode(encoded_json.encode()))
