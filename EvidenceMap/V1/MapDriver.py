import os
import re
import sys
from collections import defaultdict, Counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

modules = {key: value for key, value in sys.modules.items() if 'EvidenceMapDriver' in key}

if modules:
    module = list(modules.values())[0]
    EvidenceMapDriver = getattr(module, 'EvidenceMapDriver')
else:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from EvidenceMapDriver import EvidenceMapDriver

class V1MapDriver(EvidenceMapDriver):
    _identifier = "V1"

    def __init__(self, proposition_driver, config):
        super().__init__(proposition_driver, config)

        self.model = SentenceTransformer('NeuML/pubmedbert-base-embeddings')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def fit_propositions(self, all_model_predictions, print_output=False):
        all_entities = []
        ClusterContainer._containers = {}

        for model_predictions in all_model_predictions:  # Loop through each document
            entities = []
            for prediction in model_predictions:
                for entity in prediction['entities']:
                    entity['cluster_text'] = entity['text']

                    if entity['type'] == 'Observation' and entity['negation_status'] == 'negated':
                        entity['cluster_text'] = f"{entity['negation_phrase']} {entity['text']}" if entity[
                            'negation_position'] == 'pre' else f"{entity['text']} {entity['negation_phrase']}"

                    entities.append(entity)
            
            all_entities.extend(entities)  # Collect entities from all documents

        try:
            types, sbert_embeddings, model_embeddings = self._generate_entities_embeddings(all_entities)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return all_model_predictions, None  # Or handle it as necessary

        study_arm_clusters = self._cluster_entities(types, sbert_embeddings, all_entities, print_output)
        return all_model_predictions, study_arm_clusters


    def _get_additional_elements(self, data_provider={}):
        num_documents = len(data_provider.get('documents', []))  # Assuming data_provider contains documents

        # Start building the HTML output
        html_output = """
        <div style="display: flex; align-items: center; justify-content: center;">
            <p style="margin: 0;">Grouping Thresholds</p>
            <div class="far fa-question-circle question" data-toggle="tooltip" data-placement="right" data-html="true" title="Adjust the slider or input a value between 0 and 1 to control how the system groups similar terms together in the graph. A higher value creates broader groups, collecting more similar terms together, while a lower value leads to more precise and smaller groups. This is an experimental feature and any changes you make are saved. For more precise groups, use a lower value. To group broader, more similar terms together, use a higher value." style="margin-left: 10px;"></div>
        </div>
        <form id="enrollment_form" style="margin-top: 20px;">
            <div style="text-align:left">
                <input type="hidden" name="action" value="reparse">
        """

        # Loop through each document and generate form fields
        for i in range(num_documents):
            html_output += f"""
            <h4>Document {i + 1}</h4>
            <div class="form-group" style="display: flex; align-items: center; margin-bottom: 0">
                <label for="participant_eps_{i}" style="flex: 1; margin-right: 1rem; text-align: left;">Participant:</label>
                <input type="range" id="participant_eps_{i}" name="EvidenceMap.participant_eps[{i}]" min="0.01" max="1" step="0.01" value=\"""" + str(self.driver_config.get('participant_eps', 0.25)) + """\" oninput="this.nextElementSibling.value = this.value" style="flex: 2; margin-right: 1rem;">
                <input type="number" class="form-control" id="participant_eps_value_{i}" name="EvidenceMap.participant_eps_value[{i}]" min="0.01" max="1" step="0.01" value=\"""" + str(self.driver_config.get('participant_eps', 0.25)) + """\" oninput="this.previousElementSibling.value = this.value" style="flex: 0.5;">
            </div>
            <div class="form-group" style="display: flex; align-items: center; margin-bottom: 0">
                <label for="intervention_eps_{i}" style="flex: 1; margin-right: 1rem; text-align: left;">Intervention:</label>
                <input type="range" id="intervention_eps_{i}" name="EvidenceMap.intervention_eps[{i}]" min="0.01" max="1" step="0.01" value=\"""" + str(self.driver_config.get('intervention_eps', 0.29)) + """\" oninput="this.nextElementSibling.value = this.value" style="flex: 2; margin-right: 1rem;">
                <input type="number" class="form-control" id="intervention_eps_value_{i}" name="EvidenceMap.intervention_eps_value[{i}]" min="0.01" max="1" step="0.01" value=\"""" + str(self.driver_config.get('intervention_eps', 0.29)) + """\" oninput="this.previousElementSibling.value = this.value" style="flex: 0.5;">
            </div>
            <div class="form-group" style="display: flex; align-items: center; margin-bottom: 0">
                <label for="outcome_eps_{i}" style="flex: 1; margin-right: 1rem; text-align: left;">Outcome:</label>
                <input type="range" id="outcome_eps_{i}" name="EvidenceMap.outcome_eps[{i}]" min="0.01" max="1" step="0.01" value=\"""" + str(self.driver_config.get('outcome_eps', 0.25)) + """\" oninput="this.nextElementSibling.value = this.value" style="flex: 2; margin-right: 1rem;">
                <input type="number" class="form-control" id="outcome_eps_value_{i}" name="EvidenceMap.outcome_eps_value[{i}]" min="0.01" max="1" step="0.01" value=\"""" + str(self.driver_config.get('outcome_eps', 0.25)) + """\" oninput="this.previousElementSibling.value = this.value" style="flex: 0.5;">
            </div>
            <div class="form-group" style="display: flex; align-items: center; margin-bottom: 0">
                <label for="count_eps_{i}" style="flex: 1; margin-right: 1rem; text-align: left;">Count:</label>
                <input type="range" id="count_eps_{i}" name="EvidenceMap.count_eps[{i}]" min="0.01" max="1" step="0.01" value=\"""" + str(self.driver_config.get('count_eps', 0.01)) + """\" oninput="this.nextElementSibling.value = this.value" style="flex: 2; margin-right: 1rem;">
                <input type="number" class="form-control" id="count_eps_value_{i}" name="EvidenceMap.count_eps_value[{i}]" min="0.01" max="1" step="0.01" value=\"""" + str(self.driver_config.get('count_eps', 0.01)) + """\" oninput="this.previousElementSibling.value = this.value" style="flex: 0.5;">
            </div>
            <div class="form-group" style="display: flex; align-items: center; margin-bottom: 0">
                <label for="observation_eps_{i}" style="flex: 1; margin-right: 1rem; text-align: left;">Observation:</label>
                <input type="range" id="observation_eps_{i}" name="EvidenceMap.observation_eps[{i}]" min="0.01" max="1" step="0.01" value=\"""" + str(self.driver_config.get('observation_eps', 0.15)) + """\" oninput="this.nextElementSibling.value = this.value" style="flex: 2; margin-right: 1rem;">
                <input type="number" class="form-control" id="observation_eps_value_{i}" name="EvidenceMap.observation_eps_value[{i}]" min="0.01" max="1" step="0.01" value=\"""" + str(self.driver_config.get('observation_eps', 0.15)) + """\" oninput="this.previousElementSibling.value = this.value" style="flex: 0.5;">
            </div>
            """

        # Closing the form with buttons
        html_output += """
            </div>
            <button type="button" id="resetButton" class="btn btn-secondary" style="display:inline; margin-left: 10px; background:#347ef4; color:white;border-style: none;border-radius: 3px;">Reset</button>
            <button type="submit" class="btn btn-success" style="display:inline; margin-left: 10px; background:#347ef4; color:white;border-style: none;border-radius: 3px;">Submit</button>
        </form>
        <script type="text/javascript">
        document.addEventListener('DOMContentLoaded', function() {
            function resetForm() {
                for (let i = 0; i < """ + str(num_documents) + """; i++) {
                    document.getElementById('participant_eps_' + i).value = """ + str(self.driver_defaults.get('participant_eps', 0.25)) + """;
                    document.getElementById('participant_eps_value_' + i).value = """ + str(self.driver_defaults.get('participant_eps', 0.25)) + """;
                    document.getElementById('intervention_eps_' + i).value = """ + str(self.driver_defaults.get('intervention_eps', 0.29)) + """;
                    document.getElementById('intervention_eps_value_' + i).value = """ + str(self.driver_defaults.get('intervention_eps', 0.29)) + """;
                    document.getElementById('outcome_eps_' + i).value = """ + str(self.driver_defaults.get('outcome_eps', 0.25)) + """;
                    document.getElementById('outcome_eps_value_' + i).value = """ + str(self.driver_defaults.get('outcome_eps', 0.25)) + """;
                    document.getElementById('count_eps_' + i).value = """ + str(self.driver_defaults.get('count_eps', 0.01)) + """;
                    document.getElementById('count_eps_value_' + i).value = """ + str(self.driver_defaults.get('count_eps', 0.01)) + """;
                    document.getElementById('observation_eps_' + i).value = """ + str(self.driver_defaults.get('observation_eps', 0.15)) + """;
                    document.getElementById('observation_eps_value_' + i).value = """ + str(self.driver_defaults.get('observation_eps', 0.15)) + """;
                }
            }
            
            document.getElementById('resetButton').addEventListener('click', resetForm);
        });
        </script>
        """
        
        return html_output

    def _get_clusters(self, type, embeddings, entities_by_type):
        # Initialize the clustering algorithm with the appropriate parameters
        clustering = DBSCAN(
            eps=float({
                'Intervention': self.driver_config.get('intervention_eps', 0.29),
                'Outcome': self.driver_config.get('outcome_eps', 0.25),
                'Count': self.driver_config.get('count_eps', 0.01),
                'Participant': self.driver_config.get('participant_eps', 0.25),
                'Observation': self.driver_config.get('observation_eps', 0.15),
            }[type]),
            min_samples=1,
            metric=self.driver_config.get('metric', 'cosine')
        )

        # Fit the clustering algorithm to the embeddings
        labels = clustering.fit_predict(embeddings)
        self.clustering[type] = clustering

        clusters = defaultdict(list)
        
        # Use a dictionary to map cluster labels to their respective elements
        for label, entity in zip(labels, entities_by_type[type]):
            cluster = get_cluster_container(label, type)
            cluster.add_element(entity['cluster_text'])
            cluster.add_base_element(entity['text'])
            entity['cluster'] = cluster
            
            # Avoid duplication of clusters by ensuring the label is only added once
            if label not in clusters:
                clusters[label] = cluster

        # Returning clusters as a dictionary with cluster labels as keys
        return clusters


    def _cluster_entities(self, types, embeddings, entities, print_output=False):
        embeddings_by_type = defaultdict(list)
        entities_by_type = defaultdict(list)

        embeddings_by_type['Intervention'] = []
        entities_by_type['Intervention'] = []

        for type, embedding, entity in zip(types, embeddings, entities):
            embeddings_by_type[type].append(embedding)
            entities_by_type[type].append(entity)

        self.clustering = {}

        def find_abbreviated_clusters(clusters):
            return {cluster.cluster_id: common[0]
                    for cluster in clusters.values()
                    if (common := Counter(cluster.get_elements()).most_common(1)[0])
                    and len(common[0]) <= 8
                    and common[0].isupper()
                    and common[1] / len(cluster.get_elements()) > self.driver_config.get(
                    'abbreviation_cluster_threshold', 0.5)}

        def is_ambiguous(entity, ambiguous_pattern):
            match = ambiguous_pattern.search(entity['cluster_text'])
            if match:
                start, end = match.span()
                return (end - start) / len(entity['cluster_text']) > self.driver_config.get(
                    'ambiguouity_strictness_threshold', 0.8)
            return False

        # Perform clustering for each type
        for type, embeddings in embeddings_by_type.items():
            clusters = self._get_clusters(type, embeddings, entities_by_type)

            if (type == 'Intervention'):
                # Discover abbreviated entities and match them to the non-abbreviated entities
                abbrev_clusters = find_abbreviated_clusters(clusters)

                for abbrev_label, abbrev in abbrev_clusters.items():
                    for cluster_label, cluster in clusters.items():
                        if abbrev_label == cluster_label:
                            continue
                        pattern = re.compile(rf'\b{re.escape(abbrev)}\b')
                        if any(pattern.search(entity) for entity in cluster.get_elements()):
                            cluster.link_clusters(clusters[abbrev_label])
                            clusters[abbrev_label].cluster_id = cluster_label
                            break
                # Some clusters are ambiguous, so we need to try to assign them to the correct cluster.
                # When an abstract assigns a "group A" or "group B", it's usually preceeded/followed by the formal name
                ambiguous_terms = [
                    r'group[^s]\s*\w+', r'group[^s]\s*\d+', r'\w+\s*group[^s]',
                    r'group[^s]\s*[a-z]+', r'[a-z]+\s*group[^s]', r'group[^s]\s*[ivxlc]+',
                    r'group[^s]\s*[i-v]+'
                ]
                ambiguous_pattern = re.compile('|'.join(ambiguous_terms), re.IGNORECASE)

                last_doc = None
                ambiguous_entities = []
                multi_referencing_clusters = set()
                rules = ["either", "groups", "each", "every", "all", "both", "several", "few", "multiple"]

                for i, entity in enumerate(entities_by_type[type]):
                    if any(re.search(fr'\b{rule}\b', entity['cluster_text']) for rule in rules):
                        multi_referencing_clusters.add(entity['cluster'])

                    if entity['spacy'] != last_doc:
                        # Process the sequence of entities in the document
                        if ambiguous_entities:
                            self._process_ambiguous_entities(entity_sequence[last_doc], ambiguous_entities, unambiguous_clusters,
                                                             ambiguous_indices, unambiguous_indices)
                        ambiguous_entities, unambiguous_clusters, entity_sequence, unambiguous_indices, ambiguous_indices = [], [], [], [], []
                        last_doc = entity['spacy']
                        index = 0
                    # Check if the entity is ambiguous
                    if is_ambiguous(entity, ambiguous_pattern):
                        ambiguous_entities.append(entity['cluster_text'])
                        ambiguous_indices.append(index)
                    else:
                        unambiguous_clusters.append(entity['cluster'])
                        unambiguous_indices.append(index)
                    entity_sequence.append(entity)
                    index += 1

                if ambiguous_entities:
                    self._process_ambiguous_entities(entity_sequence, ambiguous_entities, unambiguous_clusters,
                                                     ambiguous_indices, unambiguous_indices)

                # finally we need to assign multi-referencing entities to the correct formal clusters

                # first identify most occupied clusters
                sorted_clusters = sorted(clusters.values(), key=lambda x: len(x), reverse=True)

                # figure out which clusters are the most populated, as those are most likely the relevant entities we want to link to
                total_entities = len(entities_by_type[type])

                for cluster in multi_referencing_clusters:
                    total_entities -= len(cluster)

                proportion = 0.8
                total_selected = 0
                selected_clusters = []

                for cluster in sorted_clusters:
                    if cluster in multi_referencing_clusters:
                        continue
                    total_selected += len(cluster)
                    selected_clusters.append(cluster)
                    if total_selected / total_entities >= proportion:
                        break

                for cluster in multi_referencing_clusters:
                    cluster.add_aliases(selected_clusters)

                for cluster in clusters.values():
                    if print_output:
                        print(cluster, len(cluster.get_elements()))

            for entity in entities_by_type[type]:
                if print_output:
                    print(
                        f"Type: {type}, Labels: {entity['cluster'].get_cluster_ids(entity['cluster_text'])}, Entity: {entity['cluster_text']}")

        return selected_clusters

    def _process_ambiguous_entities(self, entities, ambiguous_entities, unambiguous_clusters, ambiguous_indices,
                                    unambiguous_indices):
        match_forward = False
        first_entity = True

        if not unambiguous_clusters:
            return

        for _ in range(2):
            proposed_matches = []
            break_loop = False
            already_matched = []
            entities_matched = []
            for i, ambiguous_entity in enumerate(ambiguous_entities):
                ambiguous_entity = ambiguous_entity.lower()
                if ambiguous_entity in entities_matched:
                    index = entities_matched.index(ambiguous_entity)
                    proposed_matches.append(already_matched[index])
                    continue
                # compute distances from each ambiguous entity to each unambiguous entity
                distances = [[abs(k - j) for j in unambiguous_indices] for k in ambiguous_indices]
                # figure out the distance where an ambiguous entity has the greatest distance to the nearest unambiguous entity
                target_distance = max([min(distances[k]) for k in range(len(ambiguous_entities))])

                min_distance = float('inf')
                best_match = None
                ambiguous_index = ambiguous_indices[i]
                for j, unambiguous_cluster in enumerate(unambiguous_clusters):
                    unambiguous_index = unambiguous_indices[j]
                    if first_entity:
                        if ambiguous_index < unambiguous_index:
                            match_forward = True
                            break_loop = True
                        first_entity = False
                        if break_loop:
                            break

                    if match_forward and ambiguous_index > unambiguous_index:
                        continue
                    elif not match_forward and ambiguous_index < unambiguous_index:
                        break

                    distance = abs(ambiguous_index - unambiguous_index)
                    if distance < min_distance and distance >= target_distance and unambiguous_cluster not in already_matched:
                        min_distance = distance
                        best_match = unambiguous_cluster
                if break_loop:
                    break
                if best_match is not None:
                    proposed_matches.append(best_match)
                else:
                    proposed_matches.append(None)
                    if not match_forward:
                        match_forward = True
                        break
            if not match_forward:
                break

        if None in proposed_matches:
            print("[WARN]: Failed to match ambiguous entities")
            return
        else:
            for i, proposed_cluster in enumerate(proposed_matches):
                entity = entities[ambiguous_indices[i]]
                target_entity = entities[unambiguous_indices[i]]
                entity['cluster'].set_ambiguity_term(ambiguous_entities[i], target_entity['cluster'])

    # try a different way to generate embeddings
    def _generate_entities_embeddings(self, entities):
        phrases = [entity['cluster_text'] for entity in entities]
        types = [entity['type'] for entity in entities]

        embeddings = self.model.encode(phrases, show_progress_bar=False)

        sbert_embeddings = []
        model_embeddings = []

        for entity, embedding in zip(entities, embeddings):
            entity['sbert_embedding'] = embedding
            sbert_embeddings.append(embedding)

            try:
                cls = entity['cls'].unsqueeze(0)
                embeddings = entity['embeddings']
                model_embedding = torch.mean(torch.cat((cls, embeddings), dim=0), dim=0).cpu().detach().numpy()
                entity['model_embedding'] = model_embedding
                model_embeddings.append(model_embedding)
            except:
                entity['model_embedding'] = None
                model_embeddings.append(None)

        # Convert lists to numpy arrays
        sbert_embeddings = np.array(sbert_embeddings)
        model_embeddings = np.array(model_embeddings)

        return types, sbert_embeddings, model_embeddings


class ClusterContainer:
    _containers = {}

    def __init__(self, cluster_id, cluster_type):
        self.cluster_id = cluster_id
        self.cluster_type = cluster_type
        self._linked_clusters = set()
        self._cluster_elements = []
        self._cluster_base_elements = []
        self.cluster_aliases = set()
        self.ambiguous = False
        self._disambiguator = {}
        self._disambiguated_from = set()
        self._alias_from = set()
        ClusterContainer._containers[(self.cluster_id, self.cluster_type)] = self

    def __eq__(self, other):
        return isinstance(other, ClusterContainer) and \
            self.cluster_id == other.cluster_id and \
            self.cluster_type == other.cluster_type

    def __hash__(self):
        return hash((self.cluster_id, self.cluster_type))

    def __len__(self):
        return len(self.get_elements())

    def __str__(self):
        return str((self.cluster_id, self.cluster_type))

    def link_clusters(self, cluster):
        self._linked_clusters.add(cluster)
        cluster._linked_clusters.add(self)

    def add_element(self, element):
        self._cluster_elements.append(element)

    def add_base_element(self, element):
        self._cluster_base_elements.append(element)

    def get_elements(self, calling_cluster=[], expand_link_only=False):
        elements = self._cluster_elements.copy()
        if self.ambiguous and not calling_cluster and not expand_link_only:
            elements = [element for element in elements if element.lower() not in self._disambiguator.keys()]
        for cluster in self._linked_clusters:
            if cluster not in calling_cluster:
                elements.extend(cluster.get_elements(calling_cluster + [self]))
        if not expand_link_only:
            for cluster in self._disambiguated_from:
                terms = [i for i, x in cluster._disambiguator.items() if x == self]
                tmp_elements = [item.lower() for item in cluster.get_elements(calling_cluster + [self])]
                tmp_elements = [term for term in tmp_elements if term in terms]
                elements.extend(tmp_elements)
            if not calling_cluster and self.cluster_aliases:
                for cluster in self.cluster_aliases:
                    elements.extend(cluster.get_elements(calling_cluster + [self]))
            for cluster in self._alias_from:
                if cluster not in calling_cluster:
                    elements.extend(cluster.get_elements(calling_cluster + [self]))
        return elements

    def get_base_elements(self):
        return self._cluster_base_elements

    def get_cluster_ids(self, term=None, return_object=False):
        if term is None and self.ambiguous:
            raise Exception("Ambiguous cluster, must specify term")
        elif term is not None:
            term = term.lower()
        if term in self._disambiguator:
            return self._disambiguator[term].get_cluster_ids(term, return_object)
        if not self.cluster_aliases:
            if not return_object:
                return [self.cluster_id]
            else:
                return [self]
        else:
            if not return_object:
                return [cluster.cluster_id for cluster in self.cluster_aliases]
            else:
                return [cluster for cluster in self.cluster_aliases]

    def set_ambiguity_term(self, term, cluster):
        lower_term = term.lower()
        if lower_term not in self._disambiguator:
            self.ambiguous = True
            self._disambiguator[lower_term] = cluster
            cluster._disambiguated_from.add(self)

    def add_aliases(self, clusters):
        self.cluster_aliases.update(clusters)
        for cluster in clusters:
            cluster._alias_from.add(self)


def get_cluster_container(cluster_id, cluster_type):
    if (cluster_id, cluster_type) not in ClusterContainer._containers:
        ClusterContainer(cluster_id, cluster_type)
    return ClusterContainer._containers[(cluster_id, cluster_type)]
