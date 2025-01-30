import base64
import json
import lzma
import os
import pickle
import requests
from datetime import timedelta
import time
from flask import jsonify
from collections import defaultdict
import plotly.graph_objects as go

from flask import Flask, render_template, request, session, send_file, after_this_request, redirect, url_for
from flask_paginate import Pagination, get_page_args

from Website.summary import ResultsSummarizer
from EvidenceConfig import PALLETE_COLORS
from EvidenceConfig import SOLR_CONFIG
from Website.QueryExpand import QueryExpansion
from Website.merge_facet import facetprocessor
from Website.pipeline import SentenceClassification, Model, NegationDriver, PropositionDriver, EvidenceMapDriver
from Website.visualization import *
from acquire_pubmed import upload_to_solr

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'Website', 'templates'),
            static_folder=os.path.join(os.getcwd(), 'Website', 'static'))
app.config['SECRET_KEY'] = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

facet = "facet.field=Sentence-level_breakdown.Evidence_Elements.Participant.term_str&\
facet.field=Sentence-level_breakdown.Evidence_Elements.Intervention.term_str&\
facet.field=Sentence-level_breakdown.Evidence_Elements.Outcome.term_str&\
facet=on&facet.mincount=1&facet.limit=50&facet.sort=count&"

PIO_types = ('Participant', 'Intervention', 'Outcome')


@app.route('/')
def home():
    session.clear()
    return render_template('search.html')


@app.route('/main_search')
def main_search():
    if not request.args.get('Query'):
        query = ''
    else:
        query = request.args.get('Query').strip().strip('"')

    print(f"Processing Query: {query}")
    solr_address = SOLR_CONFIG['base_url'] + SOLR_CONFIG['EvidenceMap_core'] + "/select?" + facet + "fl=*,score&q="
    prefix = "Sentence-level_breakdown.Evidence_Elements."
    PIO_input_dict = {}

    PIO_input_dict['Participant'] = []
    PIO_input_dict['Intervention'] = []
    PIO_input_dict['Outcome'] = []

    if query.isdigit():
        # PMID
        solr_query = solr_address + "doc_id:" + query
    elif len(query.split(' ')) <= 1:
        # Query Expansion
        expanded_query = query_expansion(query)
        if query not in expanded_query:
            expanded_query.append(query)

        solr_query = solr_address
        for p in PIO_types:
            temp_query = prefix + p + ".term:"
            temp_query += "("
            for word in expanded_query:
                temp_query = temp_query + "\"" + word + "\"" + " OR "
            temp_query = temp_query.rstrip(" OR ") + ")"
            solr_query += temp_query + " OR "
        solr_query = solr_query.rstrip(" OR ")
    else:
        # Question
        temp_query = ''

        parsed_query = parse(query)

        PIO_elements = {type: [] for type in PIO_types}

        nlp = Model.nlp

        for sentence in parsed_query:
            for entity in sentence['entities']:
                if entity['type'] in PIO_types:
                    PIO_elements[entity['type']].append(entity['text'])

        for p in PIO_types:
            PIO_input_dict[p] = []
            if PIO_elements[p]:
                for e in PIO_elements[p]:
                    if nlp(e.lower())[0].lemma_ == nlp(p.lower())[0].lemma_:
                        continue
                    else:
                        PIO_input_dict[p].append(" AND ")
                        PIO_input_dict[p].append(e)
                        temp_query += " AND "
                        # Query Expansion
                        term = e
                        expanded_query = query_expansion(term)
                        if term not in expanded_query:
                            expanded_query.append(term)
                        print(expanded_query)

                        expanded_temp_query = "("
                        for idx, word in enumerate(expanded_query):
                            expanded_temp_query += prefix + p + ".term:" + "\"" + word + "\""
                            if idx < len(expanded_query) - 1:
                                expanded_temp_query += " OR "
                        temp_query += expanded_temp_query + ")"
            if PIO_input_dict[p]:
                if (PIO_input_dict[p][0] == ' AND '):
                    PIO_input_dict[p][0] = ''

        if temp_query == '':
            solr_query = solr_address + "Sentence-level_breakdown.Text: " + query
        else:
            solr_query = solr_address + temp_query.lstrip(" AND ")

    session['Participant'] = PIO_input_dict['Participant']
    session['Intervention'] = PIO_input_dict['Intervention']
    session['Outcome'] = PIO_input_dict['Outcome']
    session['query'] = query.strip("\"")
    session['solr_query'] = solr_query

    return display_by_pages(query, solr_query, PIO_input_dict['Participant'], PIO_input_dict['Intervention'],
                            PIO_input_dict['Outcome'])


@app.route('/advanced_search')
def advanced_search():
    solr_address = SOLR_CONFIG['base_url'] + SOLR_CONFIG['EvidenceMap_core'] + "/select?" + facet + "fl=*,score&q="
    prefix = "Sentence-level_breakdown.Evidence_Elements."

    PIO_input_dict = {}
    search_query = []
    query_terms = []

    for p in PIO_types:
        PIO_input_dict[p] = []
        if request.args.get(p):
            temp_query = ''

            PIO_input_dict[p] = request.args.get(p).split(',')

            PIO_input_dict[p][0] = ''

            for i in range(0, len(PIO_input_dict[p]), 2):
                temp_query = "(" + temp_query
                if len(PIO_input_dict[p][i]) > 0:
                    temp_query += " " + PIO_input_dict[p][i].strip() + " "

                # Query expansion

                term = PIO_input_dict[p][i + 1].strip()

                query_terms.append(term)
                expanded_query = query_expansion(term)
                if term not in expanded_query:
                    expanded_query.append(term)
                print(expanded_query)
                expanded_temp_query = "("
                for idx, word in enumerate(expanded_query):
                    expanded_temp_query += prefix + p + ".term:" + "\"" + word + "\""
                    if idx < len(expanded_query) - 1:
                        expanded_temp_query += " OR "
                temp_query += expanded_temp_query + ")"

                temp_query += ")"

            search_query.append(temp_query)

    if not search_query:
        solr_query = solr_address + "*:*"
    else:
        search_query = " AND ".join(search_query)
        solr_query = solr_address + search_query

    session['Participant'] = PIO_input_dict['Participant']
    session['Intervention'] = PIO_input_dict['Intervention']
    session['Outcome'] = PIO_input_dict['Outcome']
    session['solr_query'] = solr_query

    query = ' '.join(query_terms)
    session['query'] = query

    return display_by_pages(query, solr_query, PIO_input_dict['Participant'], PIO_input_dict['Intervention'],
                            PIO_input_dict['Outcome'])


@app.route('/advanced_search_page')
def advanced():
    Participant_list = session.get('Participant')
    Intervention_list = session.get('Intervention')
    Outcome_list = session.get('Outcome')
    return render_template('advanced.html', Participant_list=Participant_list, Intervention_list=Intervention_list,
                           Outcome_list=Outcome_list, color_pallette=PALLETE_COLORS)


@app.route('/introduction')
def advanintroductionced():
    return render_template('introduction.html')


def reparse_doc(doc, query_args):
    try:
        pipeline_start = query_args['parsepoint']
    except:
        pipeline_start = 'map_construction'

    print(f"Reparse Query Args: {query_args}")

    for key in query_args.keys():
        if key in ['SentenceClassification', 'Models', 'Negations', 'Propositions', 'EvidenceMap']:
            for argument, value in query_args[key].items():
                if key == 'SentenceClassification':
                    SentenceClassification.driver_config[argument] = value
                elif key == 'Models':
                    Model.driver_config[argument] = value
                elif key == 'Negations':
                    NegationDriver.driver_config[argument] = value
                elif key == 'Propositions':
                    PropositionDriver.driver_config[argument] = value
                elif key == 'EvidenceMap':
                    EvidenceMapDriver.driver_config[argument] = value

    metadata, predictions, pmid, abstract_text, has_complete_entities, has_complete_negations, has_complete_propositions, has_complete_tags, has_complete_map = EvidenceMapDriver.deconstruct_json(
        doc)

    if pipeline_start == 'sent_classification':
        has_complete_tags = False
        has_complete_entities = False
        has_complete_negations = False
        has_complete_propositions = False
        has_complete_map = False
    elif pipeline_start == 'entity_recognition':
        has_complete_entities = False
        has_complete_propositions = False
        has_complete_negations = False
        has_complete_map = False
    elif pipeline_start == 'negation_detection':
        has_complete_negations = False
        has_complete_propositions = False
        has_complete_map = False
    elif pipeline_start == 'proposition_extraction':
        has_complete_propositions = False
        has_complete_map = False
    elif pipeline_start == 'map_construction':
        has_complete_map = False

    if not has_complete_tags:
        predictions = Model.predictBodyOfText(abstract_text, flatten=True, split_newlines=True,
                                              SentenceClassificationDriver=SentenceClassification)
        has_complete_tags = True
        has_complete_entities = True
        has_complete_negations = False
        has_complete_propositions = False
        has_complete_map = False
    elif not has_complete_entities or not has_complete_propositions:
        predictions_tmp = Model.predictBodyOfText(abstract_text, flatten=True, split_newlines=True)

        # Re-align sentence tags to the new predictions
        i = 0
        j = 0

        while i < len(predictions) and j < len(predictions_tmp):
            sentence = predictions[i]['tokens']
            sentence = ''.join(sentence)
            sentence_tmp = predictions_tmp[j]['tokens']
            sentence_tmp = ''.join(sentence_tmp)
            repeat_flag = False

            if sentence == sentence_tmp:
                predictions_tmp[j]['tag'] = predictions[i]['tag']
                i += 1
                j += 1
                repeat_flag = False
            elif not sentence:
                i += 1
            elif not sentence_tmp:
                j += 1
                repeat_flag = False
            elif sentence in sentence_tmp:
                if not repeat_flag:
                    predictions_tmp[j]['tag'] = predictions[i]['tag']
                    repeat_flag = True
                i += 1
            elif sentence_tmp in sentence:
                predictions_tmp[j]['tag'] = predictions[i]['tag']
                j += 1
                repeat_flag = False
            else:
                print("Reparse Warn: Possible reconciliation error, article should be processed from scratch." )
                predictions_tmp[j]['tag'] = predictions[i]['tag']
                i += 1
                j += 1
                repeat_flag = False

            if i == len(predictions) or j == len(predictions_tmp):
                predictions = predictions_tmp
                has_complete_entities = True
                has_complete_negations = False
                has_complete_propositions = False
                has_complete_map = False

    if not has_complete_negations:
        predictions = NegationDriver.detectNegations(predictions)
        has_complete_negations = True

        if has_complete_propositions:
            for prediction in predictions:
                for proposition in prediction['propositions']:
                    if proposition['Observation'] is not None:
                        proposition['negation'] = proposition['Observation']['negation_status']
                    elif proposition['Count'] is not None:
                        proposition['negation'] = proposition['Count']['negation_status']

    if not has_complete_propositions:
        predictions = PropositionDriver.buildPropositions(predictions)
        has_complete_propositions = True

    if not has_complete_map:
        start_overall = time.perf_counter()
        start_cluster = time.perf_counter()
        predictions, proposed_arms = EvidenceMapDriver.fit_propositions(predictions, print_output=True)
        cluster_time = time.perf_counter() - start_cluster
        ([participants, _, _], EvidenceMap) = EvidenceMapDriver.build_map(predictions, proposed_arms, print_output=True)
        json = EvidenceMapDriver.build_json(participants, EvidenceMap, metadata, predictions, pmid, abstract_text)
        overall_time = time.perf_counter() - start_overall

        print( f"Level 3: Average clustering time: {cluster_time:.4f} ms/abstract")
        print( f"Level 3: Average overall time: {overall_time:.4f} ms/abstract")

        solr_url = SOLR_CONFIG['base_url'] + SOLR_CONFIG['EvidenceMap_core']
        upload_to_solr(solr_url, json)
        has_complete_map = True
        commit_response = requests.get(f'{solr_url}/update?commit=true')
        if commit_response.status_code != 200:
            print(f'Commit failed with status code {commit_response.status_code}: {commit_response.text}')

    return doc


@app.route('/visualize/<string:id>', methods=['GET', 'POST'])
def visualize(id, no_args=False):
    Participant_list = session.get('Participant') or ""
    Intervention_list = session.get('Intervention') or ""
    Outcome_list = session.get('Outcome') or ""

    query = session.get('query')

    if query is None:
        query = ''

    solr_address = SOLR_CONFIG['base_url'] + SOLR_CONFIG['EvidenceMap_core'] + "/select?rows=100000&fl=*,score&q="
    solr_query = solr_address + "doc_id:" + id

    print(f"Issuing Solr Query: {solr_query}")

    myobj = {'somekey': 'somevalue'}
    response = requests.post(solr_query + "&fl=numFound", data=myobj)
    results = json.loads(response.text)
    docs = results['response']['docs']
    doc = docs[0]

    action_arg = request.args.get('action')

    original_doc = solr_unflatten(doc)

    if not no_args:
        if action_arg == 'reparse':
            params = {}
            for key in request.args.keys():
                value = request.args.get(key)
                if '.' in key:
                    primary, secondary = key.split('.', 1)
                    if primary not in params:
                        params[primary] = {}
                    params[primary][secondary] = value
                elif key in ['parsepoint']:
                    params[key] = value

            reparse_doc(original_doc, params)

            return visualize(id, True)

    data = clean_data(original_doc)
    level3_data, num_arm, level3_provider = extract_level3(original_doc)

    temp = {}
    temp['doc_id'] = id
    temp['title'] = doc['title'][0].strip('.')
    if 'publication_metadata.pubdate' in doc:
        temp['pubdate'] = doc['publication_metadata.pubdate'][0]
    else:
        temp['pubdate'] = ""

    if 'publication_metadata.source' in doc:
        temp['source'] = doc['publication_metadata.source'][0]
    else:
        temp['source'] = ""

    if 'publication_metadata.volume' in doc:
        temp['volume'] = doc['publication_metadata.volume'][0]
    else:
        temp['volume'] = ""

    if 'publication_metadata.pages' in doc:
        temp['pages'] = doc['publication_metadata.pages'][0]
    else:
        temp['pages'] = ""

    if 'publication_metadata.pubtype' in doc:
        temp['pubtype'] = doc['publication_metadata.pubtype']
    else:
        temp['pubtype'] = []

    if 'publication_metadata.authors' in doc:
        temp['authors'] = doc['publication_metadata.authors']
    else:
        temp['authors'] = []

    return render_template('annotation.html', query=query, data=data, metadata=temp, Participant=Participant_list, \
                           Intervention=Intervention_list, Outcome=Outcome_list, level3_data=level3_data,
                           num_arm=num_arm, color_pallette=PALLETE_COLORS, level3_provider=level3_provider)



@app.route('/download/<string:id>', methods=['GET', 'POST'])
def download(id):
    @after_this_request
    def remove_file(response):
        try:
            os.remove(file_name)
        except Exception as error:
            app.logger.error("Error removing or closing downloaded file handle", error)
        return response

    solr_address = SOLR_CONFIG['base_url'] + SOLR_CONFIG['EvidenceMap_core'] + "/select?rows=100000&fl=*,score&q="
    solr_query = solr_address + "doc_id:" + id

    myobj = {'somekey': 'somevalue'}
    response = requests.post(solr_query + "&fl=numFound", data=myobj)
    results = json.loads(response.text)
    docs = results['response']['docs']
    doc = docs[0]

    original_doc = solr_unflatten(doc)
    output = json.dumps(original_doc, indent=4)

    file_name = id + '.json'
    file_path = os.path.join(os.path.dirname(__file__), file_name)

    with open(file_path, 'w') as file:
        file.write(output)

    return send_file(file_path, as_attachment=True)


@app.route('/sort_results')
def sort_results():
    option = request.args.get('sort_option')
    print(option)

    query = session.get('query')
    solr_query = session.get('solr_query')

    Participant_list = session.get('Participant')
    Intervention_list = session.get('Intervention')
    Outcome_list = session.get('Outcome')
    return display_by_pages(query, solr_query, Participant_list, Intervention_list, Outcome_list, option)


def query_expansion(term):
    QE = QueryExpansion()
    qs = QE.expand(term)
    return qs


def display_by_pages(query, solr_query, Participant_list, Intervention_list, Outcome_list, option="best"):
    print(f"Issuing Solr Query: {solr_query}")

    myobj = {'somekey': 'somevalue'}
    response = requests.post(solr_query + "&fl=numFound", data=myobj)

    results = json.loads(response.text)
    if results['responseHeader']['status'] == 400:
        response = requests.post(solr_query)
        results = json.loads(response.text)

    numFound = results['response']['numFound']
    print(f"Number of Results: {numFound}")

    # max_facets=1000
    facet_Participant = []
    facet_Intervention = []
    facet_Outcome = []

    facet_term_Participant = []
    facet_term_Intervention = []
    facet_term_Outcome = []

    facet_terms = results['facet_counts']['facet_fields']
    if facet_terms["Sentence-level_breakdown.Evidence_Elements.Participant.term_str"]:
        facet_term_Participant, facet_Participant = facet_normalize(
            facet_terms["Sentence-level_breakdown.Evidence_Elements.Participant.term_str"], 'Participant')

    if facet_terms["Sentence-level_breakdown.Evidence_Elements.Intervention.term_str"]:
        facet_term_Intervention, facet_Intervention = facet_normalize(
            facet_terms["Sentence-level_breakdown.Evidence_Elements.Intervention.term_str"], 'Intervention')

    if facet_terms["Sentence-level_breakdown.Evidence_Elements.Outcome.term_str"]:
        facet_term_Outcome, facet_Outcome = facet_normalize(
            facet_terms["Sentence-level_breakdown.Evidence_Elements.Outcome.term_str"], 'Outcome')

    if numFound <= 1:
        num_of_results = str(numFound) + ' ' + 'result'
    else:
        num_of_results = str(numFound) + ' ' + 'results'

    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')

    print(f"Issuing Solr Query: {solr_query}&start={offset}&rows={per_page}")
    response = requests.post(solr_query + "&start=" + str(offset) + "&rows=" + str(per_page), data=myobj)
    results = json.loads(response.text)
    display_results = []
    docs = results['response']['docs']

    # If we only have one result, go straight to the visualization page
    if numFound == 1:
        return redirect(url_for('visualize', id=docs[0]['doc_id']))
    
    
    for doc in docs:
    
        temp = {}
        doc_id = str(doc['doc_id'])
        temp['doc_id'] = doc_id

        temp['title'] = doc['publication_metadata.title'][0].strip('.')

        if 'publication_metadata.pubdate' in doc:
            temp['pubdate'] = doc['publication_metadata.pubdate'][0]
        else:
            temp['pubdate'] = ""

        if 'publication_metadata.source' in doc:
            temp['source'] = doc['publication_metadata.source'][0]
        else:
            temp['source'] = ""

        if 'publication_metadata.volume' in doc:
            temp['volume'] = doc['publication_metadata.volume'][0]
        else:
            temp['volume'] = ""

        if 'publication_metadata.pages' in doc:
            temp['pages'] = doc['publication_metadata.pages'][0]
        else:
            temp['pages'] = ""

        if 'publication_metadata.pubtype' in doc:
            temp['pubtype'] = doc['publication_metadata.pubtype']
        else:
            temp['pubtype'] = []

        if 'publication_metadata.authors' in doc:
            temp['authors'] = doc['publication_metadata.authors']
        else:
            temp['authors'] = []

        if 'Sentence-level_breakdown.Text' in doc:
            temp['abstract'] = abstract_format(doc['Sentence-level_breakdown.Text'])
        else:
            temp['abstract'] = ''

        display_results.append(temp)

    pagination = Pagination(page=page, per_page=per_page, total=numFound,
                            css_framework='bootstrap4', inner_window=3)

    facet_dict = {}
    facet_dict['facet_term_Participant'] = facet_term_Participant
    facet_dict['facet_term_Intervention'] = facet_term_Intervention
    facet_dict['facet_term_Outcome'] = facet_term_Outcome
    facet_file_name = ''.join(e for e in query if e.isalnum())
    dir_path = os.path.join('Website', 'resources')
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{facet_file_name}.pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(facet_dict, file)
    
    summarizer = ResultsSummarizer()
    results_summary = summarizer.generate_summary(display_results)
    docs = fetch_all_documents(solr_query)
    aggregated_data = aggregate_evidence_data(docs)
    collective_map = generate_collective_visualization(aggregated_data)

    return render_template('result_page.html', query=query.strip("\""), display_results=display_results,
                           results_summary=results_summary,collective_map=collective_map, num_of_results=num_of_results,
                           Participant=Participant_list, Intervention=Intervention_list, Outcome=Outcome_list,
                           page=page, per_page=per_page, pagination=pagination, facet_Participant=facet_Participant,
                           facet_Intervention=facet_Intervention, facet_Outcome=facet_Outcome,
                           option=option, color_pallette=PALLETE_COLORS)


@app.route('/filter_results')
def filter_results():
    query = session.get('query')
    solr_query = session.get('solr_query')
    print(query)
    facet_file_name = ''.join(e for e in query if e.isalnum())

    dir_path = os.path.join('Website', 'resources')
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{facet_file_name}.pkl")
    with open(file_path, 'rb') as file:
        facet_dict = pickle.load(file)

    for p in PIO_types:
        temp = ''
        if request.args.getlist('checked_facet_' + p):
            checked_facet = set(request.args.getlist('checked_facet_' + p))
            facet_term = set(facet_dict['facet_term_' + p].keys())
            for term in checked_facet:
                expanded_query = query_expansion(term)
                if term not in expanded_query:
                    expanded_query.append(term)
                expanded_temp_query = "("
                for idx, word in enumerate(expanded_query):
                    expanded_temp_query += "\"" + word + "\""
                    if idx < len(expanded_query) - 1:
                        expanded_temp_query += " OR "
                temp += expanded_temp_query + ")"
                solr_query = solr_query + " AND Sentence-level_breakdown.Evidence_Elements." + p + ".term:" + temp

    Participant_list = session.get('Participant')
    Intervention_list = session.get('Intervention')
    Outcome_list = session.get('Outcome')
    session['solr_query'] = solr_query

    return display_by_pages(query, solr_query, Participant_list, Intervention_list, Outcome_list)


def facet_normalize(original_facet_list, PIO_type):
    # We might want to replace this with the implemented EvidenceMap clustering algorithm
    processor = facetprocessor([], [], [])
    if PIO_type == 'Participant' or 'Outcome':
        merged_facet_sorted, merged_facet = processor.process_p_o(original_facet_list)
    else:
        merged_facet_sorted, merged_facet = processor.process_intervention(original_facet_list)

    return merged_facet, merged_facet_sorted


def abstract_format(abstract, max_length=350):
    abstract_all = ''
    if len(abstract) <= 1:
        return 'none'
    elif 1 < len(abstract) < 4:
        for sentence in abstract[1:]:
            abstract_all += sentence
    else:
        for i in range(1, 4):
            abstract_all += abstract[i]
    if len(abstract_all) <= max_length:
        return abstract_all
    else:
        while True:
            if (abstract_all[max_length] == ' ') or (abstract_all[max_length] == '.'):
                break
            else:
                max_length = max_length - 1
        return abstract_all[0:max_length]


def parse(input_text):
    abstract_text = [input_text]
    _, predictions = Model.predictBodyOfText(abstract_text, flatten=True, split_newlines=True)

    return predictions


def solr_unflatten(json_obj):
    payload = json_obj["payload"][0]
    json_obj = json.loads(lzma.decompress(base64.b64decode(payload.encode())))
    return json_obj


@app.route('/get_collective_map')
def get_collective_map():
    # Get the current solr query from session
    solr_query = session.get('solr_query')
    if not solr_query:
        solr_query = SOLR_CONFIG['base_url'] + SOLR_CONFIG['EvidenceMap_core'] + "/select?rows=100000&fl=*,score&q=*:*"
    
    # Fetch documents
    print(f"Current Solr Query: {solr_query}")
    docs = fetch_all_documents(solr_query)
    print(f"Number of documents fetched: {len(docs)}")
    
    # Aggregate evidence data
    aggregated_data = aggregate_evidence_data(docs)
    
    # Create node lists
    participants = list(aggregated_data['participants'].keys())
    interventions = list(aggregated_data['interventions'].keys())
    outcomes = list(aggregated_data['outcomes'].keys())
    
    # Create node labels and colors
    all_labels = participants + interventions + outcomes
    node_colors = (
        ['#1f77b4'] * len(participants) +  # Blue for participants
        ['#ff7f0e'] * len(interventions) +  # Orange for interventions
        ['#2ca02c'] * len(outcomes)         # Green for outcomes
    )
    
    # Create links
    sources = []
    targets = []
    values = []
    
    # Connect participants to interventions
    for i, p in enumerate(participants):
        for j, inv in enumerate(interventions):
            if aggregated_data['participants'][p] > 0:
                sources.append(i)
                targets.append(len(participants) + j)
                values.append(aggregated_data['participants'][p])
    
    # Connect interventions to outcomes
    for i, inv in enumerate(interventions):
        for j, out in enumerate(outcomes):
            if aggregated_data['interventions'][inv] > 0:
                sources.append(len(participants) + i)
                targets.append(len(participants) + len(interventions) + j)
                values.append(aggregated_data['interventions'][inv])
    
    data = {
        'data': [{
            'type': 'sankey',
            'node': {
                'pad': 15,
                'thickness': 20,
                'line': {'color': 'black', 'width': 0.5},
                'label': all_labels,
                'color': node_colors
            },
            'link': {
                'source': sources,
                'target': targets,
                'value': values
            }
        }],
        'layout': {
            'title': 'Evidence Flow Diagram',
            'font': {'size': 12},
            'width': 800,
            'height': 600
        }
    }
    
    return jsonify(data)

def fetch_all_documents(solr_query):
    myobj = {'somekey': 'somevalue'}
    response = requests.post(solr_query + "&rows=1000", data=myobj)
    results = json.loads(response.text)
    
    if 'response' not in results:
        print(f"Solr Query: {solr_query}")
        print(f"Response: {results}")
        return []
        
    return results['response']['docs']

def aggregate_evidence_data(docs):
    aggregated_data = defaultdict(lambda: defaultdict(int))
    
    for doc in docs:
        level3_data, _, _ = extract_level3(solr_unflatten(doc))
        
        # Extract participants
        for participant in level3_data.get('participants', []):
            if not participant.get('negation', False):  # Only include non-negated terms
                aggregated_data['participants'][participant['term']] += 1
        
        # Extract interventions from study_design and study_results
        for item in level3_data.get('study_design', []) + level3_data.get('study_results', []):
            if item.get('type') == 'Intervention':
                aggregated_data['interventions'][item['term']] += 1
                
        # Extract outcomes from study_design and study_results
        for item in level3_data.get('study_design', []) + level3_data.get('study_results', []):
            if item.get('type') == 'Outcome':
                aggregated_data['outcomes'][item['term']] += 1
    
    return aggregated_data

from difflib import SequenceMatcher
from nltk.corpus import wordnet

def calculate_similarity(term1, term2):
    # String similarity
    string_similarity = SequenceMatcher(None, term1.lower(), term2.lower()).ratio()
    
    # WordNet similarity
    try:
        synsets1 = wordnet.synsets(term1)
        synsets2 = wordnet.synsets(term2)
        if synsets1 and synsets2:
            wordnet_similarity = synsets1[0].path_similarity(synsets2[0]) or 0
        else:
            wordnet_similarity = 0
    except:
        wordnet_similarity = 0
    
    # Combined similarity score
    return max(string_similarity, wordnet_similarity)

def generate_collective_visualization(aggregated_data):

    def calculate_relevance_score(term, count, query_terms):
        # Higher score for terms that match query terms
        query_relevance = sum(calculate_similarity(term, q_term) for q_term in query_terms)
        # Combine with frequency count
        return (query_relevance * 0.7) + (count * 0.3)  # Weighted combination
    
    # Get query terms from session
    query = session.get('query', '').lower().split()
    
    # Score and sort elements by relevance
    TOP_N = 2
    
    # Get most relevant items from each category
    participants = dict(sorted(
        aggregated_data['participants'].items(),
        key=lambda x: calculate_relevance_score(x[0], x[1], query),
        reverse=True
    )[:TOP_N])
    
    interventions = dict(sorted(
        aggregated_data['interventions'].items(),
        key=lambda x: calculate_relevance_score(x[0], x[1], query),
        reverse=True
    )[:TOP_N])
    
    outcomes = dict(sorted(
        aggregated_data['outcomes'].items(),
        key=lambda x: calculate_relevance_score(x[0], x[1], query),
        reverse=True
    )[:TOP_N])
    # Create nodes for each category
    nodes = []
    node_colors = []
    
    # Add participants (blue)
    participant_nodes = list(participants.keys())
    nodes.extend(participant_nodes)
    node_colors.extend(['#1f77b4'] * len(participant_nodes))
    
    # Add interventions (orange)
    intervention_nodes = list(interventions.keys())
    nodes.extend(intervention_nodes)
    node_colors.extend(['#ff7f0e'] * len(intervention_nodes))
    
    # Add outcomes (green)
    outcome_nodes = list(outcomes.keys())
    nodes.extend(outcome_nodes)
    node_colors.extend(['#2ca02c'] * len(outcome_nodes))
    
    # Create connections
    sources = []
    targets = []
    values = []
    
    # Connect participants to interventions
    for p_idx, participant in enumerate(participant_nodes):
        for i_idx, intervention in enumerate(intervention_nodes):
            sources.append(p_idx)
            targets.append(len(participant_nodes) + i_idx)
            values.append(aggregated_data['participants'][participant])
    
    # Connect interventions to outcomes
    for i_idx, intervention in enumerate(intervention_nodes):
        for o_idx, outcome in enumerate(outcome_nodes):
            sources.append(len(participant_nodes) + i_idx)
            targets.append(len(participant_nodes) + len(intervention_nodes) + o_idx)
            values.append(aggregated_data['interventions'][intervention])
    
    return {
        'data': [{
            'type': 'sankey',
            'node': {
                'pad': 30,
                'thickness': 20,
                'line': {'color': 'black', 'width': 0.5},
                'label': nodes,
                'color': node_colors
            },
            'link': {
                'source': sources,
                'target': targets,
                'value': values
            }
        }],
        'layout': {
            'title': 'Combined Evidence Flow',
            'font': {'size': 14},
            'width': 1200,
            'height': 600,
            'margin': {'t': 40, 'b': 40, 'l': 50, 'r': 50}
        }
    }

print("Registered Routes:")
for rule in app.url_map.iter_rules():
    print(f"{rule.endpoint}: {rule.rule}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
