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
from datetime import datetime

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
    
    # Process start and end dates if provided in the search form
    
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

def parse_date(date_str):
    if not date_str:
        return None
    formats = ['%m/%d/%Y', '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y']
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    try:
        # Try to extract a 4-digit year
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            year = int(year_match.group(0))
            return datetime(year, 1, 1)
    except:
        pass

    return None
            


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
                print("key and value is", key,value)
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

@app.route('/download_collective', methods=['GET'])
def download_collective():
    @after_this_request
    def remove_file(response):
        try:
            os.remove(file_path)
        except Exception as error:
            app.logger.error("Error removing or closing downloaded file handle", error)
        return response

    solr_query = session.get('solr_query')
    
    if not solr_query:
        solr_query = SOLR_CONFIG['base_url'] + SOLR_CONFIG['EvidenceMap_core'] + "/select?rows=100000&fl=*,score&q=*:*"
    
    myobj = {'somekey': 'somevalue'}
    response = requests.post(solr_query, data=myobj)
    results = json.loads(response.text)
    docs = results['response'].get('docs', [])
    
    start_date = session.get('start_date')
    end_date = session.get('end_date')
    
    filtered_docs = []
    for doc in docs:
        include_doc = True
        
        if 'publication_metadata.pubdate' in doc:
            pub_date_str = doc['publication_metadata.pubdate'][0]
            pub_date = parse_date(pub_date_str)
            
            if pub_date and start_date and end_date:
                start = parse_date(start_date)
                end = parse_date(end_date)
                
                if start and end:
                    if not (start <= pub_date <= end):
                        include_doc = False
            elif pub_date and start_date:
                start = parse_date(start_date)
                if start and pub_date < start:
                    include_doc = False
            elif pub_date and end_date:
                end = parse_date(end_date)
                if end and pub_date > end:
                    include_doc = False
        
        if include_doc:
            filtered_docs.append(doc)
    
    combined_data = []
    for doc in filtered_docs:
        original_doc = solr_unflatten(doc)
        combined_data.append(original_doc)
    
    query = session.get('query', 'collective_data')
    file_name = ''.join(e for e in query if e.isalnum()) + '_collective.json'
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    
    with open(file_path, 'w') as file:
        json.dump(combined_data, file, indent=4)
    
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
    myobj = {'somekey': 'somevalue'}
    response = requests.post(solr_query + "&fl=numFound", data=myobj)
    print(f"Solr Response Status Code: {response.status_code}")
    print(f"Solr Response Status Code: {response.text}")
    results = json.loads(response.text)
    print("The result is --", results)
    if results['responseHeader']['status'] == 400:
        response = requests.post(solr_query)
        results = json.loads(response.text)
       

    numFound = results['response']['numFound']
    print(f"Number of Results: {numFound}")
    start_date = session.get('start_date')
    end_date = session.get('end_date')

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

    response = requests.post(solr_query + "&start=" + str(offset) + "&rows=" + str(per_page), data=myobj)
    results = json.loads(response.text)
    display_results = []
    docs = results['response']['docs']
    
    combined_clean_data = []
    filtered_docs = []
    for doc in docs:
        include_doc = True
        if 'publication_metadata.pubdate' in doc:
            pub_date_str = doc['publication_metadata.pubdate'][0]
            pub_date = parse_date(pub_date_str)
            
            if pub_date and start_date and end_date:
                start = parse_date(start_date)
                end = parse_date(end_date)
                
                if start and end:
                    if not (start <= pub_date <= end):
                        include_doc = False
            elif pub_date and start_date:
                start = parse_date(start_date)
                if start and pub_date < start:
                    include_doc = False
            elif pub_date and end_date:
                end = parse_date(end_date)
                if end and pub_date > end:
                    include_doc = False
        
        if include_doc:
            filtered_docs.append(doc)

       
    filtered_count = len(filtered_docs)
    if filtered_count == 0:
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
            
        num_of_results = "0 results"
        if start_date or end_date:
            num_of_results += f" (filtered from {numFound})"
            
        pagination = Pagination(page=page, per_page=per_page, total=0,
                            css_framework='bootstrap4', inner_window=3)
                            
        return render_template('result_page.html', query=query.strip("\""), display_results=[],
                           summaries=[], 
                           num_of_results=num_of_results,
                           Participant=Participant_list, Intervention=Intervention_list, Outcome=Outcome_list,
                           page=page, per_page=per_page, pagination=pagination, facet_Participant=facet_Participant,
                           facet_Intervention=facet_Intervention, facet_Outcome=facet_Outcome,
                           option=option, color_pallette=PALLETE_COLORS, start_date=start_date, end_date=end_date)
    if filtered_count == 1:
        return redirect(url_for('visualize', id=filtered_docs[0]['doc_id']))
    
    if filtered_count <= 1:
        num_of_results = str(filtered_count) + ' ' + 'result'
    else:
        num_of_results = str(filtered_count) + ' ' + 'results'
        if (start_date or end_date) and filtered_count != numFound:
            num_of_results += f" (filtered from {numFound})"
        
        for doc in filtered_docs:

            original_doc = solr_unflatten(doc)
            clean_data_result = clean_data(original_doc)
            combined_clean_data.append(clean_data_result)
    
            temp = {}
            doc_id = str(doc['doc_id'])
            temp['doc_id'] = doc_id

            temp['title'] = doc['publication_metadata.title'][0].strip('.')

            if 'publication_metadata.pubdate' in doc:
                temp['pubdate'] = doc['publication_metadata.pubdate'][0]
                print(f"Extracted date for doc {doc_id}: {temp['pubdate']}")
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

        summary_data = {
            'clean_data_combined' : combined_clean_data
        }
        summarizer = ResultsSummarizer()
        results_summary = summarizer.generate_summary(summary_data, display_results, query, filtered_docs)
    
        return render_template('result_page.html', query=query.strip("\""), display_results=display_results,summaries=results_summary['summaries']['ExtractedSummary'],
                           num_of_results=num_of_results,model_summaries = results_summary['summaries']['ModelSumm'], final_summaries = results_summary['summaries']['FinalSumm'],
                           Participant=Participant_list, Intervention=Intervention_list, Outcome=Outcome_list,
                           page=page, per_page=per_page, pagination=pagination, facet_Participant=facet_Participant,
                           facet_Intervention=facet_Intervention, facet_Outcome=facet_Outcome,
                           option=option, color_pallette=PALLETE_COLORS, start_date = start_date, end_date = end_date)


from datetime import datetime

def format_date_for_input(date_str):
    """Convert MM/DD/YYYY to YYYY-MM-DD for HTML date input."""
    if not date_str:
        return ""

    try:
        date_obj = datetime.strptime(date_str, '%m/%d/%Y')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return date_str

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

    start_date = session.get('start_date')
    end_date = session.get('end_date')

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
    print(f"Parsing query of length: {len(input_text)}")
    abstract_text = [input_text]
    _, predictions = Model.predictBodyOfText(abstract_text, flatten=True, split_newlines=True)

    return predictions


def solr_unflatten(json_obj):
    payload = json_obj["payload"][0]
    json_obj = json.loads(lzma.decompress(base64.b64decode(payload.encode())))
    return json_obj


import math
@app.route('/visualize/collective', methods=['GET', 'POST'])
def collective_visualize(no_args=False):
    try:
        
        solr_query = session.get('solr_query')
        
        if not solr_query:
            solr_query = SOLR_CONFIG['base_url'] + SOLR_CONFIG['EvidenceMap_core'] + "/select?rows=100000&fl=*,score&q=*:*"
        
        page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter="per_page")
        
        solr_query = f"{solr_query}&start={offset}&rows={per_page}"
        # Add reparse handling
        myobj = {'somekey': 'somevalue'}
        response = requests.post(solr_query + "&fl=numFound", data=myobj)
        
        results = json.loads(response.text)
        docs = results['response'].get('docs', [])

        if not docs:
            return jsonify({'data': [], 'providers': []})
        
        start_date = session.get('start_date')
        end_date = session.get('end_date')

        combined_level3_data = []
        combined_clean_data = []
        level3_provider = []
        action_arg = request.args.get('action')

        if not no_args and action_arg == 'reparse':
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
            for doc in docs:
                original_doc = solr_unflatten(doc)
                reparse_doc(original_doc, params)
            return collective_visualize(True)

        filtered_docs = []
        for doc in docs:
            include_doc = True
            
            if 'publication_metadata.pubdate' in doc:
                pub_date_str = doc['publication_metadata.pubdate'][0]
                pub_date = parse_date(pub_date_str)
                
                if pub_date and start_date and end_date:
                    start = parse_date(start_date)
                    end = parse_date(end_date)
                    
                    if start and end:
                        if not (start <= pub_date <= end):
                            include_doc = False
                elif pub_date and start_date:
                    start = parse_date(start_date)
                    if start and pub_date < start:
                        include_doc = False
                elif pub_date and end_date:
                    end = parse_date(end_date)
                    if end and pub_date > end:
                        include_doc = False
            
            if include_doc:
                filtered_docs.append(doc)


        for doc in filtered_docs:
            original_doc = solr_unflatten(doc)
            clean_data_result = clean_data(original_doc)
            combined_clean_data.append(clean_data_result)
            level3_data, num_arm, level3_provider = extract_level3(original_doc)
            if level3_data:
                combined_level3_data.append(level3_data)
        filtered_count = len(filtered_docs)
        total_pages = math.ceil(filtered_count / per_page)
        pagination_info = {
            'current_page': page,
            'total_pages': total_pages,
            'per_page': per_page,
            'total_items': filtered_count,
            'original_total': results['response']['numFound']
        }

        return jsonify({
            'data': combined_level3_data,
            'providers': level3_provider,
            'clean_data_combined': combined_clean_data,
            'pagination': pagination_info,
            'date_filter': {
                'start_date': start_date,
                'end_date': end_date
            }
        })
    except Exception as e:
        print(f"Error in collective_visualize: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/filter_by_date' ,methods=['GET', 'POST'])
def filter_by_date():
    try:
        if request.method == 'POST':
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
        else:
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
        
        print(f"Date values: start_date={start_date}, end_date={end_date}")
        
        session['start_date'] = start_date
        session['end_date'] = end_date

        # Redirect to main search with the same query
        query = session.get('query', '')
        solr_query = session.get('solr_query')
        
        print(f"Session values: query={query}, solr_query={solr_query}")
        
        # If solr_query is None, create a default one
        if solr_query is None:
            solr_address = SOLR_CONFIG['base_url'] + SOLR_CONFIG['EvidenceMap_core'] + "/select?" + facet + "fl=*,score&q="
            solr_query = solr_address + "*:*"
            session['solr_query'] = solr_query
            print(f"Created default solr_query: {solr_query}")

        Participant_list = session.get('Participant', [])
        Intervention_list = session.get('Intervention', [])
        Outcome_list = session.get('Outcome', [])
        
        print(f"PIO lists: P={len(Participant_list) if Participant_list else 0}, I={len(Intervention_list) if Intervention_list else 0}, O={len(Outcome_list) if Outcome_list else 0}")

        # Make sure we're calling display_by_pages correctly and capturing its return value
        result = display_by_pages(query, solr_query, Participant_list, Intervention_list, Outcome_list)
        print(f"display_by_pages returned: {type(result)}")
        
        # Ensure we're returning something valid
        if result is None:
            print("WARNING: display_by_pages returned None, redirecting to home")
            return redirect(url_for('home'))
            
        return result

    except Exception as e:
        print(f"Error in filter_by_date: {str(e)}")
        import traceback
        traceback.print_exc()
        # In case of error, redirect to home page
        return redirect(url_for('home'))

@app.route('/clear_date_filter')
def clear_date_filter():
        if 'start_date' in session:
            session.pop('start_date')
        if 'end_date' in session:
            session.pop('end_date')
        
        query = session.get('query', '')
        solr_query = session.get('solr_query')

        Participant_list = session.get('Participant')
        Intervention_list = session.get('Intervention')
        Outcome_list = session.get('Outcome')

        return display_by_pages(query, solr_query, Participant_list, Intervention_list, Outcome_list)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

