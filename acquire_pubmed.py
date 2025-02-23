import argparse
import datetime
import json
import os
import pickle
import shutil
import time
import urllib.parse
import urllib.request as ur
import xml.etree.ElementTree as xml_parser
from time import sleep

import pytz
import regex as re
import requests
import torch
from tqdm import tqdm

from EvidenceConfig import SOLR_CONFIG

DOWNLOAD_TRACKER_FILE = 'download_tracker.json'

SECONDS_IN_A_DAY = 86400


def move_to_cpu(item):
    if isinstance(item, torch.Tensor):
        return item.cpu()
    elif isinstance(item, dict):
        return {k: move_to_cpu(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [move_to_cpu(i) for i in item]
    elif isinstance(item, tuple):
        return tuple(move_to_cpu(i) for i in item)
    else:
        return item

def log_pmid_exception(pmid, e):
    file_path = os.path.join(output_dir, 'PubMed_errors.txt')
    with open(file_path, 'a') as f:
        f.write(f"PMID: {pmid}, Error: {e}\n")
    print(f"[PubMed Processor]: This PMID has been logged under PubMed_errors.txt")

def save_batch_to_file(output, meta_out, pmids):
    current_timestamp = int(time.time())
    os.makedirs(os.path.join(output_dir, 'raw_pubmed_unprocessed'), exist_ok=True)
    file_path = os.path.join(output_dir, 'raw_pubmed_unprocessed', f'pmid_batch_{current_timestamp}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump((output, meta_out, pmids), f)
    print(f'[PubMed Processor]: Batch with {len(pmids)} articles saved to pmid_batch_{current_timestamp}.pkl')

    return file_path


def load_batch_from_file(file_path):
    with open(file_path, 'rb') as f:
        output, meta_out, pmids = pickle.load(f)
    return output, meta_out, pmids


def save_download_tracker(downloaded_ranges):
    with open(os.path.join(output_dir, DOWNLOAD_TRACKER_FILE), 'w') as f:
        json.dump(downloaded_ranges, f)


def load_download_tracker():
    if os.path.exists(os.path.join(output_dir, DOWNLOAD_TRACKER_FILE)):
        with open(os.path.join(output_dir, DOWNLOAD_TRACKER_FILE), 'r') as f:
            t = json.load(f)
            t = [tuple(dl) for dl in t]
            t.sort(key=lambda x: x[0])
            return t
    return []


def is_range_downloaded(downloaded_ranges, year_start, year_end):
    for start, end in downloaded_ranges:
        if start <= year_start and end >= year_end:
            return True
    return False


def attempt_request(fullURL, timeout=60, retries=10):
    retry = 0
    while True:
        try:
            fetch = ur.urlopen(fullURL, timeout=timeout)
            return fetch
        except urllib.error.HTTPError as e:
            retry += 1
            if e.code == 429:
                time.sleep(1)
            else:
                if retry == retries:
                    raise e


def handle_pubmed_acquisition(batch_size, **kwargs):
    def process_batches(batches, pickle_batch=False):
        for pmids in batches:
            print(f'[PubMed Processor]: Received a batch of {len(pmids)} articles')
            output = []
            meta_out = []
            pmids_out = []
            with tqdm(total=len(pmids), desc="Downloading and preprocessing articles") as pbar:
                for pmid in pmids:
                    try:
                        title, abstract, meta_data = get_abstract_bypmid(pmid)
                    except Exception as e:
                        print(
                            f"[PubMed Processor]: PubMed Fetch raised an exception while attempting to acquire {pmid}")
                        print(f"[PubMed Processor]: {e}")
                        log_pmid_exception(pmid, e)
                        pbar.update(1)
                        continue
                    if title == "" and abstract == "":
                        print(f"\n{pmid} is empty")
                        pbar.update(1)
                        continue
                    elif title == "":
                        print(f"\n{pmid} has no title but appears to have content")
                        pbar.update(1)
                        continue
                    elif abstract == "":
                        pbar.update(1)
                        continue
                    data = []

                    sections = abstract.split("\n")
                    data.append("TITLE : " + str(title))
                    for s in sections:
                        if s == "":
                            continue
                        data.append(s)
                    output.append("\n".join(data))
                    meta_out.append(meta_data)
                    pmids_out.append(pmid)
                    pbar.update(1)

            if pickle_batch:
                file_path = save_batch_to_file(output, meta_out, pmids_out)
                yield output, meta_out, pmids_out, file_path
            else:
                yield output, meta_out, pmids_out

    if kwargs.get('pickle'):
        pickle_pmids = True
    else:
        pickle_pmids = False
    if kwargs.get('pmids'):
        batches = [kwargs['pmids'][i:i + batch_size] for i in range(0, len(kwargs['pmids']), batch_size)]
        yield from process_batches(batches, pickle_batch=pickle_pmids)
    elif kwargs.get('pipeline') == 'downloader' or kwargs.get('pipeline') is None:
        batches = get_pmids(year_begin=kwargs['year_start'], year_end=kwargs['year_end'], batch_size=batch_size)
        yield from process_batches(batches, pickle_batch=True)
    elif kwargs.get('pipeline') == 'nlp':
        while True:
            unprocessed_dir = os.path.join(output_dir, 'raw_pubmed_unprocessed')
            processed_dir = os.path.join(output_dir, 'raw_pubmed_processed')

            if not os.path.exists(unprocessed_dir):
                print("[PubMed NLP Processor]: No batches found. Sleeping...")
                time.sleep(30)
                continue

            os.makedirs(processed_dir, exist_ok=True)

            pickle_files = [f for f in os.listdir(unprocessed_dir) if f.endswith('.pkl')]
            if not pickle_files:
                print("[PubMed NLP Processor]: No batches found. Sleeping...")
                time.sleep(30)
                continue

            for file_name in pickle_files:
                file_path = os.path.join(unprocessed_dir, file_name)
                try:
                    output, meta_out, pmids = load_batch_from_file(file_path)
                    if len(pmids) == 0:
                        os.remove(file_path)
                        continue
                    yield output, meta_out, pmids

                    new_file_path = os.path.join(processed_dir, file_name)
                    os.rename(file_path, new_file_path)
                    print(f'[PubMed NLP Processor]: Moved processed file to {new_file_path}')
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue


def handle_abstract_parsing(batch, args=None):
    docs, meta_data, pmids = batch
    print(f"[Base NLP Parser]: Processing {len(docs)} documents.")
    try:
        sent_idx, predictions_batch = Model.predictBodyOfText(docs, flatten=True, split_newlines=True,
                                                              SentenceClassificationDriver=SentenceClassification)
    except SequenceTooLong as e:
        print(
            f"[Base NLP Parser]: [ERROR]: Encountered a sentence that is too long to process.  The PMID will be logged and ignored.")
        pmid_exception = pmids[e.error_code]
        print(f"[Base NLP Parser]: [ERROR]: {e} from {pmid_exception}")
        log_pmid_exception(pmid_exception, e)
        del docs[e.error_code]
        del meta_data[e.error_code]
        del pmids[e.error_code]
        return handle_abstract_parsing((docs, meta_data, pmids), args)
    except Exception as e:
        if hasattr(e, 'idx'):
            p_idx = sent_idx[e.idx]
            pmid = pmids[p_idx]

            print(
                f"[Base NLP Parser]: [ERROR]: {pmid} has caused the base NLP parser to raise an exception.  The PMID will be logged and ignored.")
            log_pmid_exception(pmid, e)
            del docs[p_idx]
            del meta_data[p_idx]
            del pmids[p_idx]

            return handle_abstract_parsing((docs, meta_data, pmids), args)
        else:
            raise e

    return docs, meta_data, pmids, sent_idx, predictions_batch


def flatten(input):
    if not isinstance(input, list):
        raise TypeError('Input must be a list of lists')

    flattened_list = []
    idx_map = []

    for i, sublist in enumerate(input):
        if not isinstance(sublist, list):
            raise TypeError(f'Expected list at index {i}, got {type(sublist).__name__}')

        flattened_list.extend(sublist)
        idx_map.extend([i] * len(sublist))

    return flattened_list, idx_map


def unflatten(input, idx_map):
    unflattened_dict = {}

    for index, item in zip(idx_map, input):
        if index not in unflattened_dict:
            unflattened_dict[index] = []
        unflattened_dict[index].append(item)

    unflattened_list = [unflattened_dict[i] for i in sorted(unflattened_dict.keys())]

    return unflattened_list

def handle_negation_detection(batch, args=None):
    docs, meta_data, pmids, sent_idx, predictions_batch = batch
    print(f"[Negation Detection]: Detecting negations in {len(docs)} documents.")
    try:
        predictions_batch = NegationDriver.detectNegations(predictions_batch)
    except Exception as e:
        if hasattr(e, 'idx'):
            p_idx = sent_idx[e.idx]
            pmid = pmids[p_idx]

            print(
                f"[Negation Detection]: [ERROR]: {pmid} has caused the negation detection handler to raise an exception.  The PMID will be logged and ignored.")
            log_pmid_exception(pmid, e)
            del docs[p_idx]
            del meta_data[p_idx]
            del pmids[p_idx]
            t_batch = unflatten(predictions_batch, sent_idx)
            del t_batch[p_idx]
            predictions_batch, sent_idx = flatten(t_batch)

            return handle_negation_detection((docs, meta_data, pmids, sent_idx, predictions_batch), args)
        else:
            raise e

    predictions_batch = unflatten(predictions_batch, sent_idx)
    return docs, meta_data, pmids, predictions_batch


def handle_proposition_formulations(batch, args=None):
    docs, meta_data, pmids, predictions_batch = batch
    print(f"[Proposition Formulation]: Formulating propositions for {len(docs)} documents.")
    with tqdm(total=len(predictions_batch), desc="Formulating propositions") as pbar:
        for i, predictions in enumerate(predictions_batch):
            try:
                PropositionDriver.buildPropositions(predictions)
            except Exception as e:
                pmid = pmids[i]
                print(
                    f"[Proposition Formulation]: [ERROR]: {pmid} has caused the negation detection handler to raise an exception.  The PMID will be logged and ignored.")
                log_pmid_exception(pmid, e)

                del docs[i]
                del meta_data[i]
                del pmids[i]
                del predictions_batch[i]

            pbar.update(1)

    return docs, meta_data, pmids, predictions_batch


def upload_to_solr(solr_url, json_doc):
    headers = {'Content-Type': 'application/json'}
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(f'{solr_url}/update/json/docs', headers=headers, data=json.dumps(json_doc))
            if response.status_code == 200:
                return response.status_code, response.text
            else:
                print(f'Error {response.status_code}: {response.text}')
        except requests.RequestException as e:
            print(f'Request failed: {e}')
        time.sleep(2 ** attempt)  # Exponential backoff
    return None, 'Failed after retries'


def handle_map_generation_and_upload(batch, args=None):
    docs, meta_data, pmids, predictions_batch = batch
    print(f"[EvidenceMap]: Generating maps for {len(docs)} documents.")
    solr_url = SOLR_CONFIG['base_url'] + SOLR_CONFIG['EvidenceMap_core']
    with tqdm(total=len(predictions_batch), desc="Generating Maps & Uploading") as pbar:
        for i, predictions in enumerate(predictions_batch):
            try:
                predictions, proposed_arms = EvidenceMapDriver.fit_propositions(predictions)
                ([participants, _, _], EvidenceMap) = EvidenceMapDriver.build_map(predictions, proposed_arms)

                # Build JSON doc for SOLR
                json_object = EvidenceMapDriver.build_json(participants, EvidenceMap, meta_data[i], predictions,
                                                           pmids[i],
                                                           docs[i])

                if not args.output:
                    upload_to_solr(solr_url, json_object)
                else:
                    os.makedirs(args.output, exist_ok=True)
                    with open(os.path.join(args.output, f"{pmids[i]}.json"), 'w') as f:
                        f.write(json.dumps(json_object))
            except Exception as e:
                pmid = pmids[i]
                print(
                    f"[EvidenceMap]: [ERROR]: {pmid} has caused the evidence map handler to raise an exception.  The PMID will be logged and ignored.")
                log_pmid_exception(pmid, e)
            pbar.update(1)

    if not args.output:
        print("[EvidenceMap]: Committing changes to SOLR.")
        commit_response = requests.get(f'{solr_url}/update?commit=true')
        if commit_response.status_code != 200:
            print(
                f'[EvidenceMap]: Commit failed with status code {commit_response.status_code}: {commit_response.text}')
        else:
            print(f"[EvidenceMap]: {len(docs)} successfully committed.")


def query_pubmed_for_ids(query):
    retstart = 0
    retstartParam = '&retstart=' + str(retstart)

    fullquery = query + retstartParam

    print(f"[PubMed Query]: Issuing query: {fullquery}")

    try:
        fetch = attempt_request(fullquery)
    except:
        fullURL = urllib.parse.quote(query, safe=':/')  # <- here
        fetch = attempt_request(fullURL)

    datam = json.loads(fetch.read().decode('utf-8'))

    try:
        count = int(datam['esearchresult']['count'])
    except:
        print(f"[PubMed Query]: Error in fetching data. {datam}")
        yield None
        return

    if count == 0:
        print(f"[PubMed Query]: No results found for query")
        yield []
        return

    if count > 9500:
        print(f"[PubMed Query]: Resultset size of {count} exceeds maximum safety limit of 9500.")
        yield None
        return

    while retstart < count:
        retstartParam = '&retstart=' + str(retstart)
        fullquery = query + retstartParam
        fetch = attempt_request(fullquery)
        data = json.loads(fetch.read().decode('utf-8'))
        retstart = int(data['esearchresult']['retstart']) + int(data['esearchresult']['retmax'])

        yield data['esearchresult']['idlist']


def get_remaining_ranges(start, end, downloaded_list):
    # making sure the list is sorted
    range_start = start
    range_end = end

    intervals = downloaded_list.copy()

    i = 0
    SECONDS_IN_A_DAY = 86400
    # Traverse through each interval finding gaps
    while start < end and i < len(intervals):
        dl_start, dl_end = intervals[i]
        if dl_start > start:
            if dl_start - start > SECONDS_IN_A_DAY:
                # If there is a gap before the current downloaded interval, yield it
                yield (start, min(dl_start - SECONDS_IN_A_DAY, end))
                start = min(dl_start, end)
            else:
                # Merge the gap into a single interval
                start = dl_start
        # Move start to end of current downloaded interval if it overlaps
        if dl_start <= start <= dl_end:
            start = dl_end + SECONDS_IN_A_DAY
        i += 1
    # If any part of the range remains, yield it
    if start < end and end - start > SECONDS_IN_A_DAY:
        yield (start, end)


def get_date_range(year_begin, year_end, mode='batch', downloaded_list=[]):
    def date_to_unix(year, month, day):
        dt = datetime.datetime(year, month, day, tzinfo=pytz.utc)
        return int(dt.timestamp())

    unix_ranges = []

    if mode == 'batch':
        mindate_unix = date_to_unix(year_begin, 1, 1)
        maxdate_unix = date_to_unix(year_end, 12, 31)
        unix_ranges.append((mindate_unix, maxdate_unix))
    elif mode == 'year':
        for year in range(year_begin, year_end + 1):
            mindate_unix = date_to_unix(year, 1, 1)
            maxdate_unix = date_to_unix(year, 12, 31)
            unix_ranges.append((mindate_unix, maxdate_unix))
    elif mode == 'month':
        for year in range(year_begin, year_end + 1):
            month_day_map = [0, 31, 29 if ((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0) else 28, 31, 30,
                             31, 30, 31, 31,
                             30, 31, 30, 31]
            for month in range(1, 13):
                mindate_unix = date_to_unix(year, month, 1)
                maxdate_unix = date_to_unix(year, month, month_day_map[month])
                unix_ranges.append((mindate_unix, maxdate_unix))
    elif mode == 'day':
        for year in range(year_begin, year_end + 1):
            month_day_map = [0, 31, 29 if ((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0) else 28, 31, 30,
                             31, 30, 31, 31,
                             30, 31, 30, 31]
            for month in range(1, 13):
                for day in range(1, month_day_map[month] + 1):
                    mindate_unix = date_to_unix(year, month, day)
                    maxdate_unix = mindate_unix
                    unix_ranges.append((mindate_unix, maxdate_unix))

    for start, end in unix_ranges:
        for remaining_start, remaining_end in get_remaining_ranges(start, end, downloaded_list):
            mindate_str = datetime.datetime.utcfromtimestamp(remaining_start).strftime('%Y/%m/%d')
            maxdate_str = datetime.datetime.utcfromtimestamp(remaining_end).strftime('%Y/%m/%d')
            print(f"[PubMed Batch Handler]: Attempting {mindate_str} to {maxdate_str}")
            yield (f'&mindate={mindate_str}', f'&maxdate={maxdate_str}', remaining_start, remaining_end)


def get_pmids(year_begin, year_end, batch_size=9500):
    # Credit to Yingcheng Sun and Weng Labs for the original implementation
    # Updated and reformatted by Maximilian Doerr
    term = "(Randomized+Controlled+Trial[Publication+Type])"
    baseURL = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    eutil = 'esearch.fcgi?'
    dbParam = 'db=pubmed'  # db=pmc
    usehistoryParam = '&usehistory=y'
    retmode = '&retmode=JSON'
    fieldParam = '&field=title'
    termParam = '&term=' + str(term)
    retmaxParam = '&retmax=' + str(batch_size)

    download_tracker = load_download_tracker()

    cyclemodes = ['batch', 'year', 'month', 'day']
    mode = 0
    breakflag = False
    changing_mode = False
    pmids = []
    while not breakflag:
        print(f"[PubMed Batch Handler]: Fetching articles from {year_begin} to {year_end} in {cyclemodes[mode]} mode.")
        for mindate, maxdate, unix_start, unix_end in get_date_range(year_begin, year_end, cyclemodes[mode],
                                                                     download_tracker):
            queryURL = baseURL + eutil + dbParam + fieldParam + termParam + usehistoryParam + retmode + retmaxParam + mindate + maxdate
            for chunk in query_pubmed_for_ids(queryURL):
                if chunk is None:
                    mode += 1
                    changing_mode = True
                    if mode >= len(cyclemodes):
                        print(f"[PubMed Batch Handler]: Unable to fetch articles from {year_begin} to {year_end}.")
                        yield [], None, None
                        return
                    print(f"[PubMed Batch Handler]: Switching to {cyclemodes[mode]} mode.")
                    break
                pmids.extend(chunk)
                if len(pmids) >= batch_size:
                    yield pmids
                    pmids = []

            if changing_mode:
                break
            start_of_today = int(time.mktime(datetime.date.today().timetuple()))
            if unix_start >= start_of_today:
                continue
            if unix_end >= start_of_today:
                unix_end = start_of_today - SECONDS_IN_A_DAY

            download_tracker.append((unix_start, unix_end))
            save_download_tracker(download_tracker)
        if not changing_mode:
            breakflag = True
            yield pmids
        else:
            changing_mode = False


def get_abstract_bypmid(pmid):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=" + str(
        pmid) + "&retmode=XML&rettype=abstract"

    fetch = attempt_request(url, timeout=10)
    datam = fetch.read().decode("utf-8")

    retries = 0


    while True:
        try:
            fetch = ur.urlopen(url, timeout=10)
            datam = fetch.read().decode('utf-8')
            break  # xml of one article
        except Exception as e:
            if str(e) == "HTTP Error 429: Too Many Requests":
                sleep(1)
                continue
            else:
                print("")
                print(f"[PubMed Fetch]: Error fetching PMID {pmid}: {e}")
                retries += 1
                if retries >= 10:
                    raise
                else:
                    sleep(1)
                    continue

    datam = re.sub("<i>", "", datam)  # remove <i> in xml to avoid failure in parsing .text
    datam = re.sub("</i>", "", datam)
    datam = re.sub("<sup>", "", datam)  # remove <i> in xml to avoid failure in parsing .text
    datam = re.sub("</sup>", "", datam)
    datam = re.sub("<sub>", "", datam)  # remove <i> in xml to avoid failure in parsing .text
    datam = re.sub("</sub>", "", datam)
    datam = re.sub("<em>", "", datam)  # remove <i> in xml to avoid failure in parsing .text
    datam = re.sub("</em>", "", datam)
    datam = re.sub("<strong>", "", datam)  # remove <i> in xml to avoid failure in parsing .text
    datam = re.sub("</strong>", "", datam)
    datam = re.sub("<b>", "", datam)  # remove <b> in xml to avoid failure in parsing .text
    datam = re.sub("</b>", "", datam)

    xmldoc = xml_parser.fromstring(datam)

    PubmedArticle = xmldoc.find("PubmedArticle")
    abstract_text = ""
    title_text = ""
    meta_data = {}

    meta_data['pubdate'] = ""
    meta_data['source'] = ""
    meta_data['volume'] = ""
    meta_data['pages'] = ""
    meta_data['pubtype'] = []
    meta_data['authors'] = []

    if PubmedArticle is not None:
        data = PubmedArticle.find("MedlineCitation")
        if data is not None:
            article = data.find("Article")
            if article is not None:
                title = article.find("ArticleTitle")
                # print("Title:",title.text)
                title_text = title.text
                abstract = article.find("Abstract")
                if abstract is not None:
                    for seg in abstract:
                        if seg.text:
                            if 'NlmCategory' in seg.attrib:
                                abstract_text = abstract_text + "\n" + seg.attrib['NlmCategory'] + " : " + seg.text
                            elif 'Label' in seg.attrib:
                                abstract_text = abstract_text + "\n" + seg.attrib['Label'] + " : " + seg.text
                            else:
                                abstract_text = abstract_text + "\n" + seg.text
                        elif 'Label' in seg.attrib and len(seg.attrib['Label'].split(':')) > 1:
                            abstract_text = abstract_text + "\n" + seg.attrib['Label'].split(':')[0] + " : " + \
                                            seg.attrib['Label'].split(':')[1]

                journal = article.find("Journal")
                if journal is not None:
                    if journal.find("ISOAbbreviation") is not None:
                        meta_data['source'] = journal.find("ISOAbbreviation").text
                    if journal.find("JournalIssue").find('Volume') is not None:
                        list_string = journal.find("JournalIssue").find('Volume').text.split()
                        # updated in 202202, to avoid Solr indexing issue
                        if len(list_string) >= 1:
                            meta_data['volume'] = ''
                        else:
                            meta_data['volume'] = list_string

                Pagination = article.find("Pagination")
                if Pagination is not None:
                    meta_data['pages'] = Pagination.find("MedlinePgn").text

                AuthorList = article.find('AuthorList')

                if AuthorList is not None:
                    for author in AuthorList.findall('Author'):
                        if author.find('Initials') is not None and author.find('LastName') is not None:
                            meta_data['authors'].append(
                                author.find('Initials').text + '. ' + author.find('LastName').text)
                PublicationTypeList = article.find('PublicationTypeList')
                if PublicationTypeList is not None:
                    for PublicationType in PublicationTypeList.findall('PublicationType'):
                        meta_data['pubtype'].append(PublicationType.text)

        PubmedData = PubmedArticle.find('PubmedData')
        if PubmedData is not None:
            History = PubmedData.find('History')
            if History.find("PubMedPubDate") is not None:
                date = History.find("PubMedPubDate")
                Year = ""
                Month = ""
                Day = ""
                if date.find('Year') is not None:
                    Year = date.find('Year').text
                if date.find('Month') is not None:
                    Month = date.find('Month').text + '/'
                if date.find('Day') is not None:
                    Day = date.find('Day').text + '/'

                meta_data['pubdate'] = Month + Day + Year

    meta_data['title'] = title_text

    abstract_text = re.sub("^\s+", "", abstract_text)

    return title_text, abstract_text, meta_data


def get_excepted_pmids():
    file_path = os.path.join(output_dir, 'PubMed_errors.txt')
    backup_file_path = file_path + '.old'

    pmids = []
    errors = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(', Error: ')
                if len(parts) == 2:
                    pmid = parts[0].replace('PMID: ', '')
                    error = parts[1]
                    pmids.append(pmid)
                    errors.append(error)

        shutil.move(file_path, backup_file_path)
    except:
        return [], []

    return pmids, errors

pipeline_stages = [
    handle_abstract_parsing,
    handle_negation_detection,
    handle_proposition_formulations,
    handle_map_generation_and_upload,
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch PubMed articles.')
    parser.add_argument('-y', '--years', type=lambda s: list(map(int, s.split('-'))),
                        help='Year range to fetch articles from. Format: YYYY-YYYY')
    parser.add_argument('-p', '--pmids', type=lambda s: s.split('|'), help='Pipe-separated list of PMID IDs to fetch.')
    parser.add_argument('-b', '--batchsize', type=int, default=100, help='Size of batches to process at a time.')
    parser.add_argument('-o', '--output', type=str,
                        help='Output directory to write to.  Specifying this will write the JSON documents to the directory instead of uploading to SOLR.')
    parser.add_argument('--tempdir', type=str, default='/tmp',
                        help='Temporary directory for storing/processing batches.')
    parser.add_argument('-P', '--pipeline', type=str, default=None,
                        help='Run the script in pipeline mode. Possible values: downloader, nlp')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Reads in a debug.pkl file generated if the pipeline terminated abnormally, and resumes the pipeline for easy debugging.')
    parser.add_argument('-e', '--reacquire-exceptions', action='store_true',
                        help='Reacquires articles that failed to download due to exceptions encountered.  This overrides the years and pmids option.')

    args = parser.parse_args()
    if args.batchsize > 9500:
        print("Batch size cannot exceed 9500.")
        exit(1)

    output_dir = args.tempdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not args.debug:
        if args.reacquire_exceptions:
            args.pmids = False
            file_path = os.path.join(output_dir, 'PubMed_errors.txt')
            pmids, errors = get_excepted_pmids()
            if not pmids:
                print("Nothing to acquire")
                exit(0)
            print(f"Attempting to acquire {len(pmids)} articles that failed to acquire.")
            batch_iterator = handle_pubmed_acquisition(pmids=pmids, batch_size=args.batchsize,
                                                       pipeline=args.pipeline, pickle=True)
        elif args.years:
            args.pmids = False
            year_begin, year_end = args.years
            if args.pipeline is None or args.pipeline == 'downloader':
                print(f"Fetching articles from {year_begin} to {year_end}.")
            else:
                print(f"Running NLP pipeline on unprocessed articles.  This is a continuous operation.")
            batch_iterator = handle_pubmed_acquisition(year_start=year_begin, year_end=year_end,
                                                       batch_size=args.batchsize,
                                                       pipeline=args.pipeline, pickle=True)
        elif args.pmids:
            if not args.pmids:
                print("Nothing to acquire")
                exit(0)
            print(f"Fetching {len(args.pmids)} specified articles.")
            batch_iterator = handle_pubmed_acquisition(pmids=args.pmids, batch_size=args.batchsize,
                                                       pipeline=args.pipeline, pickle=False)

        if args.pmids or args.pipeline == 'nlp' or args.pipeline is None:
            from Website.pipeline import SentenceClassification, Model, NegationDriver, PropositionDriver, \
                EvidenceMapDriver
            from Website.exceptions import SequenceTooLong

            batch_counter = 0
            for batch in batch_iterator:
                if len(batch) > 3 and batch[3] is not None:
                    file_path = batch[3]
                    batch = batch[:3]
                else:
                    file_path = None

                if not batch:
                    continue
                batch_counter += 1
                print(f"Batch {batch_counter}: ")
                for i, stage in enumerate(pipeline_stages):
                    try:
                        batch = stage(batch, args)
                    except Exception as e:
                        print(
                            f"[FATAL ERROR]: An uncaught exception was encountered in the pipeline.  Pickling pipeline batch into debug.pkl...")
                        cpu_batch = move_to_cpu(batch)
                        torch.cuda.empty_cache()

                        os.makedirs(output_dir, exist_ok=True)
                        debug_file_path = os.path.join(output_dir, 'debug.pkl')
                        with open(debug_file_path, 'wb') as debug_file:
                            pickle.dump((i, cpu_batch, args), debug_file)
                        print(
                            f"[FATAL ERROR]: Pipeline data has been pickled.  Use -d to open the pickle for debugging.")
                        print(f"[FATAL ERROR]: Pipeline will now terminate.")
                        raise e

                if file_path is not None:
                    processed_file_path = file_path.replace('raw_pubmed_unprocessed', 'raw_pubmed_processed')
                    os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
                    os.rename(file_path, processed_file_path)

        else:
            for _ in batch_iterator:
                pass
    else:
        from Website.pipeline import SentenceClassification, Model, NegationDriver, PropositionDriver, \
            EvidenceMapDriver, SequenceTooLong

        debug_file_path = os.path.join(output_dir, 'debug.pkl')
        with open(debug_file_path, 'rb') as debug_file:
            j, batch, t_args = pickle.load(debug_file)
        try:
            for i in range(j, len(pipeline_stages)):
                batch = pipeline_stages[i](batch, t_args)
        finally:
            if os.path.isfile(debug_file_path):
                os.remove(debug_file_path)