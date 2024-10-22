import json
import re
import urllib.parse
import urllib.request as ur

import inflect
import jellyfish
from EvidenceConfig import SOLR_CONFIG

inflect = inflect.engine()

base_url = f"{SOLR_CONFIG['base_url']}{SOLR_CONFIG['UMLS_core']}/select?q="


class QueryExpansion():
    def __init__(self):
        self.baseURL = base_url

    def term2CUI(self, term):
        _term = urllib.parse.quote(term)

        fullURL = self.baseURL + "STR%3A" + _term.replace(u"\u2018", "'").replace(u"\u2019", "'") \
            .replace('β', '%CE%B2').replace('α', '%CE%B1').replace('γ', '%CE%B3')

        f = ur.urlopen(fullURL)
        data = json.load(f)["response"]['docs']
        CUIS = []

        for t in data:

            if re.search("[,\)\(;\/]", t["STR"][0]) or re.search("unspecified", t["STR"][0].lower()) or "nos" in \
                    t["STR"][0].lower().split(" ") or t["CUI"][0].lower() in CUIS:
                continue
            if jellyfish.jaro_similarity(t["STR"][0].lower(), term.lower()) < 0.75:
                continue

            CUI = t["CUI"][0].lower()

            CUIS.append(CUI)

        return CUIS

    def CUI2terms(self, CUI):
        fullURL = self.baseURL + "CUI%3A" + CUI

        f = ur.urlopen(fullURL)
        data = json.load(f)["response"]['docs']
        terms = []
        for t in data:

            if re.search("[,\)\(;\/]", t["STR"][0]) or re.search("unspecified", t["STR"][0].lower()) or "nos" in \
                    t["STR"][0].lower().split(" ") or t["STR"][0].lower() in terms:
                continue
            # Some synonyms of "cognitive" include "&", such as "cognitve & behavioral therapy", which is not accepted by Solr
            terms.append(t["STR"][0].lower().replace('&', '').replace('  ', ' '))
        return terms

    def expand(self, term):
        # UMLS retrieval module has low recall when searching synonyms for phases with "disease" word such as "alzheimer's disease"
        term = term.replace('disease', '').replace("'s", '').strip()
        if inflect.singular_noun(term):
            term = inflect.singular_noun(term)
        CUIS = self.term2CUI(term)
        if len(CUIS) == 0:
            return []

        queries = []
        for cui in CUIS:
            terms = self.CUI2terms(cui)
            if len(terms) == 0:
                continue
            queries.extend(terms)
        if term not in queries:
            queries.append(term)
        return queries
