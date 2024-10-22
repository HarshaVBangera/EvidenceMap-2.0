import re

import inflect

inflect = inflect.engine()
from Website.QueryExpand import QueryExpansion

QE = QueryExpansion()


class facetprocessor():
    def __init__(self, participant, intervention, outcome):
        self.participant = participant
        self.intervention = intervention
        self.outcome = outcome

    def if_similar(self, t1, t2):
        cui_list1 = QE.term2CUI(t1)
        cui_list2 = QE.term2CUI(t2)
        if (set(cui_list1) & set(cui_list2)):
            return True
        else:
            return False

    def process_p_o(self, participant):
        p = iter(participant)

        p_dict = dict(zip(p, p))
        p_dict_merged = {}

        mapping_keys = {}
        keys = list(p_dict.keys()).copy()

        for key in keys:
            key_old = key

            # remove "()"
            key = re.sub("\s?\(.*\)", "", key_old)
            if re.search("\)", key):
                key = re.sub("^.*\)\s+", "", key)
            if re.search("\(", key):
                key = re.sub("\s+\(.*$", "", key)

            key = key.lower()
            try:
                if inflect.singular_noun(key):
                    key = inflect.singular_noun(key)
            except:
                pass

            if key not in p_dict_merged:
                p_dict_merged[key] = p_dict[key_old]
            else:
                p_dict_merged[key] = p_dict_merged[key] + p_dict[key_old]
            if key not in mapping_keys.keys():
                mapping_keys[key] = [key_old]
            else:
                mapping_keys[key].append(key_old)

        p_list = []

        for temp in sorted(p_dict_merged.items(), key=lambda kv: (-kv[1], kv[0])):
            p_list.append(temp[0])
            p_list.append(temp[1])

        return p_list, mapping_keys

    def process_intervention(self, intervention):
        i = iter(intervention)

        i_dict = dict(zip(i, i))
        i_dict_merged = {}

        mapping_keys = {}
        keys = list(i_dict.keys()).copy()

        for key in keys:
            key_old = key

            # remove "()"
            key = re.sub("\s?\(.*\)", "", key_old)

            if re.search("\)", key):
                key = re.sub("^.*\)\s+", "", key)
            if re.search("\(", key):
                key = re.sub("\s+\(.*$", "", key)

            if key.lower() in ["control", "placebo", "intervention", "interventions", "stardard", "active",
                               "active treatment", "active treatments", "standard of care", "stardard care",
                               "standard treatment", "standard treatments", "standard therapy", "treatment",
                               "treatments"]:
                continue

            if re.search("both|standard care|control|placebo|all", key.lower()):
                continue

            key = key.lower()

            # transfer plural to singular form
            if inflect.singular_noun(key):
                key = inflect.singular_noun(key)

            if key not in i_dict_merged:
                i_dict_merged[key] = i_dict[key_old]
            else:
                i_dict_merged[key] = i_dict_merged[key] + i_dict[key_old]

            if key not in mapping_keys.keys():

                mapping_keys[key] = [key_old]
            else:
                mapping_keys[key].append(key_old)

        i_list = []

        for temp in sorted(i_dict_merged.items(), key=lambda kv: (-kv[1], kv[0])):
            i_list.append(temp[0])
            i_list.append(temp[1])
        return i_list, mapping_keys
