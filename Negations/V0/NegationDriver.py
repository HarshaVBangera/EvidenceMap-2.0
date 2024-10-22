import os
import re
import sys

import spacy

modules = {key: value for key, value in sys.modules.items() if 'EvidenceBaseNegationDriver' in key}

if modules:
    module = list(modules.values())[0]
    EvidenceBaseNegationDriver = getattr(module, 'EvidenceBaseNegationDriver')
else:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from EvidenceBaseNegationDriver import EvidenceBaseNegationDriver


# This driver is adapted from the original negation code in the parsing pipeline.  It has been cleaned up and simplified for slightly better readability and maintainability
class V0NegationDriver(EvidenceBaseNegationDriver):
    _identifier = "V0"

    def __init__(self, model_driver, config):
        super().__init__(model_driver, config)

        self.tag_possible_phrases = config.get("tag_possible_phrases", True)

        regex_rules = []

        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'negation_triggers.txt'))

        try:
            with open(file_path, 'r') as file:
                lines = [line.strip() for line in file if line.strip()]
                lines.sort(key=lambda x: len(x.split("\t\t")[0]), reverse=True)

                for line in lines:
                    if line:
                        trigger, category = line.split("\t\t")
                        trigger = re.escape(trigger)
                        trigger = trigger.replace("\\ ", "\\s+")
                        regex_rules.append((re.compile(r"\b(" + trigger + r")\b", re.IGNORECASE), category))
        except FileNotFoundError:
            print("ERROR: negation_triggers.txt not found, no negation detection will be performed")

        self._rules = regex_rules

    def _tag_and_tokenize(self, sentence, entity):
        active_rules = []
        masked_terms = []

        special_tokens = ["[PHRASE]"]

        sentence = sentence[:entity['start']] + '[PHRASE]' + sentence[entity['end']:]

        for regex, category in self._rules:

            mask = f"{category}[{len(active_rules)}]"

            match = re.search(regex, sentence)

            if match:
                masked_term = match.group()
                masked_terms.append(masked_term)
                sentence = re.sub(regex, mask, sentence)
                active_rules.append((regex, category))
                special_tokens.append(mask)

        for token in special_tokens:
            special_case = [{spacy.symbols.ORTH: token}]
            self.nlp.tokenizer.add_special_case(token, special_case)
        tokens = [token.text for token in self.nlp(sentence)]

        return sentence, active_rules, tokens, masked_terms

    def detectNegations(self, model_prediction):
        for prediction in model_prediction:
            sentence = prediction['sentence']
            for entity in prediction['entities']:
                entity['negation_status'] = 'affirmed'
                masked_sentence, active_rules, tokens, masked_terms = self._tag_and_tokenize(sentence, entity)

                overlap = False
                preNegation = False
                postNegation = False
                prePossible = False
                postPossible = False

                sentencePortion = ''
                sb = []

                if active_rules:
                    for i in range(len(tokens)):
                        if tokens[i][:6] == '[PREN]':
                            preNegation = True
                            overlap = False

                        if tokens[i][:6] in ['[CONJ]', '[PSEU]', '[POST]', '[PREP]', '[POSP]']:
                            overlap = True

                        if i + 1 < len(tokens):
                            if tokens[i + 1][:6] == '[PREN]':
                                overlap = True
                                sentencePortion = ''

                        if preNegation == True and overlap == False:
                            tokens[i] = tokens[i].replace('[PHRASE]', '[NEGATED]')
                            sentencePortion = sentencePortion + ' ' + tokens[i]

                        sb.append(tokens[i])

                    sentencePortion = ''
                    sb.reverse()
                    tokens = sb
                    sb2 = []

                    for i in range(len(tokens)):
                        if tokens[i][:6] == '[POST]':
                            postNegation = True
                            overlap = False

                        if tokens[i][:6] in ['[CONJ]', '[PSEU]', '[POST]', '[PREP]', '[POSP]']:
                            overlap = True

                        if i + 1 < len(tokens):
                            if tokens[i + 1][:6] == '[POST]':
                                overlap = True
                                sentencePortion = ''

                        if postNegation == True and overlap == False:
                            tokens[i] = tokens[i].replace('[PHRASE]', '[NEGATED]')
                            sentencePortion = tokens[i] + ' ' + sentencePortion

                        sb2.insert(0, tokens[i])

                    sentencePortion = ''

                    if self.tag_possible_phrases:
                        tokens = sb2
                        sb3 = []

                        for i in range(len(tokens)):
                            if tokens[i][:6] == '[PREP]':
                                prePossible = True
                                overlap = False

                            if tokens[i][:6] in ['[CONJ]', '[PSEU]', '[POST]', '[PREP]', '[POSP]']:
                                overlap = True

                            if i + 1 < len(tokens):
                                if tokens[i + 1][:6] == '[PREP]':
                                    overlap = True
                                    sentencePortion = ''

                            if prePossible == True and overlap == False:
                                tokens[i] = tokens[i].replace('[PHRASE]', '[POSSIBLE]')
                                sentencePortion = sentencePortion + ' ' + tokens[i]

                            sb3.append(tokens[i])

                        sentencePortion = ''
                        sb3.reverse()
                        tokens = sb3
                        sb4 = []

                        for i in range(len(tokens)):
                            if tokens[i][:6] == '[POSP]':
                                postPossible = True
                                overlap = False

                            if tokens[i][:6] in ['[CONJ]', '[PSEU]', '[POST]', '[PREP]', '[POSP]']:
                                overlap = True

                            if i + 1 < len(tokens):
                                if tokens[i + 1][:6] == '[POSP]':
                                    overlap = True
                                    sentencePortion = ''

                            if postPossible == True and overlap == False:
                                tokens[i] = tokens[i].replace('[PHRASE]', '[POSSIBLE]')
                                sentencePortion = tokens[i] + ' ' + sentencePortion

                            sb4.insert(0, tokens[i])

                    for token in tokens:
                        if token == '[NEGATED]':
                            entity['negation_status'] = 'negated'
                            entity['negation_position'] = 'pre' if preNegation else 'post'
                            for i in range(len(active_rules) - 1, -1, -1):
                                if preNegation and active_rules[i][1] == '[PREN]':
                                    entity['negation_phrase'] = masked_terms[i]
                                    break
                                elif postNegation and active_rules[i][1] == '[POST]':
                                    entity['negation_phrase'] = masked_terms[i]
                                    break

                            break
                        elif token == '[POSSIBLE]':
                            entity['negation_status'] = 'possible'
                            entity['negation_position'] = 'pre' if preNegation else 'post'
                            for i in range(len(active_rules) - 1, -1, -1):
                                if prePossible:
                                    entity['negation_phrase'] = masked_terms[i]
                                    break
                                elif postPossible:
                                    entity['negation_phrase'] = masked_terms[i]
                                    break
        return model_prediction
