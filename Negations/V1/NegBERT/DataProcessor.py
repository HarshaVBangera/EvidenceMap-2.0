import html
import os
import random
import re
import torch

import numpy as np
from sklearn.model_selection import train_test_split
from spacy.tokens import Doc
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AutoTokenizer

from Entities import Cues, Scopes
from general_utils import get_encoding_type, pad_sequences


class Data:
    def __init__(self, file, dataset_name='sfu', frac_no_cue_sents=1.0, TASK='negation'):
        '''
        file: The path of the data file.
        dataset_name: The name of the dataset to be preprocessed. Values supported: sfu, bioscope, starsem.
        frac_no_cue_sents: The fraction of sentences to be included in the data object which have no negation/speculation cues.
        '''

        def starsem(f_path, cue_sents_only=False, frac_no_cue_sents=1.0):
            raw_data = open(f_path)
            sentence = []
            labels = []
            label = []
            scope_sents = []
            data_scope = []
            scope = []
            scope_cues = []
            data = []
            cue_only_data = []

            for line in raw_data:
                label = []
                sentence = []
                tokens = line.strip().split()
                if len(tokens) == 8:  # This line has no cues
                    sentence.append(tokens[3])
                    label.append(3)  # Not a cue
                    for line in raw_data:
                        tokens = line.strip().split()
                        if len(tokens) == 0:
                            break
                        else:
                            sentence.append(tokens[3])
                            label.append(3)
                    cue_only_data.append([sentence, label])


                else:  # The line has 1 or more cues
                    num_cues = (len(tokens) - 7) // 3
                    # cue_count+=num_cues
                    scope = [[] for i in range(num_cues)]
                    label = [[],
                             []]  # First list is the real labels, second list is to modify if it is a multi-word cue.
                    label[0].append(3)  # Generally not a cue, if it is will be set ahead.
                    label[1].append(-1)  # Since not a cue, for now.
                    for i in range(num_cues):
                        if tokens[7 + 3 * i] != '_':  # Cue field is active
                            if tokens[8 + 3 * i] != '_':  # Check for affix
                                label[0][-1] = 0  # Affix
                                affix_list.append(tokens[7 + 3 * i])
                                label[1][-1] = i  # Cue number
                                # sentence.append(tokens[7+3*i])
                                # new_word = '##'+tokens[8+3*i]
                            else:
                                label[0][
                                    -1] = 1  # Maybe a normal or multiword cue. The next few words will determine which.
                                label[1][-1] = i  # Which cue field, for multiword cue altering.

                        if tokens[8 + 3 * i] != '_':
                            scope[i].append(1)
                        else:
                            scope[i].append(0)
                    sentence.append(tokens[3])
                    for line in raw_data:
                        tokens = line.strip().split()
                        if len(tokens) == 0:
                            break
                        else:
                            sentence.append(tokens[3])
                            label[0].append(3)  # Generally not a cue, if it is will be set ahead.
                            label[1].append(-1)  # Since not a cue, for now.
                            for i in range(num_cues):
                                if tokens[7 + 3 * i] != '_':  # Cue field is active
                                    if tokens[8 + 3 * i] != '_':  # Check for affix
                                        label[0][-1] = 0  # Affix
                                        label[1][-1] = i  # Cue number
                                    else:
                                        label[0][
                                            -1] = 1  # Maybe a normal or multiword cue. The next few words will determine which.
                                        label[1][-1] = i  # Which cue field, for multiword cue altering.
                                if tokens[8 + 3 * i] != '_':
                                    scope[i].append(1)
                                else:
                                    scope[i].append(0)
                    for i in range(num_cues):
                        indices = [index for index, j in enumerate(label[1]) if i == j]
                        count = len(indices)
                        if count > 1:
                            for j in indices:
                                label[0][j] = 2
                    for i in range(num_cues):
                        sc = []
                        for a, b in zip(label[0], label[1]):
                            if i == b:
                                sc.append(a)
                            else:
                                sc.append(3)
                        scope_cues.append(sc)
                        scope_sents.append(sentence)
                        data_scope.append(scope[i])
                    labels.append(label[0])
                    data.append(sentence)
            cue_only_samples = random.sample(cue_only_data, k=int(frac_no_cue_sents * len(cue_only_data)))
            cue_only_sents = [i[0] for i in cue_only_samples]
            cue_only_cues = [i[1] for i in cue_only_samples]
            starsem_cues = (data + cue_only_sents, labels + cue_only_cues)
            starsem_scopes = (scope_sents, scope_cues, data_scope)
            return [starsem_cues, starsem_scopes]

        def bioscope(f_path, cue_sents_only=False, frac_no_cue_sents=1.0):
            file = open(f_path, encoding='utf-8')
            sentences = []
            for s in file:
                sentences += re.split("(<.*?>)", html.unescape(s))
            cue_sentence = []
            cue_cues = []
            cue_only_data = []
            scope_cues = []
            scope_scopes = []
            scope_sentence = []
            sentence = []
            cue = {}
            scope = {}
            in_scope = []
            in_cue = []
            word_num = 0
            c_idx = []
            s_idx = []
            in_sentence = 0
            for token in sentences:
                if token == '':
                    continue
                elif '<sentence' in token:
                    in_sentence = 1
                elif '<cue' in token:
                    if TASK in token:
                        in_cue.append(str(re.split('(ref=".*?")', token)[1][4:]))
                        c_idx.append(str(re.split('(ref=".*?")', token)[1][4:]))
                        if c_idx[-1] not in cue.keys():
                            cue[c_idx[-1]] = []
                elif '</cue' in token:
                    in_cue = in_cue[:-1]
                elif '<xcope' in token:
                    # print(re.split('(id=".*?")',token)[1][3:])
                    in_scope.append(str(re.split('(id=".*?")', token)[1][3:]))
                    s_idx.append(str(re.split('(id=".*?")', token)[1][3:]))
                    scope[s_idx[-1]] = []
                elif '</xcope' in token:
                    in_scope = in_scope[:-1]
                elif '</sentence' in token:
                    # print(cue, scope)
                    if len(cue.keys()) == 0:
                        cue_only_data.append([sentence, [3] * len(sentence)])
                    else:
                        cue_sentence.append(sentence)
                        cue_cues.append([3] * len(sentence))
                        for i in cue.keys():
                            scope_sentence.append(sentence)
                            scope_cues.append([3] * len(sentence))
                            if len(cue[i]) == 1:
                                cue_cues[-1][cue[i][0]] = 1
                                scope_cues[-1][cue[i][0]] = 1
                            else:
                                for c in cue[i]:
                                    cue_cues[-1][c] = 2
                                    scope_cues[-1][c] = 2
                            scope_scopes.append([0] * len(sentence))

                            if i in scope.keys():
                                for s in scope[i]:
                                    scope_scopes[-1][s] = 1

                    sentence = []
                    cue = {}
                    scope = {}
                    in_scope = []
                    in_cue = []
                    word_num = 0
                    in_sentence = 0
                    c_idx = []
                    s_idx = []
                elif '<' not in token:
                    if in_sentence == 1:
                        words = token.split()
                        sentence += words
                        if len(in_cue) != 0:
                            for i in in_cue:
                                cue[i] += [word_num + i for i in range(len(words))]
                        elif len(in_scope) != 0:
                            for i in in_scope:
                                scope[i] += [word_num + i for i in range(len(words))]
                        word_num += len(words)
            cue_only_samples = random.sample(cue_only_data, k=int(frac_no_cue_sents * len(cue_only_data)))
            cue_only_sents = [i[0] for i in cue_only_samples]
            cue_only_cues = [i[1] for i in cue_only_samples]
            return [(cue_sentence + cue_only_sents, cue_cues + cue_only_cues),
                    (scope_sentence, scope_cues, scope_scopes)]

        def sfu_review(f_path, cue_sents_only=False, frac_no_cue_sents=1.0):
            file = open(f_path, encoding=get_encoding_type(f_path))
            sentences = []
            for s in file:
                sentences += re.split("(<.*?>)", html.unescape(s))
            cue_sentence = []
            cue_cues = []
            scope_cues = []
            scope_scopes = []
            scope_sentence = []
            sentence = []
            cue = {}
            scope = {}
            in_scope = []
            in_cue = []
            word_num = 0
            c_idx = []
            cue_only_data = []
            s_idx = []
            in_word = 0
            for token in sentences:
                if token == '':
                    continue
                elif token == '<W>':
                    in_word = 1
                elif token == '</W>':
                    in_word = 0
                    word_num += 1
                elif '<cue' in token:
                    if TASK in token:
                        in_cue.append(int(re.split('(ID=".*?")', token)[1][4:-1]))
                        c_idx.append(int(re.split('(ID=".*?")', token)[1][4:-1]))
                        if c_idx[-1] not in cue.keys():
                            cue[c_idx[-1]] = []
                elif '</cue' in token:
                    in_cue = in_cue[:-1]
                elif '<xcope' in token:
                    continue
                elif '</xcope' in token:
                    in_scope = in_scope[:-1]
                elif '<ref' in token:
                    in_scope.append([int(i) for i in re.split('(SRC=".*?")', token)[1][5:-1].split(' ')])
                    s_idx.append([int(i) for i in re.split('(SRC=".*?")', token)[1][5:-1].split(' ')])
                    for i in s_idx[-1]:
                        scope[i] = []
                elif '</SENTENCE' in token:
                    if len(cue.keys()) == 0:
                        cue_only_data.append([sentence, [3] * len(sentence)])
                    else:
                        cue_sentence.append(sentence)
                        cue_cues.append([3] * len(sentence))
                        for i in cue.keys():
                            scope_sentence.append(sentence)
                            scope_cues.append([3] * len(sentence))
                            if len(cue[i]) == 1:
                                cue_cues[-1][cue[i][0]] = 1
                                scope_cues[-1][cue[i][0]] = 1
                            else:
                                for c in cue[i]:
                                    cue_cues[-1][c] = 2
                                    scope_cues[-1][c] = 2
                            scope_scopes.append([0] * len(sentence))
                            if i in scope.keys():
                                for s in scope[i]:
                                    scope_scopes[-1][s] = 1
                    sentence = []
                    cue = {}
                    scope = {}
                    in_scope = []
                    in_cue = []
                    word_num = 0
                    in_word = 0
                    c_idx = []
                    s_idx = []
                elif '<' not in token:
                    if in_word == 1:
                        if len(in_cue) != 0:
                            for i in in_cue:
                                cue[i].append(word_num)
                        if len(in_scope) != 0:
                            for i in in_scope:
                                for j in i:
                                    scope[j].append(word_num)
                        sentence.append(token)
            cue_only_samples = random.sample(cue_only_data, k=int(frac_no_cue_sents * len(cue_only_data)))
            cue_only_sents = [i[0] for i in cue_only_samples]
            cue_only_cues = [i[1] for i in cue_only_samples]
            return [(cue_sentence + cue_only_sents, cue_cues + cue_only_cues),
                    (scope_sentence, scope_cues, scope_scopes)]

        if dataset_name == 'bioscope':
            ret_val = bioscope(file, frac_no_cue_sents=frac_no_cue_sents)
            self.cue_data = Cues(ret_val[0])
            self.scope_data = Scopes(ret_val[1])
        elif dataset_name == 'sfu':
            sfu_cues = [[], []]
            sfu_scopes = [[], [], []]
            for dir_name in os.listdir(file):
                if '.' not in dir_name:
                    for f_name in os.listdir(file + "//" + dir_name):
                        r_val = sfu_review(file + "//" + dir_name + '//' + f_name, frac_no_cue_sents=frac_no_cue_sents)
                        sfu_cues = [a + b for a, b in zip(sfu_cues, r_val[0])]
                        sfu_scopes = [a + b for a, b in zip(sfu_scopes, r_val[1])]
            self.cue_data = Cues(sfu_cues)
            self.scope_data = Scopes(sfu_scopes)
        elif dataset_name == 'starsem':
            if TASK == 'negation':
                ret_val = starsem(file, frac_no_cue_sents=frac_no_cue_sents)
                self.cue_data = Cues(ret_val[0])
                self.scope_data = Scopes(ret_val[1])
            else:
                raise ValueError("Starsem 2012 dataset only supports negation annotations")
        else:
            raise ValueError("Supported Dataset types are:\n\tbioscope\n\tsfu\n\tconll_cue")

    def get_cue_dataloader(self, val_size=0.15, test_size=0.15, other_datasets=[], CUE_MODEL='bert-base-uncased',
                           MAX_LEN=128, bs=32):
        '''
        This function returns the dataloader for the cue detection.
        val_size: The size of the validation dataset (Fraction between 0 to 1)
        test_size: The size of the test dataset (Fraction between 0 to 1)
        other_datasets: Other datasets to use to get one combined train dataloader
        Returns: train_dataloader, list of validation dataloaders, list of test dataloaders
        '''
        do_lower_case = True
        if 'uncased' not in CUE_MODEL:
            do_lower_case = False
        tokenizer = AutoTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case)

        def preprocess_data(obj, tokenizer):
            dl_sents = obj.cue_data.sentences
            dl_cues = obj.cue_data.cues

            sentences = [" ".join(sent) for sent in dl_sents]

            mytexts = []
            mylabels = []
            mymasks = []
            if do_lower_case == True:
                sentences_clean = [sent.lower() for sent in sentences]
            else:
                sentences_clean = sentences
            for sent, tags in zip(sentences_clean, dl_cues):
                new_tags = []
                new_text = []
                new_masks = []
                for word, tag in zip(sent.split(), tags):
                    sub_words = tokenizer.tokenize(word)
                    for count, sub_word in enumerate(sub_words):
                        if type(tag) != int:
                            raise ValueError(tag)
                        mask = 1
                        if count > 0:
                            mask = 0
                        new_masks.append(mask)
                        new_tags.append(tag)
                        new_text.append(sub_word)
                mymasks.append(new_masks)
                mytexts.append(new_text)
                mylabels.append(new_tags)

            input_ids = pad_sequences([[tokenizer.convert_tokens_to_ids(word) for word in txt] for txt in mytexts],
                                      maxlen=MAX_LEN, dtype=torch.long, truncating="post", padding="post").tolist()

            tags = pad_sequences(mylabels,
                                 maxlen=MAX_LEN, value=4, padding="post",
                                 dtype=torch.long, truncating="post").tolist()

            mymasks = pad_sequences(mymasks, maxlen=MAX_LEN, value=0, padding='post', dtype=torch.long,
                                    truncating='post').tolist()

            attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

            random_state = np.random.randint(1, 2019)

            tra_inputs, test_inputs, tra_tags, test_tags = train_test_split(input_ids, tags, test_size=test_size,
                                                                            random_state=random_state)
            tra_masks, test_masks, _, _ = train_test_split(attention_masks, input_ids, test_size=test_size,
                                                           random_state=random_state)
            tra_mymasks, test_mymasks, _, _ = train_test_split(mymasks, input_ids, test_size=test_size,
                                                               random_state=random_state)

            random_state_2 = np.random.randint(1, 2019)

            tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(tra_inputs, tra_tags,
                                                                        test_size=(val_size / (1 - test_size)),
                                                                        random_state=random_state_2)
            tr_masks, val_masks, _, _ = train_test_split(tra_masks, tra_inputs, test_size=(val_size / (1 - test_size)),
                                                         random_state=random_state_2)
            tr_mymasks, val_mymasks, _, _ = train_test_split(tra_mymasks, tra_inputs,
                                                             test_size=(val_size / (1 - test_size)),
                                                             random_state=random_state_2)
            return [tr_inputs, tr_tags, tr_masks, tr_mymasks], [val_inputs, val_tags, val_masks, val_mymasks], [
                test_inputs, test_tags, test_masks, test_mymasks]

        tr_inputs = []
        tr_tags = []
        tr_masks = []
        tr_mymasks = []
        val_inputs = [[] for i in range(len(other_datasets) + 1)]
        test_inputs = [[] for i in range(len(other_datasets) + 1)]

        train_ret_val, val_ret_val, test_ret_val = preprocess_data(self, tokenizer)
        tr_inputs += train_ret_val[0]
        tr_tags += train_ret_val[1]
        tr_masks += train_ret_val[2]
        tr_mymasks += train_ret_val[3]
        val_inputs[0].append(val_ret_val[0])
        val_inputs[0].append(val_ret_val[1])
        val_inputs[0].append(val_ret_val[2])
        val_inputs[0].append(val_ret_val[3])
        test_inputs[0].append(test_ret_val[0])
        test_inputs[0].append(test_ret_val[1])
        test_inputs[0].append(test_ret_val[2])
        test_inputs[0].append(test_ret_val[3])

        for idx, arg in enumerate(other_datasets, 1):
            train_ret_val, val_ret_val, test_ret_val = preprocess_data(arg, tokenizer)
            tr_inputs += train_ret_val[0]
            tr_tags += train_ret_val[1]
            tr_masks += train_ret_val[2]
            tr_mymasks += train_ret_val[3]
            val_inputs[idx].append(val_ret_val[0])
            val_inputs[idx].append(val_ret_val[1])
            val_inputs[idx].append(val_ret_val[2])
            val_inputs[idx].append(val_ret_val[3])
            test_inputs[idx].append(test_ret_val[0])
            test_inputs[idx].append(test_ret_val[1])
            test_inputs[idx].append(test_ret_val[2])
            test_inputs[idx].append(test_ret_val[3])

        tr_inputs = torch.LongTensor(tr_inputs)
        tr_tags = torch.LongTensor(tr_tags)
        tr_masks = torch.LongTensor(tr_masks)
        tr_mymasks = torch.LongTensor(tr_mymasks)
        val_inputs = [[torch.LongTensor(i) for i in j] for j in val_inputs]
        test_inputs = [[torch.LongTensor(i) for i in j] for j in test_inputs]

        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags, tr_mymasks)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

        val_dataloaders = []
        for i, j, k, l in val_inputs:
            val_data = TensorDataset(i, k, j, l)
            val_sampler = RandomSampler(val_data)
            val_dataloaders.append(DataLoader(val_data, sampler=val_sampler, batch_size=bs))

        test_dataloaders = []
        for i, j, k, l in test_inputs:
            test_data = TensorDataset(i, k, j, l)
            test_sampler = RandomSampler(test_data)
            test_dataloaders.append(DataLoader(test_data, sampler=test_sampler, batch_size=bs))

        return train_dataloader, val_dataloaders, test_dataloaders

    def get_scope_dataloader(self, val_size=0.15, test_size=0.15, other_datasets=[], SCOPE_MODEL='xlnet-base-cased',
                             SCOPE_METHOD='augment', MAX_LEN=128, bs=32):
        '''
        This function returns the dataloader for the cue detection.
        val_size: The size of the validation dataset (Fraction between 0 to 1)
        test_size: The size of the test dataset (Fraction between 0 to 1)
        other_datasets: Other datasets to use to get one combined train dataloader
        Returns: train_dataloader, list of validation dataloaders, list of test dataloaders
        '''
        method = SCOPE_METHOD
        do_lower_case = True
        if 'uncased' not in SCOPE_MODEL:
            do_lower_case = False
        tokenizer = AutoTokenizer.from_pretrained(SCOPE_MODEL, do_lower_case=do_lower_case)

        def preprocess_data(obj, tokenizer_obj):
            dl_sents = obj.scope_data.sentences
            dl_cues = obj.scope_data.cues
            dl_scopes = obj.scope_data.scopes

            sentences = [" ".join([s for s in sent]) for sent in dl_sents]
            mytexts = []
            mylabels = []
            mycues = []
            mymasks = []
            if do_lower_case == True:
                sentences_clean = [sent.lower() for sent in sentences]
            else:
                sentences_clean = sentences

            for sent, tags, cues in zip(sentences_clean, dl_scopes, dl_cues):
                new_tags = []
                new_text = []
                new_cues = []
                new_masks = []
                for word, tag, cue in zip(sent.split(), tags, cues):
                    sub_words = tokenizer_obj.tokenize(word)
                    for count, sub_word in enumerate(sub_words):
                        mask = 1
                        if count > 0:
                            mask = 0
                        new_masks.append(mask)
                        new_tags.append(tag)
                        new_cues.append(cue)
                        new_text.append(sub_word)
                mymasks.append(new_masks)
                mytexts.append(new_text)
                mylabels.append(new_tags)
                mycues.append(new_cues)
            final_sentences = []
            final_labels = []
            final_masks = []
            if method == 'replace':
                for sent, cues in zip(mytexts, mycues):
                    temp_sent = []
                    for token, cue in zip(sent, cues):
                        if cue == 3:
                            temp_sent.append(token)
                        else:
                            temp_sent.append(f'[unused{cue + 1}]')
                    final_sentences.append(temp_sent)
                final_labels = mylabels
                final_masks = mymasks
            elif method == 'augment':
                for sent, cues, labels, masks in zip(mytexts, mycues, mylabels, mymasks):
                    temp_sent = []
                    temp_label = []
                    temp_masks = []
                    first_part = 0
                    for token, cue, label, mask in zip(sent, cues, labels, masks):
                        if cue != 3:
                            if first_part == 0:
                                first_part = 1
                                temp_sent.append(f'[unused{cue + 1}]')
                                temp_masks.append(1)
                                temp_label.append(label)
                                temp_sent.append(token)
                                temp_masks.append(0)
                                temp_label.append(label)
                                continue
                            temp_sent.append(f'[unused{cue + 1}]')
                            temp_masks.append(0)
                            temp_label.append(label)
                        else:
                            first_part = 0
                        temp_masks.append(mask)
                        temp_sent.append(token)
                        temp_label.append(label)
                    final_sentences.append(temp_sent)
                    final_labels.append(temp_label)
                    final_masks.append(temp_masks)
            else:
                raise ValueError("Supported methods for scope detection are:\nreplace\naugment")
            input_ids = pad_sequences(
                [[tokenizer_obj.convert_tokens_to_ids(word) for word in txt] for txt in final_sentences],
                maxlen=MAX_LEN, dtype=torch.long, truncating="post", padding="post").tolist()

            tags = pad_sequences(final_labels,
                                 maxlen=MAX_LEN, value=0, padding="post",
                                 dtype=torch.long, truncating="post").tolist()

            final_masks = pad_sequences(final_masks,
                                        maxlen=MAX_LEN, value=0, padding="post",
                                        dtype=torch.long, truncating="post").tolist()

            attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

            random_state = np.random.randint(1, 2019)

            tra_inputs, test_inputs, tra_tags, test_tags = train_test_split(input_ids, tags, test_size=test_size,
                                                                            random_state=random_state)
            tra_masks, test_masks, _, _ = train_test_split(attention_masks, input_ids, test_size=test_size,
                                                           random_state=random_state)
            tra_mymasks, test_mymasks, _, _ = train_test_split(final_masks, input_ids, test_size=test_size,
                                                               random_state=random_state)

            random_state_2 = np.random.randint(1, 2019)

            tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(tra_inputs, tra_tags,
                                                                        test_size=(val_size / (1 - test_size)),
                                                                        random_state=random_state_2)
            tr_masks, val_masks, _, _ = train_test_split(tra_masks, tra_inputs, test_size=(val_size / (1 - test_size)),
                                                         random_state=random_state_2)
            tr_mymasks, val_mymasks, _, _ = train_test_split(tra_mymasks, tra_inputs,
                                                             test_size=(val_size / (1 - test_size)),
                                                             random_state=random_state_2)

            return [tr_inputs, tr_tags, tr_masks, tr_mymasks], [val_inputs, val_tags, val_masks, val_mymasks], [
                test_inputs, test_tags, test_masks, test_mymasks]

        tr_inputs = []
        tr_tags = []
        tr_masks = []
        tr_mymasks = []
        val_inputs = [[] for i in range(len(other_datasets) + 1)]
        test_inputs = [[] for i in range(len(other_datasets) + 1)]

        train_ret_val, val_ret_val, test_ret_val = preprocess_data(self, tokenizer)
        tr_inputs += train_ret_val[0]
        tr_tags += train_ret_val[1]
        tr_masks += train_ret_val[2]
        tr_mymasks += train_ret_val[3]
        val_inputs[0].append(val_ret_val[0])
        val_inputs[0].append(val_ret_val[1])
        val_inputs[0].append(val_ret_val[2])
        val_inputs[0].append(val_ret_val[3])
        test_inputs[0].append(test_ret_val[0])
        test_inputs[0].append(test_ret_val[1])
        test_inputs[0].append(test_ret_val[2])
        test_inputs[0].append(test_ret_val[3])

        for idx, arg in enumerate(other_datasets, 1):
            train_ret_val, val_ret_val, test_ret_val = preprocess_data(arg, tokenizer)
            tr_inputs += train_ret_val[0]
            tr_tags += train_ret_val[1]
            tr_masks += train_ret_val[2]
            tr_mymasks += train_ret_val[3]
            val_inputs[idx].append(val_ret_val[0])
            val_inputs[idx].append(val_ret_val[1])
            val_inputs[idx].append(val_ret_val[2])
            val_inputs[idx].append(val_ret_val[3])
            test_inputs[idx].append(test_ret_val[0])
            test_inputs[idx].append(test_ret_val[1])
            test_inputs[idx].append(test_ret_val[2])
            test_inputs[idx].append(test_ret_val[3])

        tr_inputs = torch.LongTensor(tr_inputs)
        tr_tags = torch.LongTensor(tr_tags)
        tr_masks = torch.LongTensor(tr_masks)
        tr_mymasks = torch.LongTensor(tr_mymasks)
        val_inputs = [[torch.LongTensor(i) for i in j] for j in val_inputs]
        test_inputs = [[torch.LongTensor(i) for i in j] for j in test_inputs]

        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags, tr_mymasks)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

        val_dataloaders = []
        for i, j, k, l in val_inputs:
            val_data = TensorDataset(i, k, j, l)
            val_sampler = RandomSampler(val_data)
            val_dataloaders.append(DataLoader(val_data, sampler=val_sampler, batch_size=bs))

        test_dataloaders = []
        for i, j, k, l in test_inputs:
            test_data = TensorDataset(i, k, j, l)
            test_sampler = RandomSampler(test_data)
            test_dataloaders.append(DataLoader(test_data, sampler=test_sampler, batch_size=bs))

        return train_dataloader, val_dataloaders, test_dataloaders


class CustomData:
    def __init__(self, sentences, cues=None):
        self.sentences = sentences
        self.cues = cues

    def get_cue_dataloader(self, CUE_MODEL='bert-base-uncased', MAX_LEN=128, bs=32):
        do_lower_case = True
        if 'uncased' not in CUE_MODEL:
            do_lower_case = False
        tokenizer = AutoTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case)

        dl_sents = self.sentences
        sentences = dl_sents  # sentences = [" ".join(sent) for sent in dl_sents]

        mytexts = []
        mylabels = []
        mymasks = []
        offsets = []

        if isinstance(sentences[0], Doc):
            if do_lower_case == True:
                sentences_clean = [[token.text.lower() for token in sent] for sent in sentences]
            else:
                sentences_clean = [[token.text for token in sent] for sent in sentences]
        else:
            if do_lower_case == True:
                sentences_clean = [sent.lower().split() for sent in sentences]
            else:
                sentences_clean = [sent.split() for sent in sentences]
        # spacy tokenizes white spaces that get stripped during BERT tokenization.  We need to account for that.
        for sent in sentences_clean:
            new_text = []
            new_masks = []
            new_offsets = []
            token_idx = 0
            for word in sent:
                sub_words = tokenizer.tokenize(word)
                for count, sub_word in enumerate(sub_words):
                    mask = 1
                    if count > 0:
                        mask = 0
                    else:
                        new_offsets.append(token_idx)
                    new_masks.append(mask)
                    new_text.append(sub_word)
                token_idx += 1
            mymasks.append(new_masks)
            mytexts.append(new_text)
            offsets.append(new_offsets)

        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in mytexts],
                                  maxlen=MAX_LEN, dtype=torch.long, truncating="post", padding="post")

        mymasks = pad_sequences(mymasks, maxlen=MAX_LEN, value=0, padding='post', dtype=torch.long,
                                truncating='post').tolist()

        attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

        offsets = pad_sequences(offsets, maxlen=MAX_LEN, value=0, padding='post', dtype=torch.long, truncating='post')

        inputs = torch.LongTensor(input_ids)
        masks = torch.LongTensor(attention_masks)
        mymasks = torch.LongTensor(mymasks)
        myoffsets = torch.LongTensor(offsets)

        data = TensorDataset(inputs, masks, mymasks, myoffsets)
        dataloader = DataLoader(data, batch_size=bs)

        return dataloader

    def get_scope_dataloader(self, cues=None, SCOPE_MODEL='xlnet-base-cased', SCOPE_METHOD='augment', MAX_LEN=128,
                             bs=32):
        if cues != None:
            self.cues = cues
        if self.cues == None:
            raise ValueError("Need Cues Data to Generate the Scope Dataloader")
        method = SCOPE_METHOD
        do_lower_case = True
        if 'uncased' not in SCOPE_MODEL:
            do_lower_case = False
        tokenizer = AutoTokenizer.from_pretrained(SCOPE_MODEL, do_lower_case=do_lower_case)

        dl_sents = self.sentences
        dl_cues = self.cues

        sentences = dl_sents
        mytexts = []
        mycues = []
        mymasks = []
        offsets = []

        if isinstance(sentences[0], Doc):
            if do_lower_case == True:
                sentences_clean = [[token.text.lower() for token in sent] for sent in sentences]
            else:
                sentences_clean = [[token.text for token in sent] for sent in sentences]
        else:
            if do_lower_case == True:
                sentences_clean = [sent.lower().split() for sent in sentences]
            else:
                sentences_clean = [sent.split() for sent in sentences]

        for sent, cues in zip(sentences_clean, dl_cues):
            new_text = []
            new_cues = []
            new_masks = []
            new_offsets = []
            token_idx = 0
            for word, cue in zip(sent, cues):
                sub_words = tokenizer.tokenize(word)
                for count, sub_word in enumerate(sub_words):
                    mask = 1
                    if count > 0:
                        mask = 0
                    else:
                        new_offsets.append(token_idx)
                    new_masks.append(mask)
                    new_cues.append(cue)
                    new_text.append(sub_word)
                token_idx += 1
            mymasks.append(new_masks)
            mytexts.append(new_text)
            mycues.append(new_cues)
            offsets.append(new_offsets)
        final_sentences = []
        final_masks = []
        if method == 'replace':
            for sent, cues in zip(mytexts, mycues):
                temp_sent = []
                for token, cue in zip(sent, cues):
                    if cue == 3:
                        temp_sent.append(token)
                    else:
                        temp_sent.append(f'[unused{cue + 1}]')
                final_sentences.append(temp_sent)
            final_masks = mymasks
        elif method == 'augment':
            for sent, cues, masks in zip(mytexts, mycues, mymasks):
                temp_sent = []
                temp_masks = []
                first_part = 0
                for token, cue, mask in zip(sent, cues, masks):
                    if cue != 3:
                        if first_part == 0:
                            first_part = 1
                            temp_sent.append(f'[unused{cue + 1}]')
                            temp_masks.append(1)
                            # temp_label.append(label)
                            temp_sent.append(token)
                            temp_masks.append(0)
                            # temp_label.append(label)
                            continue
                        temp_sent.append(f'[unused{cue + 1}]')
                        temp_masks.append(0)
                        # temp_label.append(label)
                    else:
                        first_part = 0
                    temp_masks.append(mask)
                    temp_sent.append(token)
                final_sentences.append(temp_sent)
                final_masks.append(temp_masks)
        else:
            raise ValueError("Supported methods for scope detection are:\nreplace\naugment")

        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in final_sentences],
                                  maxlen=MAX_LEN, dtype=torch.long, truncating="post", padding="post")

        final_masks = pad_sequences(final_masks,
                                    maxlen=MAX_LEN, value=0, padding="post",
                                    dtype=torch.long, truncating="post").tolist()

        attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

        offsets = pad_sequences(offsets, maxlen=MAX_LEN, value=0, padding='post', dtype=torch.long, truncating='post')

        inputs = torch.LongTensor(input_ids)
        masks = torch.LongTensor(attention_masks)
        final_masks = torch.LongTensor(final_masks)
        offsets = torch.LongTensor(offsets)

        data = TensorDataset(inputs, masks, final_masks, offsets)
        dataloader = DataLoader(data, batch_size=bs)
        # print(final_sentences, mycues)

        return dataloader
