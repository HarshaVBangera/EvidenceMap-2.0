import os
import re
import sys
import time
from io import StringIO

import GPUtil
import evaluate
import numpy as np
import py7zr
import torch
from datasets import Dataset, Features, Value, ClassLabel, DatasetDict
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, TrainingArguments, Trainer, \
    DataCollatorWithPadding, EarlyStoppingCallback

modules = {key: value for key, value in sys.modules.items() if 'EvidenceBaseSentenceClassificationDriver' in key}

if modules:
    module = list(modules.values())[0]
    EvidenceBaseSentenceClassificationDriver = getattr(module, 'EvidenceBaseSentenceClassificationDriver')
else:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from EvidenceBaseSentenceClassificationDriver import EvidenceBaseSentenceClassificationDriver


class V1SentenceClassificationDriver(EvidenceBaseSentenceClassificationDriver):
    _identifier = "V1"

    _tag2id = {
        "OBJECTIVE": 0,
        "METHODS": 1,
        "RESULTS": 2,
        "BACKGROUND": 3,
        "CONCLUSIONS": 4
    }

    def __init__(self, config):
        super().__init__(config)

        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        self.driver_config['model'] = config.get('model', 'NeuML/pubmedbert-base-embeddings')

        self.corpora_dir = os.path.join(os.path.dirname(base_path), 'Training')
        self.training_dir = os.path.join(base_path, 'Training')
        self.model_dir = os.path.join(base_path, 'Models')
        self.tokenizer_dir = os.path.join(base_path, 'Tokenizer')

        self._id2tag = {id: tag for tag, id in self._tag2id.items()}

        self._load_model()

    def _excluding(self, line):
        filtered = ["©", "copyright", "Copyright", "COPYRIGHT", 'J Drugs Dermatol']
        for item in filtered:
            if item in line:
                return True
        return False

    def classifySentences(self, inputs, spacy=None):
        sentences = []
        docs = []

        def process_doc(doc):
            modified_sentences = []
            modified_docs = []

            for idx, sentence in enumerate(doc):
                excluding = self._excluding(sentence.text) and idx == len(doc) - 1
                excluding = excluding or not sentence.text

                if not excluding:
                    new_text = re.sub(r"^\s*[A-Z\s]+[:\-]\s*", "", sentence.text)

                    if spacy is not None:
                        if new_text != sentence.text:
                            modified_sentence = spacy(new_text)
                        else:
                            modified_sentence = sentence.as_doc()
                        modified_docs.append(modified_sentence)
                    modified_sentences.append(new_text)
            return modified_sentences, modified_docs

        start_overall = time.perf_counter()
        if isinstance(inputs[0], list):
            for doc in tqdm(inputs, desc="Preprocessing inputs"):
                doc_sentences, doc_modified_docs = process_doc(doc)
                sentences.append(doc_sentences)
                if spacy:
                    docs.append(doc_modified_docs)
        else:
            sentences, doc_modified_docs = process_doc(inputs)
            if spacy:
                docs.append(doc_modified_docs)

        sentences_len = [len(x) for x in sentences]
        sentences_flattened = [item for sublist in sentences for item in sublist]

        predictions_flattened = self._predict(sentences_flattened)
        predictions_flattened = [self._id2tag[pred] for pred in predictions_flattened]

        start = 0
        predictions = []
        for length in sentences_len:
            tmp = predictions_flattened[start:start + length]
            tmp[0] = "TITLE"
            start += length
            predictions.append(tmp)
        overall_time = time.perf_counter() - start_overall

        overall_average = overall_time / len(sentences_flattened) * 1000

        print(f"[Sentence Classification]: Average time: {overall_average:.4f} ms/sentence")
        if spacy is not None:
            return docs, predictions
        else:
            return sentences, predictions

    def _choose_best_gpu(self):
        self._available_gpus = []
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                # Test if GPU can be used
                torch.randn(1).cuda()
                self._available_gpus.append(i)
            except Exception as e:
                print(f"[Sentence Classification]: [WARN]: GPU {i} is faulty: {e}")

        if not self._available_gpus:
            raise RuntimeError("[Sentence Classification]: [ERROR]: No functional CUDA devices found.")

        # Get GPU with most free memory
        max_free_mem = 0
        best_gpu = self._available_gpus[0]

        self._gpu_count = len(self._available_gpus)

        try:
            gpus = GPUtil.getGPUs()

            for gpu in self._available_gpus:
                free_mem = next((x.memoryFree for x in gpus if x.id == gpu), 0)

                if free_mem > max_free_mem:
                    max_free_mem = free_mem
                    best_gpu = gpu
        except Exception as e:
            print(f"[Sentence Classification]: [WARN]: GPU information retrieval failed: {e}")
            print(f"[Sentence Classification]: [ERROR]: {e}")

            max_vram = 0
            for gpu in self._available_gpus:
                try:
                    torch.cuda.set_device(gpu)
                    vram = torch.cuda.get_device_properties(gpu).total_memory
                    if vram > max_vram:
                        max_vram = vram
                        best_gpu = gpu
                except Exception as vram_e:
                    print(f"[Sentence Classification]: [WARN]: Unable to get VRAM for GPU {gpu}: {vram_e}")

        self._available_gpus.remove(best_gpu)
        self._available_gpus.insert(0, best_gpu)
        self._device = torch.device(f'cuda:{best_gpu}')

        return best_gpu

    def _load_model(self):
        model_path = os.path.join(self.model_dir, self.driver_config['model'])
        tokenizer_path = os.path.join(self.tokenizer_dir, self.driver_config['model'])

        self.require_training = False

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            config = AutoConfig.from_pretrained(model_path)
            self.max_length = config.max_position_embeddings

            if torch.cuda.is_available():
                try:
                    gpu_id = self._choose_best_gpu()
                    self.model = torch.nn.DataParallel(self.model, device_ids=self._available_gpus).to(self._device)
                    print(f"[Sentence Classification]: Primary device: GPU {gpu_id}")
                except RuntimeError as e:
                    print(
                        f"[Sentence Classification]: [ERROR]: CUDA devices detected but can't be accessed.  This could indicate a hardware failure.")
                    print(f"[Sentence Classification]: [WARN]: Falling back to CPU operation.")
                    self._device = torch.device('cpu')
                    print(f"[Sentence Classification]: Primary device: CPU")
            else:
                self._device = torch.device('cpu')
                print(f"[Sentence Classification]: Primary device: CPU")

        except Exception as e:
            self.require_training = True

        if self.require_training:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.driver_config['model'], num_labels=5)
            self.tokenizer = AutoTokenizer.from_pretrained(self.driver_config['model'])
            config = AutoConfig.from_pretrained(self.driver_config['model'])
            self.max_length = config.max_position_embeddings

            print(f"[Sentence Classification]: Model not found, training new model...")

            self._prepare_training_data()

            self._train_model()
            self._evaluate_model()

            print(f"[Sentence Classification]: Training complete.")
            print(
                f"[Sentence Classification]: The model has been saved and as such the training process will not be repeated.")

        print(f"[Sentence Classification]: Model loaded.")

    def _prepare_training_data(self):
        try:
            self.dataset = DatasetDict.load_from_disk(self.training_dir)
            print("[Sentence Classification]: Dataset loaded successfully.")
        except:
            print("[Sentence Classification]: No prebuilt dataset found, preparing a new dataset...")
            self.dataset = self._convert_raw_to_dataset()
            self.dataset.save_to_disk(self.training_dir)
            print("[Sentence Classification]: Dataset saved to disk and loaded successfully.")

    def _train_model(self):
        self.pad_token = self.tokenizer.pad_token
        if self.pad_token is not None:
            print(f"[Sentence Classification]: Pad token is present: {self.pad_token}")
        else:
            print("[Sentence Classification]: No pad token is set for this tokenizer.")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model.resize_token_embeddings(len(self.tokenizer))

        self._accuracy = evaluate.load('accuracy')
        self._recall = evaluate.load('recall')
        self._precision = evaluate.load('precision')
        self._f1 = evaluate.load('f1')

        print(f"[Sentence Classification]: Prepare training data...")
        self._preprocessed_dataset = self.dataset.map(self._preprocess_dataset, batched=True)
        print(f"[Sentence Classification]: Training data prepared.")
        print(f"[Sentence Classification]: Training model...")

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=os.path.join(self.model_dir, self.driver_config['model']),
            learning_rate=self.driver_config.get('learning_rate', 2e-5),
            num_train_epochs=self.driver_config.get('num_train_epochs', 60),
            max_steps=-1,
            per_device_train_batch_size=self.driver_config.get('per_device_train_batch_size', 4),
            per_device_eval_batch_size=self.driver_config.get('per_device_eval_batch_size', 1),
            weight_decay=self.driver_config.get('weight_decay', 0.01),
            warmup_steps=self.driver_config.get('warmup_steps', 10),
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            save_total_limit=1
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self._preprocessed_dataset["train"],
            eval_dataset=self._preprocessed_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        self.trainer.train()
        self.trainer.save_model(os.path.join(self.model_dir, self.driver_config['model']))
        self.tokenizer.save_pretrained(os.path.join(self.tokenizer_dir, self.driver_config['model']))

    def _predict(self, sentences):
        if len(sentences) > 5000:
            print("[Sentence Classification]: Tokenizing inputs...")
        inputs = self.tokenizer(sentences, truncation=True, padding='max_length',
                                max_length=self.max_length, return_tensors='pt')

        all_predictions = []

        if not torch.cuda.is_available():
            with torch.no_grad():
                all_predictions = self.model(**inputs).logits.argmax(dim=-1).tolist()
        else:
            batch_size = self.driver_config.get('per_device_eval_batch_size', 1) * self._gpu_count
            for i in tqdm(range(0, len(inputs['input_ids']), batch_size), desc="Sentence Classification"):
                try:
                    batch_inputs = {name: tensor[i:i + batch_size].to(self._device) for name, tensor in inputs.items()}
                    with torch.no_grad():
                        batch_predictions = self.model(**batch_inputs).logits.argmax(dim=-1).tolist()
                except torch.cuda.OutOfMemoryError as e:
                    print("[Sentence Classification]: [WARN]: Caught CUDA OOM, attempting to clear cache and retry...")
                    print(e)
                    torch.cuda.empty_cache()
                    try:
                        with torch.no_grad():
                            batch_predictions = self.model(**batch_inputs).logits.argmax(dim=-1).tolist()
                    except torch.cuda.OutOfMemoryError as e:
                        print(
                            "[Sentence Classification]: [WARN]: Retried and still caught CUDA OOM, moving to CPU and retrying...")
                        print(e)
                        try:
                            # Extract the original model from DataParallel
                            primary_device = self._device
                            model_actual = self.model.module if isinstance(self.model,
                                                                           torch.nn.DataParallel) else self.model
                            current_device = next(model_actual.parameters()).device

                            model_actual.to('cpu')
                            batch_inputs = {name: tensor[i:i + batch_size].to('cpu') for name, tensor in inputs.items()}
                            with torch.no_grad():
                                batch_predictions = self.model(**batch_inputs).logits.argmax(dim=-1).tolist()
                            print(
                                "[Sentence Classification]: Successfully ran on CPU. Moving back to CUDA for the next iteration.")

                            # Move the model back to the original device
                            model_actual.to(current_device)
                            if primary_device.type == 'cuda':
                                self.model.to(primary_device)
                        except Exception as e:
                            print(f"[Sentence Classification]: [ERROR]: Failed after moving to CPU: {e}")
                            raise e
                except Exception as e:
                    print(f"[Sentence Classification]: [ERROR]: Failed due to unexpected error: {e}")
                    raise e
                all_predictions.extend(batch_predictions)

        return all_predictions

    def _compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        metrics_result = {}
        metrics_result['accuracy'] = self._accuracy.compute(predictions=predictions, references=labels)['accuracy']
        metrics_result['precision'] = \
            self._precision.compute(predictions=predictions, references=labels, average="weighted")['precision']
        metrics_result['recall'] = self._recall.compute(predictions=predictions, references=labels, average="weighted")[
            'recall']
        metrics_result['f1'] = self._f1.compute(predictions=predictions, references=labels, average="weighted")['f1']

        return metrics_result

    def _evaluate_model(self):
        result = self.trainer.evaluate(self._preprocessed_dataset['test'])

        print(result)
        return result

    def _extract_data(self, file_name):
        filepath = os.path.join(self.corpora_dir, file_name)
        data = ""
        with py7zr.SevenZipFile(filepath, mode='r') as z:
            for name, f in z.readall().items():
                data += f.read().decode()
            return StringIO(data)

    def _read_data(self, file):
        documents, labels, document_ids = [], [], []
        lines = file.readlines()
        doc, doc_labels, doc_ids = [], [], []
        for line in lines:
            if line.strip():
                try:
                    label, text = line.strip().split('\t')
                    doc.append(text.strip())
                    doc_labels.append(label.strip())
                    doc_ids.append(document_id)
                except:
                    if line.startswith("###"):
                        document_id = line.replace("#", "").strip()
                    continue
            else:
                documents.extend(doc)
                labels.extend(doc_labels)
                document_ids.extend(doc_ids)
                doc, doc_labels, doc_ids = [], [], []
        if doc:
            documents.extend(doc)
            labels.extend(doc_labels)
            document_ids.extend(doc_ids)

        return documents, labels, document_ids

    def _create_dataset(self, file):
        dataset_dict = {
            'sentences': [],
            'labels': [],
            'document_id': []
        }
        features = Features({
            'sentences': Value(dtype='string'),
            'labels': ClassLabel(names=list(self._tag2id.keys())),
            'document_id': Value(dtype='string')
        })
        documents, labels, ids = self._read_data(file)
        labels = self._map_tags_to_ids(labels)
        dataset_dict['sentences'] = documents
        dataset_dict['labels'] = labels
        dataset_dict['document_id'] = ids
        dataset = Dataset.from_dict(dataset_dict, features=features)

        return dataset

    def _preprocess_dataset(self, examples):
        input_ids = self.tokenizer(examples['sentences'], truncation=True, return_tensors='pt', padding='max_length',
                                   max_length=self.max_length)
        input_ids['labels'] = examples['labels']

        return input_ids

    def _convert_raw_to_dataset(self):
        with self._extract_data('train.7z') as train_file, open(os.path.join(self.corpora_dir, 'test.txt'),
                                                                'r') as test_file, open(
            os.path.join(self.corpora_dir, 'dev.txt'),
            'r') as dev_file:
            train_dataset = self._create_dataset(train_file)
            test_dataset = self._create_dataset(test_file)
            validation_dataset = self._create_dataset(dev_file)

        combined_dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset,
            'validation': validation_dataset
        })

        return combined_dataset

    def _map_tags_to_ids(self, tags):
        return [self._tag2id[tag] for tag in tags]