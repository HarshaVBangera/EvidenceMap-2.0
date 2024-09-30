from Models.EvidenceBaseModelDriver import EvidenceBaseModelDriver

# A simple training utility to train SpERT.  Play around as needed to enhance performance.  Uses Huggingface models, or path to local saved model as the base

ModelConfig = {
    'corpora_group': 'Combined',
    'split_ratio': (0.8, 0.1, 0.1),
    'neg_entity_count': 100,
    'neg_relation_count': 100,
    'epochs': 60,
    'split_mode': 'sentence',
    'no_overlapping': False,
    'max_pairs': 10,
    'rel_filter_threshold': 0.3,
    'lowercase': True,
    'sampling_processes': 16
}

ModelsToTrain = [
    #    'dmis-lab/biobert-base-cased-v1.2',
    #    'Tsubasaz/clinical-pubmed-bert-base-512',
    #    'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
    "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"
]

data_prepared = False

for ModelToTrain in ModelsToTrain:
    Model = EvidenceBaseModelDriver.load_driver("V2", ModelToTrain, ModelConfig)
    if data_prepared == False:
        Model.convertRawDataToModelInput()
        data_prepared = True
    Model.trainModel()
