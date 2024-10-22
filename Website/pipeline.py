from SentenceClassification.EvidenceBaseSentenceClassificationDriver import EvidenceBaseSentenceClassificationDriver
from Models.EvidenceBaseModelDriver import EvidenceBaseModelDriver
from Negations.EvidenceBaseNegationDriver import EvidenceBaseNegationDriver
from Propositions.EvidenceBasePropositionDriver import EvidenceBasePropositionDriver
from EvidenceMap.EvidenceMapDriver import EvidenceMapDriver
from EvidenceConfig import PIPELINE_CONFIG

SentenceClassificationMainConfig = PIPELINE_CONFIG.get('SentenceClassification', {})
SentenceClassificationPreferredDriver = SentenceClassificationMainConfig.get('preferred-driver', 'V0')
SentenceClassificationConfig = SentenceClassificationMainConfig.get(SentenceClassificationPreferredDriver, {}).get('config', {})

ModelMainConfig = PIPELINE_CONFIG.get('BaseNLP', {})
ModelPreferredDriver = ModelMainConfig.get('preferred-driver', 'V2')
ModelBase = ModelMainConfig.get(ModelPreferredDriver, {}).get('base_model', 'dmis-lab/biobert-base-cased-v1.2')
ModelConfig = ModelMainConfig.get(ModelPreferredDriver, {}).get('config', {})

NegationMainConfig = PIPELINE_CONFIG.get('Negation', {})
NegationPreferredDriver = NegationMainConfig.get('preferred-driver', 'V1')
NegationConfig = NegationMainConfig.get(NegationPreferredDriver, {}).get('config', {})

PropositionMainConfig = PIPELINE_CONFIG.get('Proposition', {})
PropositionPreferredDriver = PropositionMainConfig.get('preferred-driver', 'V0')
PropositionConfig = PropositionMainConfig.get(PropositionPreferredDriver, {}).get('config', {})

EvidenceMapMainConfig = PIPELINE_CONFIG.get('EvidenceMap', {})
EvidenceMapPreferredDriver = EvidenceMapMainConfig.get('preferred-driver', 'V1')
EvidenceMapConfig = EvidenceMapMainConfig.get(EvidenceMapPreferredDriver, {}).get('config', {})

# Instantiate the pipeline
SentenceClassification = EvidenceBaseSentenceClassificationDriver.load_driver(SentenceClassificationPreferredDriver, SentenceClassificationConfig)
Model = EvidenceBaseModelDriver.load_driver(ModelPreferredDriver, ModelBase, ModelConfig)
NegationDriver = EvidenceBaseNegationDriver.load_driver(NegationPreferredDriver, Model, NegationConfig)
PropositionDriver = EvidenceBasePropositionDriver.load_driver(PropositionPreferredDriver, Model, NegationDriver, PropositionConfig)
EvidenceMapDriver = EvidenceMapDriver.load_driver(EvidenceMapPreferredDriver, PropositionDriver, EvidenceMapConfig)