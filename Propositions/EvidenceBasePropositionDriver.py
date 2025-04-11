import importlib
import os
import sys
from abc import ABC, abstractmethod

modules = {key: value for key, value in sys.modules.items() if 'EvidenceBaseModelDriver' in key}

if modules:
    module = list(modules.values())[0]
    EvidenceBaseModelDriver = getattr(module, 'EvidenceBaseModelDriver')
else:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Models')))
    from EvidenceBaseModelDriver import EvidenceBaseModelDriver

modules = {key: value for key, value in sys.modules.items() if 'EvidenceBaseNegationDriver' in key}

if modules:
    module = list(modules.values())[0]
    EvidenceBaseNegationDriver = getattr(module, 'EvidenceBaseNegationDriver')
else:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Negations')))
    from EvidenceBaseNegationDriver import EvidenceBaseNegationDriver


class EvidenceBasePropositionDriver(ABC):
    available_drivers = {}
    default_driver = "V0"

    """
    Abstract class that defines the methods that must be implemented by a driver for a specific evidence base proposition generator.
    """

    @abstractmethod
    def __init__(self, model_driver: EvidenceBaseModelDriver, negation_driver: EvidenceBaseNegationDriver, config):
        self.driver_config = config
        self.driver_defaults = self.driver_config.copy()

        # we'll use this spacy model to split the input into sentences throughout all the drivers
        self.nlp = model_driver.nlp
        self.model_driver = model_driver
        self.negation_driver = negation_driver

    # this is triggered when a subclass is created
    # it's used to register the driver in the available_drivers dictionary
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "_identifier" in cls.__dict__:
            EvidenceBasePropositionDriver.available_drivers[cls._identifier] = cls
        else:
            raise ValueError(f"Class {cls.__name__} does not define an identifier")

    @abstractmethod
    def buildPropositions(self, model_predictions):
        pass

    # autoloads all the drivers in the Models directory
    @classmethod
    def import_subclasses(cls):
        parent_directory = os.path.abspath(os.path.dirname(__file__))
        for child_dir_name in os.listdir(parent_directory):
            if child_dir_name == "__pycache__":
                continue
            child_dir_path = os.path.join(parent_directory, child_dir_name)
            if os.path.isdir(child_dir_path):
                module_path = f"Propositions.{child_dir_name}.PropositionDriver"
                try:
                    importlib.import_module(module_path)
                except (ModuleNotFoundError, AttributeError) as e:
                    print(f"[WARN]: Failed to import module {module_path}: {e}")
                    continue

    # loads a driver by its identifier, and initializes it with the model and config
    @classmethod
    def load_driver(cls, identifier, model, negation, config):
        driver_class = cls.available_drivers.get(identifier)
        if driver_class:
            return driver_class(model, negation, config)
        else:
            raise ValueError(f"No driver found with identifier {identifier}")


EvidenceBasePropositionDriver.import_subclasses()
