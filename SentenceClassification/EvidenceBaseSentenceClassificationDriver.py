import importlib
import os
from abc import ABC, abstractmethod


class EvidenceBaseSentenceClassificationDriver(ABC):
    available_drivers = {}
    default_driver = "V1"

    """
    Abstract class that defines the methods that must be implemented by a driver for a specific evidence base model.
    """

    @abstractmethod
    def __init__(self, config):
        self.driver_config = config
        self.driver_defaults = self.driver_config.copy()

    # this is triggered when a subclass is created
    # it's used to register the driver in the available_drivers dictionary
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "_identifier" in cls.__dict__:
            EvidenceBaseSentenceClassificationDriver.available_drivers[cls._identifier] = cls
        else:
            raise ValueError(f"Class {cls.__name__} does not define an identifier")

    @abstractmethod
    def classifySentences(self, inputs):
        pass

    # autoloads all the drivers in the Models directory
    @classmethod
    def import_subclasses(cls):
        parent_directory = os.path.abspath(os.path.dirname(__file__))
        for child_dir_name in os.listdir(parent_directory):
            if child_dir_name == "__pycache__":
                continue
            if child_dir_name == "Training":
                continue
            child_dir_path = os.path.join(parent_directory, child_dir_name)
            if os.path.isdir(child_dir_path):
                module_path = f"SentenceClassification.{child_dir_name}.SentenceClassificationDriver"
                try:
                    importlib.import_module(module_path)
                except (ModuleNotFoundError, AttributeError) as e:
                    print(f"[WARN]: Failed to import module {module_path}: {e}")
                    continue

    @classmethod
    def load_driver(cls, identifier, config):
        driver_class = cls.available_drivers.get(identifier)
        if driver_class:
            return driver_class(config)
        else:
            raise ValueError(f"No driver found with identifier {identifier}")


EvidenceBaseSentenceClassificationDriver.import_subclasses()
