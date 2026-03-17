"""Base abstract class defining the interface for all multimodal models."""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class establishing the core interface for multimodal models.

    This class enforces a consistent structure for all specialized model
    implementations, ensuring they handle input preparation and response
    generation in a unified way.
    """

    @abstractmethod
    def prepare_input(self, inputs):
        """
        Process raw multimedia data into model-compatible tensors.

        Parameters
        ----------
        inputs : list of dict
            A collection of input samples, where each dictionary contains
            keys like 'text', 'video', or 'audio'.

        Returns
        -------
        Any
            The formatted tensors or model inputs ready for inference.
        """
        pass

    @abstractmethod
    def generate(self, inputs):
        """
        Execute the model's inference logic to produce a response.

        Parameters
        ----------
        inputs : Any
            The processed inputs returned by `prepare_input`.

        Returns
        -------
        tuple
            A tuple containing (text_response, optional_audio_response).
        """
        pass
