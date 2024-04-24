
import abc
from typing import Any

from ..dataset.dataset import Dataset


class Algorithm(abc.ABC):

    def __init__(self):

        return

    @abc.abstractmethod
    def run(self, dataset: Dataset) -> Any:

        return
