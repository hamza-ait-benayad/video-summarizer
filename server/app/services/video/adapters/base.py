from abc import ABC, abstractmethod
from typing import Any


class BaseVideoAdapter(ABC):

    @abstractmethod
    def get_metadata(self, url: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_transcript(self, url: str) -> list[dict]:
        pass
