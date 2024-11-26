from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from torch.hub import load_state_dict_from_url


@dataclass
class Weights:
    url: str


class WeightsEnum(Enum):
    @classmethod
    def verify(cls, obj: Any) -> Any:
        if obj is not None:
            if type(obj) is str:
                obj = cls[obj.replace(cls.__name__ + '.', '')]
            elif not isinstance(obj, cls):
                raise TypeError(
                    f'Invalid Weight class provided; expected {cls.__name__} '
                    f'but received {obj.__class__.__name__}.'
                )
        return obj

    def get_state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return load_state_dict_from_url(self.url, *args, **kwargs)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}.{self._name_}'

    @property
    def url(self):
        return self.value.url