from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Union

from torch.hub import load_state_dict_from_url


@dataclass
class GeneratorWeights:
    url: str
    inception: bool = False


class GeneratorWeightsEnum(Enum):
    @classmethod
    def verify(
        cls, obj: Union['GeneratorWeightsEnum', str, None]
    ) -> Union['GeneratorWeightsEnum', None]:
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

    def __eq__(self, other: Any) -> bool:
        other = self.verify(other)
        return isinstance(other, self.__class__) and self.name == other.name

    def _assert_generator_weights(self) -> GeneratorWeights:
        if not isinstance(self.value, GeneratorWeights):
            raise TypeError(
                f'Expected GeneratorWeights, but got {type(self.value).__name__}'
            )
        return self.value

    @property
    def url(self) -> str:
        return self._assert_generator_weights().url

    @property
    def inception(self) -> bool:
        return self._assert_generator_weights().inception
