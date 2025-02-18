import warnings

warnings.warn(
    'The module `torchattack.eval` is deprecated and will be removed in a future version. '
    'Please use `torchattack.evaluate` instead.',
    DeprecationWarning,
    stacklevel=2,
)
from torchattack.evaluate import *  # noqa: E402, F403
