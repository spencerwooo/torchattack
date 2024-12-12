# Development

uv is used for development.

First, [install uv](https://docs.astral.sh/uv/getting-started/installation/).

Next, install dev dependencies.

```shell
uv sync --dev
```

Install dependency group `test` to run tests.

```shell
uv sync --dev --group test
```

Install dependency group `docs` to build documentation.

```shell
uv sync --dev --group docs
```

Install dependency group `test` to run tests.

```shell
uv sync --dev --group test
```

To install everything, run:

```shell
uv sync --all-groups
```

To run tests:

```shell
pytest torchattack
```
