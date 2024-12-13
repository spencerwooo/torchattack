# Development

uv is used for development.

First, [install uv](https://docs.astral.sh/uv/getting-started/installation/).

## Installing Dependencies

Next, install dev dependencies.

```shell
uv sync --dev
```

### Installing Test Dependencies

Install dependency group `test` to run tests.

```shell
uv sync --dev --group test
```

### Installing Documentation Dependencies

Install dependency group `docs` to build documentation.

```shell
uv sync --dev --group docs
```

### Installing All Dependencies

To install everything, run:

```shell
uv sync --all-groups
```

## Running Tests

To run tests:

```shell
pytest torchattack
```

## Additional Information

For more details on using uv, refer to the [uv documentation](https://docs.astral.sh/uv/).
