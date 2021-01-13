# IGRE CLI scripts

As time flows some of our algirthms stabilize in sense of 
- *algorithm* - architecture of ANN, processing pipeline
- *parametrization* - especially optimizers
- *usage* - dimensionality of inputs, ...

When this happen, it is useful to create CLI support for such algorithms here in scripts folder. Let be the standard here usage of [Click](https://click.palletsprojects.com) tool.

## Test coverage

To prevent functionality of CLI in the future, it is recommended to cover all procedures here with tests or move them into `*.stable` package.