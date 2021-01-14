# IGRE CLI tools

CLI tools enable image processing without any IDE and without hardcoding of parameters. Once a method has stable
- **algorithm** - processing pipeline, useful success rate, ...
- **parametrization** - e.g. ANN architecture, optimizers config, ...
- **usage** - input and output types

It is put in this package, it SHOULD be equipped with tests and CLI SHOULD be defined.

## Contributing

- Use [Click](https://click.palletsprojects.com) for CLI definition.
- Beware of cryptic names i.e. *processor, manager, filter* are forbidden. Full name of the `<package>.<script>:<method>` SHOULD be selfdocumenting.
- Check your script `--help` twice
- Github checks MUST pass before merge
- Leave a note about addes CLI in project root [README.md](../README.md)

## Future

Once the value of scripts here will be non-zero we elevate this package into PyPI ;)