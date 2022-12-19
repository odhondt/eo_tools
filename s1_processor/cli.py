import click


@click.command()
@click.option('--config-file', '-c', required=False, type=click.Path(),
              help='Full path to an INI-style configuration text file.')
@click.option('--version', is_flag=True,
              help='Print S1_NRB version information. Overrides all other arguments.')
def cli(config_file, section, debug, version):
    import processor
    if version:
        print(processor.__version__)
    else:
        processor.process(config_file=config_file)