import os
import sys
from typing import Union

import click

from image_transformation.utils.transformation_utils import Transformation


def validate_source(ctx: click.Context,
                    param: click.Parameter, value: str) -> str:
    """
    Validate the source parameter.

    Args:
        ctx (click.Context): The Click context.
        param (click.Parameter): The Click parameter.
        value (str): The value of the source parameter.

    Returns:
        str: The validated source value.

    Raises:
        click.BadParameter: If the source is not a
            directory or a .jpg file.
    """
    if os.path.isdir(value):
        return value
    elif value and not value.casefold().endswith('.jpg'):
        raise click.BadParameter('Source file must be a .jpg image.')
    return value


def validate_directory(ctx: click.Context, param: click.Parameter,
                       value: Union[str, None]) -> Union[str, None]:
    """
    Validate the destination directory parameter.

    Args:
        ctx (click.Context): The Click context.
        param (click.Parameter): The Click parameter.
        value (Union[str, None]): The value of the destination directory parameter.

    Returns:
        Union[str, None]: The validated directory value.

    Raises:
        click.BadParameter: If the directory cannot be created or is invalid.
    """
    if value:
        try:
            os.makedirs(value, exist_ok=True)
            if not os.path.isdir(value):
                raise click.BadParameter(f'{value} is not a valid directory.')
        except Exception as e:
            raise click.BadParameter(f"Couldn't create directory {value}: {e}") from e
    return value


@click.command()
@click.option('--src', required=True,
              help='Source image or directory', callback=validate_source)
@click.option('--dst',
              help='Destination directory', callback=validate_directory)
def transformation(src: str, dst: Union[str, None]) -> None:
    """
     Perform the transformation based on the source and destination.

     Args:
         src (str): The source image or directory.
         dst (Union[str, None]): The destination directory.

     Raises:
         click.UsageError: If the source is a directory and the destination is not provided.
     """
    if os.path.isfile(src):
        click.echo(f'Source file is a valid .jpg image: {src}')
        Transformation(image_path=src)
        sys.exit(0)

    elif os.path.isdir(src):
        if not dst:
            raise click.UsageError(
                'If --src is a directory,'
                '--dst must also be specified and must be a directory.')
        click.echo(f'Source directory: {src}')
        click.echo(f'Destination directory: {dst}')
        Transformation(input_dir=src, output_dir=dst)
        sys.exit(0)

    else:
        raise click.UsageError(
            'Invalid source. Must be either a .jpg image or a directory.')


if __name__ == '__main__':
    transformation()
