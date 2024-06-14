import os
import sys
from typing import Union

import click

from data_transformation.utils.transformation_class import Transformation


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
        value (Union[str, None]): The value of the
            destination directory parameter.

    Returns:
        Union[str, None]: The validated directory value.

    Raises:
        click.BadParameter: If the directory cannot
            be created or is invalid.
    """
    if value:
        try:
            os.makedirs(value, exist_ok=True)
            if not os.path.isdir(value):
                raise click.BadParameter(f'{value} is not a valid directory.')
        except Exception as e:
            raise click.BadParameter(
                f"Couldn't create directory {value}: {e}") from e
    return value


def transformation(src: str, dst: str | None = None,
                   keep_dir_structure: bool = False) -> None:
    """
     Perform the transformation based on the
     source and destination.

     Args:
         src (str): The source image or directory.
         dst (Union[str, None]): The destination directory.
         keep_dir_structure (bool): Whether to keep the
            directory structure.

     Raises:
         click.UsageError: If the source is a directory
            and the destination is not provided.
     """
    if os.path.isfile(src):
        if dst:
            raise click.UsageError(
                'If --src is a file, --dst must not be specified.')
        Transformation(image_path=src)
        sys.exit(0)

    elif os.path.isdir(src):
        if not dst:
            raise click.UsageError(
                'If --src is a directory,'
                '--dst must also be specified and must be a directory.')
        Transformation(
            input_dir=src, output_dir=dst,
            keep_dir_structure=keep_dir_structure)
        sys.exit(0)

    else:
        raise click.UsageError(
            'Invalid source. Must be either a .jpg image or a directory.')


@click.command()
@click.option('--src', required=True,
              help='Source image or directory', callback=validate_source)
@click.option('--dst',
              help='Destination directory', callback=validate_directory)
def cli_transformation(src: str, dst: str | None = None) -> None:
    transformation(src, dst)


if __name__ == '__main__':
    img_path = '../leaves/images/Apple_rust/image (2).JPG'
    src_dir = '../leaves/images/Apple_rust'
    dst_dir = '../output/Apple_rust'
    transformation(src=src_dir, dst=dst_dir, keep_dir_structure=True)
