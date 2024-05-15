import albumentations as A
from typing import Iterable


def wrap_in_compose_is_list(transform_list):
    # set check shapes to false because of the multitemporal case
    return A.Compose(transform_list, is_check_shapes=False) if isinstance(transform_list, Iterable) else transform_list