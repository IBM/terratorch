from terratorch.datasets import HLSBands

def _are_sublists_of_int(item) -> (bool, bool):

    if all([isinstance(i, list) for i in item]):
        if all([isinstance(i, int) for i in sum(item, [])]):
            return True, True
        else:
            raise Exception(f"It's expected sublists be [int, int], but rceived {model_bands}")
    elif len(item) == 2 and type(item[0]) == type(item[1]) == int:
        return False, True
    else:
        return False, False

def _estimate_in_chans(model_bands: list[HLSBands] | list[str] | tuple[int, int] = None) -> int:

    # Conditional to deal with the different possible choices for the bands 
    # Bands as lists of strings or enum
    is_sublist, requires_special_eval = _are_sublists_of_int(model_bands)

    # Bands as intervals limited by integers
    # The bands numbering follows the Python convention (starts with 0)
    # and includes the extrema (so the +1 in order to include the last band)
    if requires_special_eval:

        if is_sublist:
            in_chans = sum([i[-1] - i[0] + 1 for i in model_bands])
        else:
            in_chans = model_bands[-1] - model_bands[0] + 1
    else:
        in_chans = len(model_bands) 

    return in_chans


