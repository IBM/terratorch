from terratorch.datasets import HLSBands

def _are_sublists_of_int(item) -> bool:

    if all([isinstance(i, list) for i in item]):
        if all([isinstance(i, int) for i in sum(item, [])]):
            return True
        else:
            return False
    else:
        return False

def _estimate_in_chans(model_bands: list[HLSBands] | list[str] | tuple[int, int] = None) -> int:

    # Conditional to deal with the different possible choices for the bands 
    # Bands as lists of strings or enum
    if all([isinstance(b, str) or isinstance(b, HLSBands) for b in model_bands]): 
        in_chans = len(model_bands)
    # Bands as intervals limited by integers
    elif all([isinstance(b, int) for b in model_bands] or _are_sublists_of_int(model_bands)):

        if _are_sublists_of_int(model_bands):
            in_chans = sum([i[-1] - i[0] for i in model_bands])
        else:
            in_chans = model_bands[-1] - model_bands[0]
    else:
        raise Exception(f"Expected bands to be list(str) or [int, int] but received {model_bands}")
    
    return in_chans


