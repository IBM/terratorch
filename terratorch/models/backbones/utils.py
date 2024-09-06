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


