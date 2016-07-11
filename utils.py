from typing import List, Sequence

def flatten(simply_nested_list: List[Sequence]) -> List:

    return [item for sublist in simply_nested_list for item in sublist]