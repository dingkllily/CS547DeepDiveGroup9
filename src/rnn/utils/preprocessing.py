import numpy as np

def one_hot_encoding(categories, data) -> np.ndarray:
    """
    Args:
      categories: [Nc, ]
      data: [N, ]
    Returns:
      result: [N, Nc]
    """
    sorted_categories = sorted(categories)

    category_map = {c: i for i, c in enumerate(sorted_categories)}

    ids = np.array(list(map(lambda c: category_map[c], data)))
   
    result = np.zeros((data.shape[0], len(categories)))

    result[np.arange(ids.size), ids] = 1

    return result