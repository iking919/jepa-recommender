"""Special token IDs shared by every dataset and model.

Item IDs are contiguous integers starting at ``FIRST_ITEM_ID``, so the
vocabulary size of a dataset is ``number_of_items + FIRST_ITEM_ID``.
"""

PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 1
FIRST_ITEM_ID = 2
