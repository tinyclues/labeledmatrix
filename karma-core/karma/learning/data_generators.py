import numpy as np

from karma.core.dataframe import DataFrame
from karma.core.utils.utils import use_seed, coerce_to_tuple_and_check_all_strings, slice_batches, flatten
from karma.core.instructions.constant import One


def feedback_by_granularity_generator(df, implicit=False, sync_blocks=None, contrast_granularity=None,
                                      sorted_granularity=True, response=None, negative_positive_ratio=2,
                                      batch_size=10**5, batch_shuffle=True, seed=None):

    with use_seed(seed):
        df_copy = df.copy()

        if contrast_granularity is None:
            contrast_granularity = 'constant'
            df_copy = df_copy.with_instruction_column(contrast_granularity, One((), contrast_granularity
                                                                                  , dtype=np.int8))

        df_copy[contrast_granularity].create_index()
        granularity_index = df_copy[contrast_granularity].index

        if sorted_granularity:
            granularities = sorted(granularity_index.keys())
        else:
            granularities = np.random.permutation(granularity_index.keys())

        for g in granularities:
            indices = granularity_index[g]
            for batch_slice in slice_batches(0, len(indices), size=batch_size):
                batch_indices = indices[batch_slice]
                if batch_shuffle:
                    batch_indices = np.random.permutation(batch_indices)
                if not implicit:
                    yield df_copy[batch_indices]
                else:
                    if not set(flatten(sync_blocks)).issubset(df_copy.column_names):
                        raise ValueError('features in sync_blocks: {} are not a subset of {}'
                                         .format(flatten(sync_blocks), df_copy.column_names))
                    label = np.zeros(len(batch_indices) * (negative_positive_ratio + 1), dtype=np.float32)
                    label[:len(batch_indices)] = 1

                    stable_indices = np.arange(len(batch_indices) * (negative_positive_ratio + 1),
                                               dtype=np.int32) % len(batch_indices)

                    response_df = DataFrame({response or 'response': label})
                    for col in coerce_to_tuple_and_check_all_strings(sync_blocks[0]):
                        response_df[col] = df_copy[col][batch_indices][stable_indices][:]
                    for sync in sync_blocks[1:]:
                        randomized_indices = stable_indices.copy()
                        for j in range(1, negative_positive_ratio + 1):
                            np.random.shuffle(randomized_indices[len(batch_indices) * j: len(batch_indices) * (j + 1)])
                        for col in coerce_to_tuple_and_check_all_strings(sync):
                            response_df[col] = df_copy[col][batch_indices][randomized_indices][:]
                    yield response_df
