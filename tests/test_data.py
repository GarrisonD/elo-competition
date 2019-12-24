import pandas as pd

from elo_competition.data import reduce_mem_usage


def test_reduce_mem_usage():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1.0, 2.0, 3.0],
            "C": ["test", "test", "test"],
            "D": [1 ** 25, 2 ** 25, 3 ** 25],
        }
    )

    assert df.A.dtype == "int64"
    assert df.B.dtype == "float64"
    assert df.C.dtype == "object"
    assert df.D.dtype == "int64"

    df_optimized = reduce_mem_usage(df)

    assert (df == df_optimized).all(axis=None)

    # doesn't mutate original data frame
    assert df.A.dtype == "int64"
    assert df.B.dtype == "float64"
    assert df.C.dtype == "object"
    assert df.D.dtype == "int64"

    assert df_optimized.A.dtype == "int8"
    assert df_optimized.B.dtype == "float32"
    assert df_optimized.C.dtype == "object"
    assert df_optimized.D.dtype == "int64"
