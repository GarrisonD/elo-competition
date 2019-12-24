import pandas as pd


def read_card_owners_csv(file_path):
    return pd.read_csv(file_path, parse_dates=["first_active_month"])


def read_transactions_csv(file_path):
    return pd.read_csv(
        file_path,
        true_values=["Y"],
        false_values=["N"],
        parse_dates=["purchase_date"],
        usecols=[
            "authorized_flag",
            "card_id",
            "installments",
            "month_lag",
            "purchase_amount",
            "purchase_date",
        ],
    )


def reduce_mem_usage(df):
    df = df.copy()

    for column in df.columns:
        if df[column].dtype == "int":
            df[column] = pd.to_numeric(df[column], downcast="integer")

        if df[column].dtype == "float":
            df[column] = pd.to_numeric(df[column], downcast="float")

    return df
