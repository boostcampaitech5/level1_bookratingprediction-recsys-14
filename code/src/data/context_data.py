import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset


def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6


def random_imputation(df: pd.DataFrame, col: str = "age") -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        imputation 할 data
    col : str
        df 중 random imputation 할 column
    """

    imputed_df = df[col].dropna().sample(df[col].isnull().sum())
    imputed_df.index = df[lambda x: x[col].isnull()].index
    df.loc[df[col].isnull(), col] = imputed_df

    return df


def get_language_data(books):
    isbn_dict = {
        "0": "en",
        "1": "en",
        "2": "fr",
        "3": "de",
        "4": "ja",
        "5": "en",
        "6": "en",
        "7": "zh-CN",
        "8": "es",
        "9": "es",
        "B": "en",
    }

    na_book_list = books[books["language"].isna()]
    guessed_book_list = na_book_list["isbn"].apply(lambda x: x[0]).map(isbn_dict)
    book_list = books["language"].combine(
        guessed_book_list, lambda x, y: y if type(x) == float else x
    )

    return book_list


def gaussian_imputation(
    series: pd.Series, lower: float = 0.0, upper: float = 100.0
) -> pd.Series:
    """Imputate missing values by sampling from gaussain distribution.

    Arg:
        series (pd.Series): Target series that contains missing values.
        lower (float): Lower bound of smpaling values.
        upper (float): Upper bound of sampling values.

    Return:
        imputed_series (pd.Series): Imputed result of target series.
    """
    imputed_series = series.copy()
    mu, sigma = series.mean(), series.std()
    trunc_gaussian = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
    )
    imputed_series[imputed_series.isna()] = (
        trunc_gaussian.rvs(series.isna().sum()).astype(int).astype(float)
    )

    return imputed_series


def get_country_data(users, fill_na: str = "unitedstatesofamerica"):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    fill_na : str
        users 중 country를 모르는(Nan) 값을 채울 문자열
    """

    trans_dict = {
        "alachua": "unitedstatesofamerica",
        "alderney": "unitedkingdom",
        "america": "unitedstatesofamerica",
        "aroostook": "unitedstatesofamerica",
        "bergued": "spain",
        "bermuda": "unitedkingdom",
        "c": "na",
        "ca": "belize",
        "camden": "unitedstatesofamerica",
        "cananda": "canada",
        "catalonia": "spain",
        "catalunya": "spain",
        "catalunyaspain": "spain",
        "caymanislands": "unitedkingdom",
        "channelislands": "unitedkingdom",
        "cherokee": "unitedstatesofamerica",
        "csa": "unitedstatesofamerica",
        "deutschland": "germany",
        "disgruntledstatesofamerica": "unitedstatesofamerica",
        "espaa": "spain",
        "euskalherria": "spain",
        "everywhereandanywhere": "na",
        "faraway": "na",
        "ferrara": "italy",
        "fortbend": "unitedstatesofamerica",
        "framingham": "unitedstatesofamerica",
        "galiza": "spain",
        "guam": "unitedstatesofamerica",
        "guernsey": "unitedkingdom",
        "hereandthere": "na",
        "italia": "italy",
        "k1c7b1": "canada",
        "kern": "unitedstatesofamerica",
        "kuwait": "unitedkingdom",
        "labelgique": "belgium",
        "lachineternelle": "na",
        "lafrance": "france",
        "lasuisse": "switzerland",
        "litalia": "italy",
        "lkjlj": "canada",
        "lleida": "spain",
        "losestadosunidosdenorteamerica": "unitedstatesofamerica",
        "maracopa": "unitedstatesofamerica",
        "maricopa": "unitedstatesofamerica",
        "morgan": "unitedstatesofamerica",
        "naontheroad": "na",
        "nz": "newzealand",
        "orangeco": "unitedstatesofamerica",
        "orense": "spain",
        "pender": "unitedstatesofamerica",
        "petrolwarnation": "unitedstatesofamerica",
        "phillipines": "philipines",
        "polk": "unitedstatesofamerica",
        "puertorico": "unitedstatesofamerica",
        "quit": "na",
        "republicofpanama": "panama",
        "richmondcountry": "unitedstatesofamerica",
        "rutherford": "unitedstatesofamerica",
        "saintloius": "unitedstatesofamerica",
        "shelby": "unitedstatesofamerica",
        "space": "na",
        "sthelena": "unitedkingdom",
        "stthomasi": "unitedstatesofamerica",
        "tdzimi": "na",
        "theworldtomorrow": "na",
        "tobago": "trinidadandtobago",
        "ua": "unitedstatesofamerica",
        "uae": "unitedarabemirates",
        "uk": "unitedkingdom",
        "unitedsates": "unitedstatesofamerica",
        "unitedstaes": "unitedstatesofamerica",
        "unitedstate": "unitedstatesofamerica",
        "unitedstates": "unitedstatesofamerica",
        "universe": "na",
        "unknown": "na",
        "urugua": "uruguay",
        "us": "unitedstatesofamerica",
        "usa": "unitedstatesofamerica",
        "usacanada": "unitedstatesofamerica",
        "usacurrentlylivinginengland": "unitedstatesofamerica",
        "usofa": "unitedstatesofamerica",
        "vanwert": "unitedstatesofamerica",
        "worcester": "england",
        "ysa": "unitedstatesofamerica",
    }

    def get_unchanged(first_str, second_str):
        if type(second_str) == float:
            return first_str
        return second_str

    temp_list = users["location"].apply(lambda x: x.split(sep=",")[-1])
    country_list = temp_list.str.replace("[^0-9a-zA-Z]", "", regex=True)

    country_list = country_list.combine(
        country_list.map(trans_dict, na_action="ignore"), get_unchanged
    )

    country_list = country_list.apply(lambda x: fill_na if x == "" or x == "na" else x)

    return country_list


def get_publisher_data(books):
    """
    Parameters
    ----------
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    """

    books_df = books.copy(deep=True)

    print(books_df["publisher"].nunique())

    publisher_dict = (books_df["publisher"].value_counts()).to_dict()
    publisher_count_df = pd.DataFrame(
        list(publisher_dict.items()), columns=["publisher", "count"]
    )

    publisher_count_df = publisher_count_df.sort_values(by=["count"], ascending=False)

    modify_list = publisher_count_df[publisher_count_df["count"] > 1].publisher.values

    for publisher in modify_list:
        try:
            number = (
                books_df[books_df["publisher"] == publisher]["isbn"]
                .apply(lambda x: x[:4])
                .value_counts()
                .index[0]
            )
            right_publisher = (
                books_df[books_df["isbn"].apply(lambda x: x[:4]) == number]["publisher"]
                .value_counts()
                .index[0]
            )
            books_df.loc[
                books_df[books_df["isbn"].apply(lambda x: x[:4]) == number].index,
                "publisher",
            ] = right_publisher
        except:
            pass

    publisher_list = books_df["publisher"]
    print(books_df["publisher"].nunique())

    return publisher_list


def get_age_data(users, books, ratings):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings : pd.DataFrame
        train_ratings.csv를 인덱싱한 데이터
    """

    users_df = users.copy(deep=True)

    merge_temp = ratings.merge(books, how="left", on="isbn")
    merge_data = merge_temp.merge(users, how="inner", on="user_id")

    null_age_users = users_df[users_df["age"].isna()]

    modify_age_by_isbn = merge_data.groupby("isbn")["age"].agg(["mean", "count"])
    modify_age_by_isbn.sort_values(by="count", ascending=False)

    for index in null_age_users.index:
        count = 0
        sum = 0
        u_id = users_df.loc[index, "user_id"]
        isbn = merge_data[merge_data["user_id"] == u_id]["isbn"].values

        for i in isbn:
            if modify_age_by_isbn.loc[i, "mean"] == np.nan:
                count += 1
                sum += modify_age_by_isbn.loc[i, "mean"]

        if count != 0:
            users_df.loc[index, "age"] = round(sum / count)

    null_age_users = users_df[users_df["age"].isna()]

    modify_age_by_location = users_df.groupby("location_country")["age"].mean()
    users_df.loc[users_df["age"].isna(), "age"] = null_age_users[
        "location_country"
    ].map(modify_age_by_location)

    users_df.loc[users_df["age"] != users_df["age"], "age"] = users_df["age"].mean()

    age_list = users_df["age"]

    return age_list


def process_context_data(users, books, ratings1, ratings2, sub):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    ----------
    """

    ######################## Process Data

    ############# Users

    users["location_city"] = users["location"].apply(lambda x: x.split(",")[0])
    users["location_state"] = users["location"].apply(lambda x: x.split(",")[1])
    # users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users["location_country"] = get_country_data(users)
    users = users.drop(["location"], axis=1)

    users["age"] = users["age"].fillna(int(users["age"].mean()))
    users["age"] = users["age"].apply(age_map)

    ############# Books

    books["language"] = get_language_data(books)

    ######################## Merge Data

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    context_df = ratings.merge(users, on="user_id", how="left").merge(
        books[["isbn", "category", "publisher", "language", "book_author"]],
        on="isbn",
        how="left",
    )

    train_df = ratings1.merge(users, on="user_id", how="left").merge(
        books[["isbn", "category", "publisher", "language", "book_author"]],
        on="isbn",
        how="left",
    )

    test_df = ratings2.merge(users, on="user_id", how="left").merge(
        books[["isbn", "category", "publisher", "language", "book_author"]],
        on="isbn",
        how="left",
    )

    ######################## Index Data

    ############# Users

    # user_id
    ids = pd.concat([ratings1["user_id"], sub["user_id"]]).unique()
    idx2user = {idx: id for idx, id in enumerate(ids)}
    user2idx = {id: idx for idx, id in idx2user.items()}
    train_df["user_id"] = train_df["user_id"].map(user2idx)
    test_df["user_id"] = test_df["user_id"].map(user2idx)

    # location
    loc_city2idx = {v: k for k, v in enumerate(context_df["location_city"].unique())}
    train_df["location_city"] = train_df["location_city"].map(loc_city2idx)
    test_df["location_city"] = test_df["location_city"].map(loc_city2idx)

    loc_state2idx = {v: k for k, v in enumerate(context_df["location_state"].unique())}
    train_df["location_state"] = train_df["location_state"].map(loc_state2idx)
    test_df["location_state"] = test_df["location_state"].map(loc_state2idx)

    loc_country2idx = {
        v: k for k, v in enumerate(context_df["location_country"].unique())
    }
    train_df["location_country"] = train_df["location_country"].map(loc_country2idx)
    test_df["location_country"] = test_df["location_country"].map(loc_country2idx)

    ############# Books

    # isbn
    isbns = pd.concat([ratings1["isbn"], sub["isbn"]]).unique()
    idx2isbn = {idx: isbn for idx, isbn in enumerate(isbns)}
    isbn2idx = {isbn: idx for idx, isbn in idx2isbn.items()}
    train_df["isbn"] = train_df["isbn"].map(isbn2idx)
    test_df["isbn"] = test_df["isbn"].map(isbn2idx)

    # category
    category2idx = {v: k for k, v in enumerate(context_df["category"].unique())}
    train_df["category"] = train_df["category"].map(category2idx)
    test_df["category"] = test_df["category"].map(category2idx)

    # publisher
    publisher2idx = {v: k for k, v in enumerate(context_df["publisher"].unique())}
    train_df["publisher"] = train_df["publisher"].map(publisher2idx)
    test_df["publisher"] = test_df["publisher"].map(publisher2idx)

    # language
    language2idx = {v: k for k, v in enumerate(context_df["language"].unique())}
    train_df["language"] = train_df["language"].map(language2idx)
    test_df["language"] = test_df["language"].map(language2idx)

    # author
    author2idx = {v: k for k, v in enumerate(context_df["book_author"].unique())}
    train_df["book_author"] = train_df["book_author"].map(author2idx)
    test_df["book_author"] = test_df["book_author"].map(author2idx)

    idx = {
        "user2idx": user2idx,
        "isbn2idx": isbn2idx,
        "idx2user": idx2user,
        "idx2isbn": idx2isbn,
        "loc_city2idx": loc_city2idx,
        "loc_state2idx": loc_state2idx,
        "loc_country2idx": loc_country2idx,
        "category2idx": category2idx,
        "publisher2idx": publisher2idx,
        "language2idx": language2idx,
        "author2idx": author2idx,
    }

    return idx, train_df, test_df


def context_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
    ----------
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.data_path + "users.csv")
    books = pd.read_csv(args.data_path + "books.csv")
    train = pd.read_csv(args.data_path + "train_ratings.csv")
    test = pd.read_csv(args.data_path + "test_ratings.csv")
    sub = pd.read_csv(args.data_path + "sample_submission.csv")

    idx, context_train, context_test = process_context_data(
        users, books, train, test, sub
    )

    field_dims = np.array(
        [
            len(idx["user2idx"]),
            len(idx["isbn2idx"]),
            6,
            len(idx["loc_city2idx"]),
            len(idx["loc_state2idx"]),
            len(idx["loc_country2idx"]),
            len(idx["category2idx"]),
            len(idx["publisher2idx"]),
            len(idx["language2idx"]),
            len(idx["author2idx"]),
        ],
        dtype=np.uint32,
    )

    data = {
        "train": context_train,
        "test": context_test.drop(["rating"], axis=1),
        "field_dims": field_dims,
        "users": users,
        "books": books,
        "sub": sub,
        "idx2user": idx["idx2user"],
        "idx2isbn": idx["idx2isbn"],
        "user2idx": idx["user2idx"],
        "isbn2idx": idx["isbn2idx"],
    }

    return data


def context_data_split(args, data):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """

    X_train, X_valid, y_train, y_valid = train_test_split(
        data["train"].drop(["rating"], axis=1),
        data["train"]["rating"],
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )
    data["X_train"], data["X_valid"], data["y_train"], data["y_valid"] = (
        X_train,
        X_valid,
        y_train,
        y_valid,
    )
    return data


def context_data_loader(args, data):
    """
    Parameters
    ----------
    Args:
        batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        data_shuffle : bool
            data shuffle 여부
    ----------
    """

    train_dataset = TensorDataset(
        torch.LongTensor(data["X_train"].values),
        torch.LongTensor(data["y_train"].values),
    )
    valid_dataset = TensorDataset(
        torch.LongTensor(data["X_valid"].values),
        torch.LongTensor(data["y_valid"].values),
    )
    test_dataset = TensorDataset(torch.LongTensor(data["test"].values))

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    data["train_dataloader"], data["valid_dataloader"], data["test_dataloader"] = (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
    )

    return data
