import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import KFold


def dl_data_load(args):
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

    ids = pd.concat([train["user_id"], sub["user_id"]]).unique()
    isbns = pd.concat([train["isbn"], sub["isbn"]]).unique()

    idx2user = {idx: id for idx, id in enumerate(ids)}
    idx2isbn = {idx: isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id: idx for idx, id in idx2user.items()}
    isbn2idx = {isbn: idx for idx, isbn in idx2isbn.items()}

    train["user_id"] = train["user_id"].map(user2idx)
    sub["user_id"] = sub["user_id"].map(user2idx)
    test["user_id"] = test["user_id"].map(user2idx)

    train["isbn"] = train["isbn"].map(isbn2idx)
    sub["isbn"] = sub["isbn"].map(isbn2idx)
    test["isbn"] = test["isbn"].map(isbn2idx)

    field_dims = np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)

    data = {
        "train": train,
        "test": test.drop(["rating"], axis=1),
        "field_dims": field_dims,
        "users": users,
        "books": books,
        "sub": sub,
        "idx2user": idx2user,
        "idx2isbn": idx2isbn,
        "user2idx": user2idx,
        "isbn2idx": isbn2idx,
    }

    return data


def dl_data_split(args, data):
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

    kf = KFold(n_splits=args.k, random_state=args.seed, shuffle=True)
    for fold_idx, (train_index, valid_index) in enumerate(kf.split(data["train"])):
        train_fold = data["train"].iloc[train_index]
        valid_fold = data["train"].iloc[valid_index]

        X_train_fold, y_train_fold = (
            train_fold.drop(["rating"], axis=1),
            train_fold["rating"],
        )
        X_valid_fold, y_valid_fold = (
            valid_fold.drop(["rating"], axis=1),
            valid_fold["rating"],
        )

        (
            data[f"X_train_fold_{fold_idx}"],
            data[f"X_valid_fold_{fold_idx}"],
            data[f"y_train_fold_{fold_idx}"],
            data[f"y_valid_fold_{fold_idx}"],
        ) = (X_train_fold, X_valid_fold, y_train_fold, y_valid_fold)
    # X_train, X_valid, y_train, y_valid = train_test_split(
    #     data["train"].drop(["rating"], axis=1),
    #     data["train"]["rating"],
    #     test_size=args.test_size,
    #     random_state=args.seed,
    #     shuffle=True,
    # )
    # data["X_train"], data["X_valid"], data["y_train"], data["y_valid"] = (
    #     X_train,
    #     X_valid,
    #     y_train,
    #     y_valid,
    # )

    return data


def dl_data_loader(args, data, fold_idx):
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
        torch.LongTensor(data[f"X_train_fold_{fold_idx}"].values),
        torch.LongTensor(data[f"y_train_fold_{fold_idx}"].values),
    )
    valid_dataset = TensorDataset(
        torch.LongTensor(data[f"X_valid_fold_{fold_idx}"].values),
        torch.LongTensor(data[f"y_valid_fold_{fold_idx}"].values),
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
