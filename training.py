import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import Nets



# "[0 0 0 1 0 0 ..]"のようなstrを1次元のnumpy配列に変換
def str_to_ndarray(s_list):
    return np.array(list(map(int, s_list.strip("[]").split(" "))))


def make_dataset(df: pd.DataFrame):
    inputs = torch.from_numpy(np.vstack(np.array(df["x_train"]))).float()
    # inputs = torch.randn(100, 128)
    labels = torch.from_numpy(np.vstack(np.array(df["y_train"]))).float()
    return inputs, labels


def my_train(model, criterion, optName, tol, lr, epoc, modelName):
    # 例として、入力層が10ノード、隠れ層が5ノード、出力層が2ノードのネットワークを作成

    optimizer_class = getattr(optim, optName)
    optimizer = optimizer_class(model.parameters(), lr=lr)

    df = pd.read_csv("training_data.csv")
    df["x_train"] = df["x_train"].apply(str_to_ndarray)
    df["y_train"] = df["y_train"].apply(str_to_ndarray)
    inputs, labels = make_dataset(df)

    old_loss = float("inf")
    tolerance_count = 0
    tolerance_threshold = 100  # tol以下の変化が100回連続したら終了

    # 学習ループ
    model.train()
    for epoch in range(epoc):
        # 順伝播
        output = model(inputs)
        loss = criterion(output, labels)

        # 逆伝播とパラメータ更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 損失を表示
        print(f"Epoch {epoch+1}, Loss: {loss.item():.8f}")

        # 損失の変化がtol以下の場合、tolerance_countを増加
        if abs(old_loss - loss.item()) <= tol:
            tolerance_count += 1
        else:
            tolerance_count = 0  # tol以下にならない場合はカウントをリセット

        # tolerance_thresholdに達した場合、早期終了
        if tolerance_count >= tolerance_threshold:
            print(f"Early stopping at epoch {epoch+1}")
            break

        old_loss = loss.item()

    # モデルを保存
    torch.save(model, "./pth/" + modelName)
    return loss.item(), epoch


def main():
    directory_path = "./pth/"

    # ここを変更
    param = {
        "model": Nets.Net3(128, 512, 64),
        "criterion": nn.CrossEntropyLoss(),
        "optName": "Adam",
        "tol": 1e-5,
        "lr": 1,
        "max_epoc": 1,
        "modelName": "takada_model",
    }
    memoName = "memo.txt"
    ###########
    with open(directory_path + memoName, "r") as f:
        num = len(f.readlines())
        param["modelName"] += "_" + str(num) + ".pth"

    loss, epoch = my_train(
        model=param["model"],
        criterion=param["criterion"],
        optName=param["optName"],
        tol=param["tol"],
        lr=param["lr"],
        epoc=param["max_epoc"],
        modelName=param["modelName"],
    )

    with open(directory_path + memoName, "a") as f:
        print(
            param["modelName"],
            param["optName"],
            param["criterion"],
            param["lr"],
            epoch + 1,
            f"{loss:.8f}",
            param["model"],
            sep=", ",
            file=f,
        )


main()
