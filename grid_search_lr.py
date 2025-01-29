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

# DataFrameからデータセットを作成
def make_dataset(df: pd.DataFrame):
    inputs = torch.from_numpy(np.vstack(np.array(df["x_train"]))).float()
    labels = torch.from_numpy(np.vstack(np.array(df["y_train"]))).float()
    return inputs, labels

# 学習関数
def my_train(model, criterion, optName, tol, lr, epoc, modelName):
    # Optimizerの初期化
    optimizer_class = getattr(optim, optName)
    optimizer = optimizer_class(model.parameters(), lr=lr)

    # データの読み込み
    df = pd.read_csv("training_data.csv")
    df["x_train"] = df["x_train"].apply(str_to_ndarray)
    df["y_train"] = df["y_train"].apply(str_to_ndarray)
    inputs, labels = make_dataset(df)

    old_loss = float("inf")
    tolerance_count = 0
    tolerance_threshold = 500  # tol以下の変化が500回連続したら終了

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

        # tol以下の変化が連続した場合のカウント処理
        if abs(old_loss - loss.item()) <= tol:
            tolerance_count += 1
        else:
            tolerance_count = 0

        # tolerance_thresholdに達した場合、早期終了
        if tolerance_count >= tolerance_threshold:
            print(f"Early stopping at epoch {epoch+1}")
            break

        old_loss = loss.item()

    # モデルを保存
    model_path = f"./pth/{modelName}"
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")
    return loss.item(), epoch

# メイン関数
def main():
    directory_path = "./pth/"
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]  # 学習率リスト

    # パラメータ
    param = {
        "model": None,  # モデルは後で動的に設定
        "criterion": nn.CrossEntropyLoss(),
        "optName": "Adam",
        "tol": 1e-5,
        "max_epoc": 100000,
        "modelName": "takada_model",  # モデル名は固定
    }
    memoName = "memo.csv"

    modelName_orig = param["modelName"]
    for lr in learning_rates:
        # 各学習率で新しいモデルをインスタンス化
        param["model"] = Nets.Net(128, 1024, 1024, 64)

        print(f"Training with learning rate: {lr}")
        
        # 既存の記録数をカウントしてモデル名を設定
        with open(directory_path + memoName, "r") as f:
            num = len(f.readlines())
            param["modelName"] = f"{modelName_orig}_{num}.pth"

        # 学習率を変更してトレーニング
        loss, epoch = my_train(
            model=param["model"],
            criterion=param["criterion"],
            optName=param["optName"],
            tol=param["tol"],
            lr=lr,
            epoc=param["max_epoc"],
            modelName=param["modelName"],
        )

        # 結果をメモファイルに記録
        with open(directory_path + memoName, "a") as f:
            print(
                param["modelName"],
                param["optName"],
                param["criterion"],
                lr,
                epoch + 1,
                f"{loss:.8f}",
                param["model"],
                sep=", ",
                file=f,
            )

        print(f"Training for learning rate {lr} completed.")

# 実行
main()
