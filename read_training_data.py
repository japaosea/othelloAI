
import othello







# 適当な入力データと正解ラベルを作成
# def make_dataset(df: pd.DataFrame):
#     inputs = torch.from_numpy(np.vstack(np.array(df["x_train"]))).float()
#     # inputs = torch.randn(100, 128)
#     labels = torch.from_numpy(np.vstack(np.array(df["y_train"]))).float()
#     return inputs, labels


def main():
    # 学習データが間違っていないか確認
    df = pd.read_csv("training_data_temp.csv")
    df["x_train"] = df["x_train"].apply(str_to_ndarray)
    for i in range(5):
        board = df["x_train"][i]
        game = othello.Othello()
        game.load_board(board)
        game.print_board()
        print(np.array(board).reshape(2, 8, 8))


def compare_csv(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    # セルごとに比較
    diff = df1 != df2

    # 差分がある行のみを抽出
    differences = df1[diff.any(axis=1)]

    print("Differing rows:")
    print(differences)


# main()
# compare_csv("training_data.csv", "training_data_temp.csv")
