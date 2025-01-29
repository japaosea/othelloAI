import pandas as pd
import re
import numpy as np
import othello

# 参考：https://qiita.com/TokyoYoshida/items/07bd3cdca6a7e50c3114


def winner(raw: pd.DataFrame):
    blackScore = (
        raw["blackScore"].reset_index().rename(columns={"index": "tournamentId"})
    )
    blackScore["winner"] = np.where(
        blackScore["blackScore"] > 32,
        "B",
        np.where(blackScore["blackScore"] < 32, "W", "D"),
    )
    return blackScore


# 列の値を数字に変換するdictonaryを作る
def left_build_conv_table():
    left_table = ["a", "b", "c", "d", "e", "f", "g", "h"]
    left_conv_table = {}
    n = 1

    for t in left_table:
        left_conv_table[t] = n
        n = n + 1

    return left_conv_table


left_conv_table = left_build_conv_table()


# dictionaryを使って列の値を数字に変換する
def left_convert_colmn_str(col_str):
    return left_conv_table[col_str]


# 1手を数値に変換する
def convert_move(v):
    l = left_convert_colmn_str(v[:1])  # 列の値を変換する
    r = int(v[1:])  # 行の値を変換する
    return np.array([l - 1, r - 1], dtype="int8")


def make_transcript_df(raw: pd.DataFrame):
    # 正規表現を使って2文字ずつ切り出す
    transcripts_raw = raw["transcript"].str.extractall("(..)")

    # Indexを再構成して、1行1手の表にする
    df = transcripts_raw.reset_index().rename(
        columns={"level_0": "tournamentId", "match": "move_no", 0: "move_str"}
    )
    df["move"] = df.apply(lambda x: convert_move(x["move_str"]), axis=1)
    return df


def demo(transcript):  # 1試合のみ対応
    game = othello.Othello()
    for m in transcript["move"].to_list():
        print("--------------------------")
        game.print_board()
        game.show_training_data()

        # この辺にパスの時の処理
        if not game.can_place_stone(m[0], m[1]):
            print("pass")
            game.switch_player()
            game.print_board()

        print("move : ", m)
        result = game.place_stone(m[0], m[1])  # 石を置く

        if not result:
            print("invalid!!")
            break
    game.print_board()


def run_match(transcript):  # 1試合のみ対応
    x_train = []
    y_train = []
    Turn = []
    game = othello.Othello()  # ゲームの初期化
    for m in transcript["move"].to_list():
        if not game.can_place_stone(m[0], m[1]):  # パスのチェック
            game.switch_player()

        x_train.append(
            np.array(game.create_training_data())
        )  # board情報（x_train）とmove情報（y_train）の出力
        y_train.append(np.array(othello.to_onehot(m)))
        Turn.append(game.current_player)
        result = game.place_stone(m[0], m[1])  # 石を置く

        # エラー処理
        if not result:
            raise ValueError("invalid move!!")

    df = pd.DataFrame(
        {
            "tournamentId": transcript["tournamentId"],
            "move_no": transcript["move_no"],
            "x_train": x_train,
            "y_train": y_train,
            "Turn": Turn,
        }
    )
    return df


def make_training_data(transcript, winner):
    training_data = pd.DataFrame()
    for id, ts in transcript.groupby("tournamentId").__iter__():
        one_match_df = run_match(ts)
        training_data = pd.concat([training_data, one_match_df], ignore_index=True)

    # # 勝者の手番のみを取り出す
    training_data = pd.merge(
        training_data, winner, how="right", on="tournamentId"
    ).query("Turn == winner")

    # x_train, y_trainを1次元にする
    training_data["x_train"] = training_data["x_train"].apply(lambda x: x.ravel())
    training_data["y_train"] = training_data["y_train"].apply(lambda x: x.ravel())

    return training_data[["x_train", "y_train"]]


# 2つのcsvファイルを連結し，１つのcsvファイルを新規作成
def joint(csv1, csv2, fileName="training_data"):
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    result = pd.concat([df1, df2], ignore_index=True)
    result.to_csv(f"{fileName}.csv")


def main(year):
    year = str(year)
    # csvの読み込み
    # header: tournamentId,tournamentName,blackPlayerId,blackPlayerName,whitePlayerId,whitePlayerName,blackScore,blackTheoreticalScore,transcript
    raw = pd.read_csv(f"./data/WTH_{year}/wthor_{year}.csv")

    transcript_df = make_transcript_df(raw=raw)
    winner_df = winner(raw=raw)

    td = make_training_data(transcript_df, winner=winner_df)
    # td = make_training_data(transcript_df.query('tournamentId < 10'), winner_df.query('tournamentId < 10'))
    # print(td)
    np.set_printoptions(
        linewidth=10000
    )  # numpy おせっかいブロック（これしないと改行入れられる）
    td.to_csv(f"training_data_{year}.csv", index=False)


# main(2022)
data_path = "./data/"
joint(
    csv1=data_path + "training_data_2023.csv", csv2=data_path + "training_data_2022.csv"
)
