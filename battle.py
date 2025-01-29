from othelloNN2024 import capture_board, analyze_board, perform_move, get_mouse_position
import sys
import time
import cv2
import torch

import othello
from Nets import Net, Net2, Net3

# BOUNDING_BOX = (20, 173, 841, 990)  # for lab PC
BOUNDING_BOX = (77,140,914,976)  # for takadamac

def valid(board, move_index):
    game = othello.Othello()
    game.load_board(board)
    row, col = move_index // 8, move_index % 8
    return game.can_place_stone(row, col)


def vs_me(pthFile):
    # load model #
    model = torch.load("./pth/" + pthFile)
    model.eval()

    ######################

    while True:
        print("打ち終わったらボタンを押してください")
        input()
        # capture board #
        board_image = capture_board(BOUNDING_BOX)

        # Analyze the board and get the numerical vector
        board_state = analyze_board(board_image)
        board_tensor = torch.tensor(board_state, dtype=torch.float32)

        # Optional: Display the captured image using OpenCV (for debugging)
        # cv2.imshow('Captured Board', board_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        ######################

        # select best_move #
        with torch.no_grad():  # Disable gradient computation for inference
            predicted_move_probs = model(board_tensor)
            print(predicted_move_probs)
            best_move = torch.argmax(predicted_move_probs, dim=-1)
            # validation of best_move
            while True:
                if not valid(board_state, best_move):  # best_move is inappropriate
                    ##########
                    # if best_move is inappropriate,
                    # select the next largest value's index
                    ##########
                    predicted_move_probs[best_move] = -100
                    best_move = torch.argmax(predicted_move_probs, dim=-1)
                else:
                    break

        print(f"Predicted best move index: {best_move.item()}")
        perform_move(best_move.item())

        ######################


def run_match(pthFile):
    # load model #
    model = torch.load("./pth/" + pthFile)
    model.eval()

    ######################

    while True:
        time.sleep(5)
        # capture board #
        board_image = capture_board(BOUNDING_BOX)

        # Analyze the board and get the numerical vector
        board_state = analyze_board(board_image)
        board_tensor = torch.tensor(board_state, dtype=torch.float32)

        # Print the board state as an 8x8 grid
        # print(np.array(board_state).reshape(2, 8, 8))

        # Optional: Display the captured image using OpenCV (for debugging)
        # cv2.imshow('Captured Board', board_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        ######################

        # select best_move #
        with torch.no_grad():  # Disable gradient computation for inference
            predicted_move_probs = model(board_tensor)
            best_move = torch.argmax(predicted_move_probs, dim=-1)
            # validation of best_move
            count = 0
            while True:
                if not valid(board_state, best_move):  # best_move is inappropriate
                    ##########
                    # if best_move is inappropriate,
                    # select the next largest value's index
                    ##########
                    count += 1
                    predicted_move_probs[best_move] = -100
                    best_move = torch.argmax(predicted_move_probs, dim=-1)
                    if count >= 10:
                        print("pass\n---press Enter---")
                        input()
                        break
                else:
                    break

        print(f"Predicted best move index, count: {best_move.item()}, {count}")
        perform_move(best_move.item())

        ######################


# Run the training and prediction process
if __name__ == "__main__":
    run_match("takada_model_15.pth")
    # model = OthelloNN()
    # train_model(model)
