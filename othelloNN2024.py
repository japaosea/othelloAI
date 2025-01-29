import pyautogui
import cv2
import numpy as np
from PIL import ImageGrab
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the board size (8x8 board flattened to 64 elements)
BOARD_SIZE = 64
GRID_SIZE = 8
# Define the bounding box (manually or via pyautogui to select a region)
# For example, bbox = (left, top, right, bottom)

# BOUNDING_BOX = (20, 173, 841, 990)  # for lab PC
BOUNDING_BOX = (77,140,914,976)  # for takadamac



# Define the neural network model
class OthelloNN(nn.Module):
    def __init__(self):
        super(OthelloNN, self).__init__()
        # Input layer to hidden layer (64 to 128 neurons)
        self.fc1 = nn.Linear(BOARD_SIZE * 2, 128)
        # Hidden layer to output layer (128 to 64 neurons)
        self.fc2 = nn.Linear(128, BOARD_SIZE)

    def forward(self, x):
        # Forward pass through the network
        x = torch.relu(self.fc1(x))  # Hidden layer with ReLU activation
        x = torch.softmax(self.fc2(x), dim=-1)  # Output layer with softmax
        return x


# Training function
def update_parameter(model, X_train, y_train, epochs=10, learning_rate=0.001):
    # Loss function (Cross-Entropy for classification)
    criterion = nn.CrossEntropyLoss()
    # Optimizer (Adam optimizer)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        optimizer.zero_grad()  # Reset gradients

        # Forward pass: calculate predictions
        outputs = model(X_train)

        # Compute the loss
        loss = criterion(outputs, y_train)

        # Backward pass: compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # Print the loss every 2 epochs
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.inputs = []
        self.labels = []

        with open(csv_file, "r") as file:
            lines = file.readlines()[1:]  # Get lines starting from the 4th line

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Split by comma to get 2-column data
                parts = line.split(",")
                if len(parts) != 2:
                    continue  # Skip if data is incomplete

                input_str, label_str = parts
                input_str = input_str.strip().replace("[", "").replace("]", "")
                label_str = label_str.strip().replace("[", "").replace("]", "")

                # Convert to list by splitting on spaces
                input_list = [int(x) for x in input_str.split()]
                label_list = [int(x) for x in label_str.split()]

                # Convert to tensor
                input_tensor = torch.tensor(input_list, dtype=torch.float32)
                label_tensor = torch.tensor(label_list, dtype=torch.float32)

                self.inputs.append(input_tensor)
                self.labels.append(label_tensor)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def train_model(model):
    # Create dataset and dataloader
    dataset = CustomDataset("training_data.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Check if data has been loaded correctly
    # print(f"Dataset size: {len(dataset)}\n")

    # Display the first 5 samples
    # for i in range(5):
    # inputs, labels = dataset[i]
    # print(f"Sample {i + 1}:")
    # print(f"Input data: {inputs}")
    # print(f"Label: {labels}\n")

    X_train = []
    y_train = []
    for data, target in dataloader:
        X_train.append(data)
        y_train.append(target)

    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)

    # Example training data (10 random examples for illustration)
    # Random board states (input)
    # X_train = torch.tensor(np.random.randint(0, 3, size=(10, BOARD_SIZE)).astype(np.float32))  # 10 board states
    # Random correct moves (output)
    # y_train = torch.tensor(np.random.randint(0, 64, size=(10,)))  # 10 best move indices (labels)

    # Train the model
    update_parameter(model, X_train, y_train, epochs=1000000, learning_rate=0.001)

    sys.exit()


# Function to get the current mouse position
def get_mouse_position():
    print(
        "Move your mouse to the top-left corner of the bounding box and wait for 5 seconds..."
    )
    time.sleep(5)  # Wait for 5 seconds for the user to move the mouse
    top_left = pyautogui.position()
    print(f"Top-left corner: {top_left}")

    print(
        "Now move your mouse to the bottom-right corner of the bounding box and wait for 5 seconds..."
    )
    time.sleep(5)  # Wait for 5 seconds for the user to move the mouse
    bottom_right = pyautogui.position()
    print(f"Bottom-right corner: {bottom_right}")

    return top_left, bottom_right


# Function to capture a specific region of the screen
def capture_board(bbox):
    # Capture the screen within the bounding box
    screenshot = ImageGrab.grab(bbox)
    # Convert the screenshot to a format that OpenCV can work with
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    return screenshot


# Function to analyze the captured board image and return a numerical vector
def analyze_board(board_image):
    height, width, _ = board_image.shape
    cell_height = height // GRID_SIZE
    cell_width = width // GRID_SIZE

    # Create an empty list to hold the board state (0 for empty, 1 for black, 2 for white)
    black_board_state = []
    white_board_state = []

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            # Extract the cell image from the board
            cell = board_image[
                row * cell_height : (row + 1) * cell_height,
                col * cell_width : (col + 1) * cell_width,
            ]

            # Convert the cell to grayscale for easier color detection
            gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

            # Calculate the average color of the cell (to determine black, white, or empty)
            avg_color = np.mean(gray_cell)

            # Threshold values can be adjusted based on the colors in your Othello game
            if avg_color < 60:  # Black pieces
                black_board_state.append(1)
                white_board_state.append(0)
            elif avg_color > 160:  # White pieces
                black_board_state.append(0)
                white_board_state.append(1)
            else:  # Empty
                black_board_state.append(0)
                white_board_state.append(0)
    board_state = black_board_state + white_board_state  # Black is assumed to be me!!!
    return board_state


# Function to map the board index to screen coordinates
def board_index_to_screen_position(index):
    left, top, right, bottom = BOUNDING_BOX  # Unpack the bounding box coordinates

    # Calculate width and height of the board
    board_width = right - left
    board_height = bottom - top

    # Calculate width and height of each cell (assuming 8x8 grid)
    cell_width = board_width // GRID_SIZE
    cell_height = board_height // GRID_SIZE

    # Determine the row and column from the index (0-63)
    row = index // GRID_SIZE
    col = index % GRID_SIZE

    # Calculate the center of the corresponding cell on the screen
    x_pos = left + col * cell_width + cell_width // 2
    y_pos = top + row * cell_height + cell_height // 2

    return (x_pos, y_pos)


# Function to click on a specific screen position
def click_on_position(position):
    pyautogui.moveTo(
        position[0], position[1]
    )  # Move the mouse to the specified position
    pyautogui.click()  # Simulate a mouse click at the position


def perform_move(predicted_move_index):
    # Convert the predicted move index (0-63) to screen coordinates
    screen_position = board_index_to_screen_position(predicted_move_index)

    # Simulate the mouse click on the calculated screen position
    print(screen_position)  # for debug
    click_on_position(screen_position)
    click_on_position(screen_position)


# Example usage
def main():
    # Get the positions
    top_left, bottom_right = get_mouse_position()
    top_left, bottom_right = get_mouse_position()
    # Print the bounding box coordinates
    print(
        f"Bounding box: (left={top_left.x}, top={top_left.y}, right={bottom_right.x}, bottom={bottom_right.y})"
    )
    sys.exit()

    # Capture the board image
    board_image = capture_board(BOUNDING_BOX)

    # Analyze the board and get the numerical vector
    board_state = analyze_board(board_image)
    board_tensor = torch.tensor(board_state, dtype=torch.float32)

    # Print the board state as an 8x8 grid
    print(np.array(board_state).reshape(2, 8, 8))

    # Optional: Display the captured image using OpenCV (for debugging)
    cv2.imshow("Captured Board", board_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # sys.exit()

    # Create the model
    # model = OthelloNN()

    # Train the model. currently using a random set
    # train_model(model)

    # torch.save(model, 'my_model.pth')
    model = torch.load("my_model.pth")
    # model.load_state_dict(torch.load('model.pth'))

    # Test the model with an example board state (prediction)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for inference
        predicted_move_probs = model(board_tensor)
        print(predicted_move_probs)
        best_move = torch.argmax(predicted_move_probs, dim=-1)

    print(f"Predicted best move index: {best_move.item()}")
    perform_move(best_move.item())


def run_match(pthFile):
    # load model #
    model = torch.load("./pth/" + pthFile)
    model.eval()

    ######################

    while True:
        time.sleep(3)
        # capture board #
        board_image = capture_board(BOUNDING_BOX)

        # Analyze the board and get the numerical vector
        board_state = analyze_board(board_image)
        board_tensor = torch.tensor(board_state, dtype=torch.float32)

        # Print the board state as an 8x8 grid
        # print(np.array(board_state).reshape(2, 8, 8))

        # Optional: Display the captured image using OpenCV (for debugging)
        cv2.imshow("Captured Board", board_image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        ######################

        # select best_move #
        with torch.no_grad():  # Disable gradient computation for inference
            predicted_move_probs = model(board_tensor)
            print(predicted_move_probs)
            best_move = torch.argmax(predicted_move_probs, dim=-1)

        print(f"Predicted best move index: {best_move.item()}")
        perform_move(best_move.item())

        ######################


# Run the training and prediction process
if __name__ == "__main__":
    main()
    # run_match('my_model_5.pth')
    # model = OthelloNN()
    # train_model(model)
