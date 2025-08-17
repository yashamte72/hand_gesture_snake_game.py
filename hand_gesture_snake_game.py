import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Snake Game Config
WIDTH, HEIGHT = 640, 480
CELL_SIZE = 20
GRID_W, GRID_H = WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE

snake = [(GRID_W // 2, GRID_H // 2)]
direction = (0, -1)  # start moving up
food = (random.randint(0, GRID_W - 1), random.randint(0, GRID_H - 1))
score = 0

# Capture Webcam
cap = cv2.VideoCapture(0)

last_move_time = time.time()
move_delay = 0.2  # Snake speed (seconds per move)


def new_food():
    return (random.randint(0, GRID_W - 1), random.randint(0, GRID_H - 1))


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2  # center of screen

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Index fingertip = landmark 8
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # Draw fingertip
            cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)

            # Decide direction based on finger position
            if x < cx - 100:
                direction = (-1, 0)  # Left
            elif x > cx + 100:
                direction = (1, 0)  # Right
            elif y < cy - 100:
                direction = (0, -1)  # Up
            elif y > cy + 100:
                direction = (0, 1)  # Down

    # Snake moves every move_delay seconds
    if time.time() - last_move_time > move_delay:
        last_move_time = time.time()
        new_head = (snake[0][0] + direction[0], snake[0][1] + direction[1])

        # Wrap-around (screen edges)
        new_head = (new_head[0] % GRID_W, new_head[1] % GRID_H)

        # Check collision with itself
        if new_head in snake:
            snake = [(GRID_W // 2, GRID_H // 2)]
            direction = (0, -1)
            food = new_food()
            score = 0
        else:
            snake.insert(0, new_head)
            if new_head == food:
                score += 1
                food = new_food()
            else:
                snake.pop()

    # Draw Game
    game_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Draw snake (as circles for smooth snake-like body)
    for (x, y) in snake:
        center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
        cv2.circle(game_frame, center, CELL_SIZE // 2, (0, 255, 0), -1)

    # Draw food (as red circle like fruit)
    fx, fy = food
    center_f = (fx * CELL_SIZE + CELL_SIZE // 2, fy * CELL_SIZE + CELL_SIZE // 2)
    cv2.circle(game_frame, center_f, CELL_SIZE // 2, (0, 0, 255), -1)

    # Draw score
    cv2.putText(game_frame, f"Score: {score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show windows
    cv2.imshow("Snake Game", game_frame)
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
