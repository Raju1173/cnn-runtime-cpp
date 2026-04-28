import pygame
import numpy as np
import subprocess
import struct
import os
import time
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(BASE_DIR)

BUILD_DIR = os.path.join(PROJECT_ROOT, "out", "build", "x64-Release")

INFERENCE_EXE = os.path.join(BUILD_DIR, "Inference.exe")
INPUT_FILE = os.path.join(BASE_DIR, "Input.bin")

PREDICT_INTERVAL = 0.25
IMG_SIZE = 28


def center_image(img):
    coords = np.argwhere(img > 0.1)

    if len(coords) == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = img[y0:y1 + 1, x0:x1 + 1]

    h, w = cropped.shape
    size = max(h, w)

    square = np.zeros((size, size), dtype=np.float32)

    y_off = (size - h) // 2
    x_off = (size - w) // 2

    square[y_off:y_off + h, x_off:x_off + w] = cropped

    resized = cv2.resize(square, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    return resized


def preprocess_canvas(canvas: pygame.Surface) -> np.ndarray:
    raw = pygame.surfarray.array3d(canvas)
    raw = np.transpose(raw, (1, 0, 2))

    gray = raw[:, :, 0].astype(np.float32) / 255.0

    gray = 1.0 - gray

    gray = center_image(gray)

    return gray


def write_input_bin(pixels: np.ndarray):
    flat = pixels.flatten().astype(np.float32)

    with open(INPUT_FILE, 'wb') as f:
        f.write(struct.pack(f'{len(flat)}f', *flat))


def run_inference() :
    try:
        result = subprocess.run([INFERENCE_EXE, INPUT_FILE], cwd=BUILD_DIR, capture_output=True, text=True, timeout=1.0)
        name = result.stdout.strip()
        return name if name else '?'
    except FileNotFoundError:
        return 'Inference.exe not found'
    except subprocess.TimeoutExpired:
        return '(timeout)'
    except Exception as e:
        return f'Error: {e}'


for path, desc in [(INFERENCE_EXE, "Compiled Inference.exe"),]:
    if not os.path.exists(path):
        print(f"ERROR: {desc} not found at '{path}'")
        exit(1)


pygame.init()
screen = pygame.display.set_mode((512, 576))
pygame.display.set_caption("Doodle Classifier")
clock = pygame.time.Clock()

canvas = pygame.Surface((512, 512))
canvas.fill("WHITE")

predBG = pygame.Surface((512, 64))
predBG.fill((200, 200, 200))

fontLarge = pygame.font.SysFont('CenturyGothic', 40, bold=True)

running = True
drawing = False
lastPos = None
radius = 3

prediction = "DRAW SOMETHING"
lastPredTime = 0.0

pygame.mouse.set_visible(False)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                drawing = True
                lastPos = event.pos

                y = max(0, event.pos[1] - 64)
                pygame.draw.circle(canvas, "BLACK", (event.pos[0], y), radius)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False

        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                y1 = max(0, lastPos[1] - 64)
                y2 = max(0, event.pos[1] - 64)

                pygame.draw.line(canvas, "BLACK", (lastPos[0], y1), (event.pos[0], y2), radius * 2)

                pygame.draw.circle(canvas, "BLACK", (event.pos[0], y2), radius)

                lastPos = event.pos

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LCTRL] and keys[pygame.K_z]:
        canvas.fill("WHITE")
        prediction = "DRAW SOMETHING"

    now = time.time()
    if now - lastPredTime >= PREDICT_INTERVAL:
        lastPredTime = now

        pixels = preprocess_canvas(canvas)
        write_input_bin(pixels)
        prediction = run_inference()

    screen.fill("BLACK")
    screen.blit(canvas, (0, 64))

    predBG.fill((200, 200, 200))
    predSurf = fontLarge.render(prediction.upper(), True, "BLACK")

    predX = max(4, 256 - predSurf.get_width() // 2)
    predY = 32 - predSurf.get_height() // 2

    predBG.blit(predSurf, (predX, predY))
    screen.blit(predBG, (0, 0))

    mx, my = pygame.mouse.get_pos()
    pygame.draw.circle(screen, "YELLOW", (mx, my), radius)

    pygame.display.flip()
    clock.tick(60)

if os.path.exists(INPUT_FILE):
    os.remove(INPUT_FILE)

pygame.quit()