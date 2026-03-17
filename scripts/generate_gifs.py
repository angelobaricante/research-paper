"""Generate GIFs showing trained agents navigating mazes.

Creates side-by-side GIFs for PRNG-trained and QRNG-trained agents
solving both PRNG and QRNG mazes.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import A2C

from config.settings import MODELS_DIR, METRICS_DIR, NUM_TRAIN
from env.maze_env import MazeEnv
from maze.generator import load_mazes


CELL_SIZE = 24
COLORS = {
    "wall": (30, 30, 30),
    "passage": (240, 240, 240),
    "agent": (41, 128, 255),
    "goal": (46, 204, 113),
    "path": (174, 214, 241),
    "start": (255, 165, 0),
}


def render_frame(grid, agent_pos, goal_pos, path_history, title=""):
    """Render a single maze frame as a PIL Image."""
    gs = grid.shape[0]
    img_size = gs * CELL_SIZE
    margin_top = 30 if title else 0
    img = Image.new("RGB", (img_size, img_size + margin_top), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    if title:
        draw.text((img_size // 2 - len(title) * 4, 8), title, fill=(0, 0, 0))

    # Draw maze
    for r in range(gs):
        for c in range(gs):
            x0 = c * CELL_SIZE
            y0 = r * CELL_SIZE + margin_top
            x1 = x0 + CELL_SIZE
            y1 = y0 + CELL_SIZE

            if grid[r, c] == 1:
                color = COLORS["wall"]
            elif (r, c) in path_history:
                color = COLORS["path"]
            else:
                color = COLORS["passage"]

            draw.rectangle([x0, y0, x1, y1], fill=color)

    # Goal
    gr, gc = goal_pos
    draw.rectangle(
        [gc * CELL_SIZE + 2, gr * CELL_SIZE + margin_top + 2,
         (gc + 1) * CELL_SIZE - 2, (gr + 1) * CELL_SIZE + margin_top - 2],
        fill=COLORS["goal"],
    )

    # Start marker
    if path_history:
        sr, sc = list(path_history)[0]
        draw.rectangle(
            [sc * CELL_SIZE + 2, sr * CELL_SIZE + margin_top + 2,
             (sc + 1) * CELL_SIZE - 2, (sr + 1) * CELL_SIZE + margin_top - 2],
            fill=COLORS["start"],
        )

    # Agent
    ar, ac = agent_pos
    padding = 4
    draw.ellipse(
        [ac * CELL_SIZE + padding, ar * CELL_SIZE + margin_top + padding,
         (ac + 1) * CELL_SIZE - padding, (ar + 1) * CELL_SIZE + margin_top - padding],
        fill=COLORS["agent"],
    )

    return img


def record_episode(model, maze, title=""):
    """Run one episode and record frames."""
    env = MazeEnv(mazes=[maze])
    obs, _ = env.reset()
    goal = env._goal

    frames = []
    path = [env.agent_pos]
    path_set = {env.agent_pos}

    # First frame
    frames.append(render_frame(maze, env.agent_pos, goal, path_set, title))

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        path.append(env.agent_pos)
        path_set.add(env.agent_pos)
        frames.append(render_frame(maze, env.agent_pos, goal, path_set, title))
        done = terminated or truncated

    # Hold last frame longer
    for _ in range(15):
        frames.append(frames[-1])

    return frames, info["success"], info["steps"]


def make_gif(frames, path, duration=80):
    """Save frames as GIF."""
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"  Saved: {path} ({len(frames)} frames)")


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    prng_model = A2C.load(str(MODELS_DIR / "a2c_prng" / "final_model"))
    qrng_model = A2C.load(str(MODELS_DIR / "a2c_qrng" / "final_model"))

    # Load test mazes (first 3 from each group)
    prng_mazes = load_mazes("prng")
    qrng_mazes = load_mazes("qrng")
    prng_test = [prng_mazes[i] for i in range(NUM_TRAIN, NUM_TRAIN + 3)]
    qrng_test = [qrng_mazes[i] for i in range(NUM_TRAIN, NUM_TRAIN + 3)]

    # Generate GIFs for each condition
    conditions = [
        ("PRNG-agent on PRNG maze", prng_model, prng_test, "prng_on_prng"),
        ("PRNG-agent on QRNG maze", prng_model, qrng_test, "prng_on_qrng"),
        ("QRNG-agent on QRNG maze", qrng_model, qrng_test, "qrng_on_qrng"),
        ("QRNG-agent on PRNG maze", qrng_model, prng_test, "qrng_on_prng"),
    ]

    for desc, model, mazes, prefix in conditions:
        print(f"\n{desc}:")
        for i, maze in enumerate(mazes):
            title = f"{desc} (maze #{i})"
            frames, success, steps = record_episode(model, maze, title)
            status = "SOLVED" if success else "FAILED"
            gif_path = METRICS_DIR / f"gif_{prefix}_maze{i}.gif"
            make_gif(frames, gif_path)
            print(f"    Maze {i}: {status} in {steps} steps")

    # Also make a comparison GIF: same maze, both agents side by side
    print("\n--- Side-by-side comparison ---")
    for i in range(2):
        maze = prng_test[i]
        frames_prng, s1, st1 = record_episode(prng_model, maze, "PRNG-trained agent")
        frames_qrng, s2, st2 = record_episode(qrng_model, maze, "QRNG-trained agent")

        # Pad shorter sequence
        max_len = max(len(frames_prng), len(frames_qrng))
        while len(frames_prng) < max_len:
            frames_prng.append(frames_prng[-1])
        while len(frames_qrng) < max_len:
            frames_qrng.append(frames_qrng[-1])

        # Combine side by side
        combined = []
        gap = 10
        for fp, fq in zip(frames_prng, frames_qrng):
            w = fp.width + gap + fq.width
            h = max(fp.height, fq.height)
            combo = Image.new("RGB", (w, h), (255, 255, 255))
            combo.paste(fp, (0, 0))
            combo.paste(fq, (fp.width + gap, 0))
            combined.append(combo)

        gif_path = METRICS_DIR / f"gif_comparison_maze{i}.gif"
        make_gif(combined, gif_path, duration=100)
        print(f"  Maze {i}: PRNG={'SOLVED' if s1 else 'FAILED'}({st1}st) vs QRNG={'SOLVED' if s2 else 'FAILED'}({st2}st)")

    print(f"\nAll GIFs saved to {METRICS_DIR}")


if __name__ == "__main__":
    main()
