"""Play Super Mario Bros manually and see your step count.

Controls:
  Arrow keys = move (right, left)
  Z          = jump (A button)
  X          = run/fire (B button)

  SIMPLE_MOVEMENT actions (7):
    0: NOOP, 1: right, 2: right+A, 3: right+B,
    4: right+A+B, 5: A, 6: left

Keyboard mapping:
  Right       -> action 1 (right)
  Right+Z     -> action 2 (right + jump)
  Right+X     -> action 3 (right + run)
  Right+Z+X   -> action 4 (right + jump + run)
  Z           -> action 5 (jump)
  Left        -> action 6 (left)
  Nothing     -> action 0 (noop)
"""

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import pyglet
from pyglet.window import key
import numpy as np


ENV_ID = "SuperMarioBros-1-2-v3"

env = gym_super_mario_bros.make(ENV_ID)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Track pressed keys
keys_pressed = set()

window = pyglet.window.Window(width=256 * 3, height=240 * 3, caption="Play Mario - Step Counter")
label = pyglet.text.Label(
    "", font_name="Courier", font_size=16,
    x=10, y=window.height - 10, anchor_y="top",
    color=(255, 255, 0, 255),
)
result_label = pyglet.text.Label(
    "", font_name="Courier", font_size=24,
    x=window.width // 2, y=window.height // 2,
    anchor_x="center", anchor_y="center",
    color=(255, 50, 50, 255),
)


@window.event
def on_key_press(symbol, modifiers):
    keys_pressed.add(symbol)

@window.event
def on_key_release(symbol, modifiers):
    keys_pressed.discard(symbol)


def get_action():
    """Map keyboard to SIMPLE_MOVEMENT action."""
    right = key.RIGHT in keys_pressed
    left = key.LEFT in keys_pressed
    jump = key.Z in keys_pressed
    run = key.X in keys_pressed

    if right and jump and run:
        return 4  # right + A + B
    elif right and jump:
        return 2  # right + A
    elif right and run:
        return 3  # right + B
    elif right:
        return 1  # right
    elif jump:
        return 5  # A (jump)
    elif left:
        return 6  # left
    else:
        return 0  # NOOP


# Game state
obs = env.reset()
steps = 0
total_reward = 0.0
done = False
flag_got = False
MAX_STEPS = 4000
episode = 1
episode_results = []
image_data = None
show_result_timer = 0


def update(dt):
    global obs, steps, total_reward, done, flag_got, episode, show_result_timer, image_data

    if show_result_timer > 0:
        show_result_timer -= dt
        if show_result_timer <= 0:
            # Reset for next episode
            obs = env.reset()
            steps = 0
            total_reward = 0.0
            done = False
            flag_got = False
            result_label.text = ""
        return

    if done:
        return

    action = get_action()
    result = env.step(action)
    if len(result) == 5:
        obs, reward, done, truncated, info = result
    else:
        obs, reward, done, info = result

    steps += 1
    total_reward += reward

    if info.get("flag_get", False):
        flag_got = True

    # Render
    frame = env.render(mode="rgb_array")
    image_data = frame

    # Update HUD
    x_pos = info.get("x_pos", 0)
    label.text = (
        f"Episode: {episode}  Steps: {steps}  "
        f"Reward: {total_reward:.0f}  X: {x_pos}  "
        f"Action: {SIMPLE_MOVEMENT[action]}"
    )

    if steps >= MAX_STEPS and not done:
        done = True

    if done:
        status = "FLAG!" if flag_got else ("TIME" if steps >= MAX_STEPS else "DEAD")
        result_text = f"{status} | Steps: {steps} | Reward: {total_reward:.0f}"
        result_label.text = result_text
        result_label.color = (50, 255, 50, 255) if flag_got else (255, 50, 50, 255)

        episode_results.append({
            "episode": episode,
            "steps": steps,
            "reward": total_reward,
            "flag": flag_got,
        })
        print(f"Episode {episode}: {steps} steps, reward={total_reward:.0f}, {status}")
        episode += 1
        show_result_timer = 3.0  # Show result for 3 seconds


@window.event
def on_draw():
    window.clear()
    if image_data is not None:
        # Flip vertically for pyglet
        frame = np.flipud(image_data)
        img = pyglet.image.ImageData(
            frame.shape[1], frame.shape[0], "RGB", frame.tobytes()
        )
        img.blit(0, 0, width=window.width, height=window.height)
    label.draw()
    if result_label.text:
        result_label.draw()


@window.event
def on_close():
    print("\n=== Session Summary ===")
    if episode_results:
        for r in episode_results:
            flag = "FLAG" if r["flag"] else "DEAD"
            print(f"  Ep {r['episode']}: {r['steps']} steps, reward={r['reward']:.0f} [{flag}]")
        steps_list = [r["steps"] for r in episode_results]
        print(f"\nAverage steps: {np.mean(steps_list):.0f}")
        print(f"Min steps: {min(steps_list)}")
        print(f"Max steps: {max(steps_list)}")
    else:
        print("  No episodes completed.")
    env.close()


# Run at ~15 fps (the NES runs at ~60fps but with skip=1 this feels right)
pyglet.clock.schedule_interval(update, 1 / 15.0)

print(f"\n{'='*50}")
print(f"  PLAY MARIO - {ENV_ID}")
print(f"  Controls: Arrow keys + Z (jump) + X (run)")
print(f"  Each step = 1 action (no frame skip)")
print(f"{'='*50}\n")

pyglet.app.run()
