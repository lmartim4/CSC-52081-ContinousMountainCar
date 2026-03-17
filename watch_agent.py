"""Watch a trained PPO agent play Super Mario Bros.

Usage:
  python watch_agent.py                          # symbolic PPO (default)
  python watch_agent.py --model models/pixel_ppo_v2/final_model --pixel
"""

import argparse
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
import pyglet
import numpy as np

from src.wrappers.ram_wrappers import CustomRewardRAM, SkipFrame, RAMGridObservation, FrameStackGrid, FlattenGrid
from src.wrappers.pixel_wrappers import CustomReward, CustomSkipFrame


def make_eval_env(env_id="SuperMarioBros-1-1-v3", pixel=False):
    """Create env for evaluation (with reward shaping, no monitor)."""
    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    if pixel:
        env = CustomReward(env)
        env = CustomSkipFrame(env, skip=4)
    else:
        env = CustomRewardRAM(env)
        env = SkipFrame(env, skip=4)
        env = RAMGridObservation(env)
        env = FrameStackGrid(env, n_stack=4)
        env = FlattenGrid(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/symbolic_ppo/final_model', help='Path to model .zip')
    parser.add_argument('--env', default='SuperMarioBros-1-1-v3', help='Environment ID')
    parser.add_argument('--pixel', action='store_true', help='Use pixel env instead of symbolic')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to watch')
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = make_eval_env(env_id=args.env, pixel=args.pixel)

    # Raw env for rendering (unwrap to get the base NES env)
    raw_env = env.unwrapped

    window = pyglet.window.Window(width=256 * 3, height=240 * 3, caption="Agent Playing Mario")
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

    # State
    state = {
        'obs': None,
        'done': True,
        'episode': 0,
        'steps': 0,
        'total_reward': 0.0,
        'flag': False,
        'image_data': None,
        'show_result_timer': 0,
        'results': [],
    }

    def start_episode():
        reset_result = env.reset()
        state['obs'] = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        state['done'] = False
        state['steps'] = 0
        state['total_reward'] = 0.0
        state['flag'] = False
        state['episode'] += 1
        result_label.text = ""

    def update(dt):
        if state['show_result_timer'] > 0:
            state['show_result_timer'] -= dt
            if state['show_result_timer'] <= 0:
                if state['episode'] < args.episodes:
                    start_episode()
                else:
                    print("\n=== Done ===")
                    for r in state['results']:
                        s = "FLAG!" if r['flag'] else "DEAD"
                        print(f"  Ep {r['ep']}: {r['steps']} steps, reward={r['reward']:.0f} [{s}]")
                    if state['results']:
                        print(f"\nFlag rate: {np.mean([r['flag'] for r in state['results']]):.0%}")
                    pyglet.app.exit()
            return

        if state['done']:
            start_episode()
            return

        # Agent picks action
        action, _ = model.predict(state['obs'], deterministic=True)
        step_result = env.step(int(action))
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        state['obs'] = obs
        state['steps'] += 1
        state['total_reward'] += float(reward)

        if isinstance(info, dict) and info.get('flag_get', False):
            state['flag'] = True

        # Render
        frame = raw_env.render(mode='rgb_array')
        state['image_data'] = frame

        # HUD
        x_pos = info.get('x_pos', 0) if isinstance(info, dict) else 0
        action_name = SIMPLE_MOVEMENT[int(action)]
        label.text = (
            f"Episode: {state['episode']}  Steps: {state['steps']}  "
            f"Reward: {state['total_reward']:.0f}  X: {x_pos}  "
            f"Action: {action_name}"
        )

        if done:
            state['done'] = True
            status = "FLAG!" if state['flag'] else "DEAD"
            result_label.text = f"{status} | Steps: {state['steps']} | Reward: {state['total_reward']:.0f}"
            result_label.color = (50, 255, 50, 255) if state['flag'] else (255, 50, 50, 255)
            state['results'].append({
                'ep': state['episode'],
                'steps': state['steps'],
                'reward': state['total_reward'],
                'flag': state['flag'],
            })
            print(f"Episode {state['episode']}: {state['steps']} steps, reward={state['total_reward']:.0f}, {status}")
            state['show_result_timer'] = 3.0

    @window.event
    def on_draw():
        window.clear()
        if state['image_data'] is not None:
            frame = np.flipud(state['image_data'])
            img = pyglet.image.ImageData(
                frame.shape[1], frame.shape[0], "RGB", frame.tobytes()
            )
            img.blit(0, 0, width=window.width, height=window.height)
        label.draw()
        if result_label.text:
            result_label.draw()

    @window.event
    def on_close():
        env.close()

    pyglet.clock.schedule_interval(update, 1 / 30.0)

    print(f"\nWatching agent: {args.model}")
    print(f"Environment: {args.env}")
    print(f"Mode: {'pixel' if args.pixel else 'symbolic (RAM)'}")
    print(f"Episodes: {args.episodes}\n")

    pyglet.app.run()


if __name__ == "__main__":
    main()
