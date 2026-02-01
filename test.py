import os, time
os.environ["MUJOCO_GL"] = "egl"
os.environ["EGL_PLATFORM"] = "surfaceless"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/nvidia:" + os.environ.get("LD_LIBRARY_PATH", "")

import gymnasium as gym
import metaworld
import mujoco
import imageio.v2 as imageio
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy

seed = 0
save_dir = f"/home/xli990/paichichi/videos/reach_2views/{int(time.time())}"
os.makedirs(save_dir, exist_ok=True)

env = gym.make("Meta-World/MT1", env_name="reach-v3",
               render_mode="None", disable_env_checker=True, seed=seed)

# env = gym.wrappers.RecordVideo(
#     env,
#     video_folder=save_dir,
#     name_prefix="reach_full_episode",
#     episode_trigger=lambda ep: True,
# )


policy = SawyerReachV3Policy()

obs, info = env.reset()

u = env.unwrapped
model, data = u.model, u.data

# 列出相机名字（如果 XML 里定义了 camera）
cam_names = [model.cam(i).name for i in range(model.ncam)]
print("available cameras:", cam_names)

camA = "corner2"
camB = "topview"

H, W = 480, 480
rendererA = mujoco.Renderer(model, height=H, width=W)
rendererB = mujoco.Renderer(model, height=H, width=W)

mp4_a = os.path.join(save_dir, f"view_{camA}.mp4")
mp4_b = os.path.join(save_dir, f"view_{camB}.mp4")
writer_a = imageio.get_writer(mp4_a, fps=30)
writer_b = imageio.get_writer(mp4_b, fps=30)

try:
    for t in range(500):

        rendererA.update_scene(data, camera=camA)
        frame_a = rendererA.render()

        rendererB.update_scene(data, camera=camB)
        frame_b = rendererB.render()

        writer_a.append_data(frame_a)
        writer_b.append_data(frame_b)

        a = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(a)
        last_obs = obs
        last_info = info
        if terminated or truncated or info.get("success", None) > 0:
            break

    hand_pos = last_obs[:3]
    puck_pos = last_obs[4:7]
    goal_pos = last_obs[-3:]

    print("LAST hand_pos:", hand_pos)
    print("LAST puck_pos:", puck_pos)
    print("LAST goal_pos:", goal_pos)
    print("LAST info keys:", last_info.keys())
    print("done, steps:", t+1, "success:", info.get("success", None))
    print(info)
finally:
    try:
        rendererA.close()
    except:
        pass
    try:
        rendererB.close()
    except:
        pass
    writer_a.close()
    writer_b.close()
    env.close()
print("saved:", mp4_a)
print("saved:", mp4_b)
print("folder:", save_dir)