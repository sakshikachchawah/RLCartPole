# app.py
"""
Streamlit app to:
 - Load trained DQN model(s) and simulate CartPole-v1 environment with real-time frames and GIF export.
 - Visualize training history (reward curves and moving average) from saved training_history.npz or uploaded file.
 - List and load checkpoint files from the 'checkpoints' folder.
"""

import os
import time
from io import BytesIO

import gymnasium as gym
import imageio
import numpy as np
import streamlit as st
from PIL import Image

from dqn_agent import DQNAgent

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_DEFAULT = "dqn_model_latest.pth"
HISTORY_DEFAULT = "training_history.npz"


# ---------------------------
# Helper functions
# ---------------------------
def list_checkpoints():
    files = []
    if os.path.exists(CHECKPOINT_DIR):
        for fname in sorted(os.listdir(CHECKPOINT_DIR)):
            if fname.endswith(".pth"):
                files.append(os.path.join(CHECKPOINT_DIR, fname))
    # Also include default models if exist
    for default in [MODEL_DEFAULT, "dqn_model.pth", "dqn_model_best.pth"]:
        if os.path.exists(default):
            files.append(default)
    return files


def load_agent_from_file(model_path: str, state_size: int, action_size: int) -> DQNAgent:
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    agent.load(model_path)
    return agent


def frames_to_gif_bytes(frames, duration=0.03):
    with BytesIO() as buf:
        imageio.mimsave(buf, frames, format='GIF', duration=duration)
        buf.seek(0)
        return buf.read()


def moving_average(x, window=100):
    if len(x) < 1:
        return np.array([])
    window = min(window, len(x))
    return np.convolve(x, np.ones(window) / window, mode="valid")


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="DQN Agent for CartPole-v1", layout="centered")

st.title("DQN Agent for CartPole-v1")

st.markdown(
    """
This app demonstrates:
- Simulation: load a trained DQN (including checkpoint models) and run simulations with frame rendering and GIF export.
- Training Visualization: view reward curves and moving averages from training history (training_history.npz).
"""
)

# Sidebar: choose page and optionally pick checkpoint
page = st.sidebar.selectbox("Page", ["Simulation", "Training Visualization"])

# Prepare environment dimension info
env_temp = gym.make("CartPole-v1")
state_sample, _ = env_temp.reset()
STATE_SIZE = state_sample.shape[0]
ACTION_SIZE = env_temp.action_space.n
env_temp.close()

# Shared session state
if "agent" not in st.session_state:
    st.session_state.agent = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "loaded_model_path" not in st.session_state:
    st.session_state.loaded_model_path = None

# List available checkpoints/models
available_models = list_checkpoints()

if page == "Simulation":
    st.header("Simulation")

    st.sidebar.subheader("Model / Checkpoint")
    selected_model = st.sidebar.selectbox("Choose model file to load", options=["(none)"] + available_models)
    if st.sidebar.button("Load Selected Model"):
        if selected_model == "(none)":
            st.warning("Select a model file first.")
        else:
            try:
                st.session_state.agent = load_agent_from_file(selected_model, STATE_SIZE, ACTION_SIZE)
                st.session_state.model_loaded = True
                st.session_state.loaded_model_path = selected_model
                st.success(f"Loaded model: {selected_model}")
            except Exception as e:
                st.exception(f"Failed to load model: {e}")

    # Quick load default model button
    if st.sidebar.button("Quick Load Default Model"):
        if os.path.exists(MODEL_DEFAULT):
            try:
                st.session_state.agent = load_agent_from_file(MODEL_DEFAULT, STATE_SIZE, ACTION_SIZE)
                st.session_state.model_loaded = True
                st.session_state.loaded_model_path = MODEL_DEFAULT
                st.success(f"Loaded model: {MODEL_DEFAULT}")
            except Exception as e:
                st.exception(f"Failed to load default model: {e}")
        else:
            st.warning(f"Default model '{MODEL_DEFAULT}' not found.")

    st.write(f"Loaded model: {st.session_state.loaded_model_path}" if st.session_state.model_loaded else "No model loaded yet.")

    st.subheader("Simulation Controls")
    num_episodes = st.slider("Number of episodes to run", min_value=1, max_value=20, value=3, step=1)
    max_steps = st.slider("Max steps per episode", min_value=50, max_value=1000, value=500, step=50)
    run_sim = st.button("Run Simulation")

    if run_sim:
        if not st.session_state.model_loaded or st.session_state.agent is None:
            st.warning("Load a trained model first.")
        else:
            # Create environment that returns rgb frames
            env = gym.make("CartPole-v1", render_mode="rgb_array")
            agent = st.session_state.agent

            img_placeholder = st.empty()
            progress_bar = st.progress(0.0)
            metric_cols = st.columns(4)
            ep_metric = metric_cols[0].empty()
            step_metric = metric_cols[1].empty()
            score_metric = metric_cols[2].empty()
            avg_metric = metric_cols[3].empty()

            all_scores = []
            all_frames_for_gif = []

            for ep in range(1, num_episodes + 1):
                obs, _ = env.reset()
                done = False
                ep_score = 0.0
                step = 0
                frames = []

                # Initial render
                try:
                    frame = env.render()
                except Exception:
                    frame = None
                if frame is not None:
                    frames.append(frame)
                    img_placeholder.image(frame, caption=f"Episode {ep} Step {step}", use_column_width=True)

                while not done and step < max_steps:
                    action = agent.select_action(obs, epsilon=0.0)  # greedy
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    obs = next_obs
                    ep_score += reward
                    step += 1

                    try:
                        frame = env.render()
                    except Exception:
                        frame = None
                    if frame is not None:
                        frames.append(frame)
                        img_placeholder.image(frame, caption=f"Episode {ep} Step {step}", use_column_width=True)

                    # Update metrics
                    ep_metric.metric("Current Episode", f"{ep}/{num_episodes}")
                    step_metric.metric("Step", f"{step}")
                    score_metric.metric("Episode Score", f"{ep_score:.1f}")
                    avg_score_so_far = np.mean(all_scores) if len(all_scores) > 0 else 0.0
                    avg_metric.metric("Average Score (completed eps)", f"{avg_score_so_far:.2f}")

                    overall_progress = ((ep - 1) * max_steps + step) / (num_episodes * max_steps)
                    progress_bar.progress(min(1.0, overall_progress))

                    # brief pause to let UI update
                    time.sleep(0.02)

                all_scores.append(ep_score)
                all_frames_for_gif.extend(frames)
                avg_metric.metric("Average Score (completed eps)", f"{np.mean(all_scores):.2f}")
                st.write(f"Episode {ep} finished. Score: {ep_score:.2f}, Steps: {step}")

            env.close()
            progress_bar.progress(1.0)
            st.success("Simulation complete.")

            if len(all_frames_for_gif) > 0:
                try:
                    gif_bytes = frames_to_gif_bytes(all_frames_for_gif, duration=0.03)
                    st.subheader("Simulation GIF")
                    st.image(gif_bytes, format="GIF")
                    st.download_button("Download GIF", data=gif_bytes, file_name="simulation.gif", mime="image/gif")
                except Exception as e:
                    st.error(f"Failed to create/display GIF: {e}")

            st.subheader("Summary")
            st.write(f"Number of episodes run: {num_episodes}")
            st.write(f"Average score: {np.mean(all_scores):.2f}")
            st.write(f"Scores per episode: {['{:.1f}'.format(s) for s in all_scores]}")

elif page == "Training Visualization":
    st.header("Training Visualization")

    st.sidebar.subheader("Training history")
    use_default_history = st.sidebar.button("Load Default History")
    uploaded_history = st.sidebar.file_uploader("Or upload training_history.npz", type=["npz"])

    history_data = None
    if use_default_history:
        if os.path.exists(HISTORY_DEFAULT):
            history_data = np.load(HISTORY_DEFAULT)
            st.success(f"Loaded {HISTORY_DEFAULT}")
        else:
            st.warning(f"Default history file '{HISTORY_DEFAULT}' not found.")
    if uploaded_history is not None:
        try:
            history_data = np.load(uploaded_history)
            st.success("Uploaded training history loaded.")
        except Exception as e:
            st.exception(f"Failed to load uploaded history: {e}")

    if history_data is not None:
        scores = history_data.get("scores", None)
        losses = history_data.get("losses", None)
        if scores is None:
            st.error("History file does not contain 'scores' array.")
        else:
            st.subheader("Rewards per Episode")
            st.line_chart(scores)

            st.subheader("Rewards with Moving Average")
            window = st.slider("Moving average window", min_value=1, max_value=200, value=50)
            ma = moving_average(scores, window=window)
            # Create x-axis alignment for moving average
            if len(ma) > 0:
                x = np.arange(len(ma)) + window  # align with episode numbers
                import pandas as pd
                df = pd.DataFrame({"moving_avg": ma}, index=x)
                st.line_chart(df)
            else:
                st.write("Not enough data for moving average with current window.")

            if losses is not None:
                st.subheader("Per-episode Average Loss")
                st.line_chart(losses)

    st.write("---")
    st.subheader("Available Checkpoints / Models")
    models = available_models
    if len(models) == 0:
        st.write("No checkpoint files found in the 'checkpoints' directory.")
    else:
        for m in models:
            st.write(m)
        sel = st.selectbox("Load a checkpoint to inspect", options=models)
        if st.button("Load selected checkpoint"):
            try:
                agent = load_agent_from_file(sel, STATE_SIZE, ACTION_SIZE)
                st.success(f"Loaded checkpoint: {sel}")
                # quick policy test: run a single greedy episode and show final score
                env = gym.make("CartPole-v1", render_mode="rgb_array")
                obs, _ = env.reset()
                done = False
                score = 0.0
                frames = []
                while not done:
                    action = agent.select_action(obs, epsilon=0.0)
                    obs, r, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    score += r
                    try:
                        frames.append(env.render())
                    except Exception:
                        pass
                env.close()
                st.write(f"Test run score (greedy): {score:.2f}")
                if len(frames) > 0:
                    gif_bytes = frames_to_gif_bytes(frames, duration=0.03)
                    st.image(gif_bytes, format="GIF")
            except Exception as e:
                st.exception(f"Failed to load checkpoint: {e}")