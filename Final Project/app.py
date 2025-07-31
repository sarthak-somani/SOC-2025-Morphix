# app.py
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import os
import requests
import subprocess
import pickle
import torch
import sys

# --- Constants ---
STYLEGAN_REPO_DIR = "stylegan2-ada-pytorch"
# Add the StyleGAN repo to Python's path to find the custom 'torch_utils' module
sys.path.append(STYLEGAN_REPO_DIR)

MODEL_URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
MODEL_PATH = os.path.join(STYLEGAN_REPO_DIR, "ffhq.pkl")
MORPHIX_REPO_DIR = "SOC-2025-Morphix"

# --- Page Configuration ---
st.set_page_config(
    page_title="Latent Editor UI",
    page_icon="🎨",
    layout="wide",
)

# --- Helper Functions ---
def tensor_to_pil(tensor):
    """Converts a PyTorch tensor to a PIL Image."""
    tensor = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(tensor[0].cpu().numpy(), 'RGB')

def image_to_bytes(img):
    """Converts a PIL Image to bytes for downloading."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def get_w_from_z(z_latent):
    """Maps a Z-space latent vector to the W+ space."""
    G = st.session_state.backend_assets["G"]
    device = st.session_state.backend_assets["device"]
    z_tensor = torch.from_numpy(z_latent).to(device)
    with torch.no_grad():
        w_latent = G.mapping(z_tensor, None)
    return w_latent

def get_image_from_w(w_latent):
    """Generates a PIL image from a W+ space vector."""
    G = st.session_state.backend_assets["G"]
    with torch.no_grad():
        img_tensor = G.synthesis(w_latent, noise_mode='const')
    return tensor_to_pil(img_tensor)

# --- Backend Loading ---
@st.cache_resource
def load_backend():
    """
    Clones repositories, downloads the model, and loads all assets into memory.
    This function runs only once and its return value is cached.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(STYLEGAN_REPO_DIR):
        st.info("Cloning StyleGAN2-ADA repository...")
        subprocess.run(["git", "clone", "https://github.com/NVlabs/stylegan2-ada-pytorch.git", STYLEGAN_REPO_DIR], check=True)
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading StyleGAN model (ffhq.pkl)...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    with open(MODEL_PATH, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    st.success("✅ StyleGAN model loaded successfully!")
    if not os.path.exists(MORPHIX_REPO_DIR):
        st.info("Cloning Morphix repository for latent vectors...")
        subprocess.run(["git", "clone", "https://github.com/sarthak-somani/SOC-2025-Morphix.git", MORPHIX_REPO_DIR], check=True)
    try:
        vectors = {
            'age_w': np.load(os.path.join(MORPHIX_REPO_DIR, 'Models', 'age.npy')),
            'smile_w': np.load(os.path.join(MORPHIX_REPO_DIR, 'Models', 'smile.npy')),
            'gender_w': np.load(os.path.join(MORPHIX_REPO_DIR, 'Models', 'gender.npy')),
            'eyeglasses_w': np.load(os.path.join(MORPHIX_REPO_DIR, 'Models', 'eyeglasses.npy')),
        }
        st.success("✅ Latent vectors loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"Error loading latent vectors: {e}")
        return None
    return {
        "G": G, "device": device,
        **{k: torch.from_numpy(v).to(device) for k, v in vectors.items()}
    }

# --- State Management for Undo/Redo ---
def capture_state():
    """Captures the current editable state, which is now the w_latent vector."""
    return {"w_latent": st.session_state.w_latent.clone()}

def load_state(state):
    """Loads a state dictionary back into the session."""
    st.session_state.w_latent = state["w_latent"]

# --- Image Generation ---
def generate_image_and_update_state():
    """Generates an image from the current w_latent and applies all edits."""
    assets = st.session_state.backend_assets
    w_base = st.session_state.w_latent

    # Apply edits from sliders and presets
    w_edited = w_base + \
               assets['age_w'] * st.session_state.age_strength + \
               assets['smile_w'] * st.session_state.smile_strength + \
               assets['gender_w'] * st.session_state.gender_strength + \
               assets['eyeglasses_w'] * st.session_state.eyeglasses_strength

    st.session_state.image = get_image_from_w(w_edited)

# --- Session State Initialization ---
if 'backend_assets' not in st.session_state:
    with st.spinner("🚀 Starting up... Loading models and assets..."):
        st.session_state.backend_assets = load_backend()

if 'image' not in st.session_state and st.session_state.backend_assets:
    G = st.session_state.backend_assets["G"]
    # Main state variables
    st.session_state.z_latent = np.random.randn(1, G.z_dim)
    st.session_state.w_latent = get_w_from_z(st.session_state.z_latent)
    st.session_state.age_strength = 0.0
    st.session_state.smile_strength = 0.0
    st.session_state.gender_strength = 0.0
    st.session_state.eyeglasses_strength = 0.0

    # Style Mixing state variables
    st.session_state.source_a_z = np.random.randn(1, G.z_dim)
    st.session_state.source_b_z = np.random.randn(1, G.z_dim)
    st.session_state.source_a_img = get_image_from_w(get_w_from_z(st.session_state.source_a_z))
    st.session_state.source_b_img = get_image_from_w(get_w_from_z(st.session_state.source_b_z))

    # History for undo/redo
    st.session_state.undo_stack = []
    st.session_state.redo_stack = []

    generate_image_and_update_state()

# --- Callbacks ---
def record_undo():
    st.session_state.undo_stack.append(capture_state())
    st.session_state.redo_stack.clear()

def random_face_callback():
    record_undo()
    G = st.session_state.backend_assets["G"]
    st.session_state.z_latent = np.random.randn(1, G.z_dim)
    st.session_state.w_latent = get_w_from_z(st.session_state.z_latent)
    reset_all_callback(record_history=False) # Reset sliders but don't create a second undo state

def reset_all_callback(record_history=True):
    if record_history: record_undo()
    st.session_state.age_strength = 0.0
    st.session_state.smile_strength = 0.0
    st.session_state.gender_strength = 0.0
    st.session_state.eyeglasses_strength = 0.0
    generate_image_and_update_state()

def toggle_eyeglasses_callback():
    record_undo()
    st.session_state.eyeglasses_strength = 3.5 if st.session_state.eyeglasses_strength == 0.0 else 0.0
    generate_image_and_update_state()

def undo_callback():
    if st.session_state.undo_stack:
        st.session_state.redo_stack.append(capture_state())
        load_state(st.session_state.undo_stack.pop())
        generate_image_and_update_state()

def redo_callback():
    if st.session_state.redo_stack:
        st.session_state.undo_stack.append(capture_state())
        load_state(st.session_state.redo_stack.pop())
        generate_image_and_update_state()

def new_source_face_callback(source_key):
    G = st.session_state.backend_assets["G"]
    z = np.random.randn(1, G.z_dim)
    w = get_w_from_z(z)
    img = get_image_from_w(w)
    st.session_state[f'source_{source_key}_z'] = z
    st.session_state[f'source_{source_key}_img'] = img

def apply_style_mix_callback(crossover):
    record_undo()
    w_a = get_w_from_z(st.session_state.source_a_z)
    w_b = get_w_from_z(st.session_state.source_b_z)
    w_mixed = w_a.clone()
    w_mixed[:, crossover:, :] = w_b[:, crossover:, :]
    st.session_state.w_latent = w_mixed
    reset_all_callback(record_history=False) # Reset sliders to show the pure mix

# --- UI Layout ---
st.title("🎨 Real-Time Latent Editing Interface")

if not st.session_state.backend_assets or 'image' not in st.session_state:
    st.error("Application failed to initialize. Please check logs or restart.")
else:
    with st.sidebar:
        st.header("Main Controls")

        # Undo/Redo
        undo_disabled = not st.session_state.undo_stack
        redo_disabled = not st.session_state.redo_stack
        col1, col2 = st.columns(2)
        with col1: st.button("Undo", on_click=undo_callback, disabled=undo_disabled, use_container_width=True)
        with col2: st.button("Redo", on_click=redo_callback, disabled=redo_disabled, use_container_width=True)

        st.button("New Random Face", on_click=random_face_callback, use_container_width=True)
        st.button("Reset All Edits", on_click=reset_all_callback, use_container_width=True)

        st.markdown("---")
        st.header("Attribute Sliders")
        st.slider("Age", -5.0, 5.0, key="age_strength", on_change=generate_image_and_update_state)
        st.slider("Smile", -5.0, 5.0, key="smile_strength", on_change=generate_image_and_update_state)
        st.slider("Gender", -5.0, 5.0, key="gender_strength", on_change=generate_image_and_update_state)

        st.markdown("---")
        st.header("Presets")
        btn_text = "Remove Eyeglasses" if st.session_state.eyeglasses_strength != 0.0 else "Add Eyeglasses"
        st.button(btn_text, on_click=toggle_eyeglasses_callback, use_container_width=True)

        st.markdown("---")
        st.header("Style Mixing")
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.source_a_img, caption="Source A")
            st.button("New Face A", on_click=new_source_face_callback, args=('a',), use_container_width=True)
        with col2:
            st.image(st.session_state.source_b_img, caption="Source B")
            st.button("New Face B", on_click=new_source_face_callback, args=('b',), use_container_width=True)

        crossover = st.slider("Mixing Crossover Point", 0, 18, 8, help="0-3: Coarse styles (pose, shape). 4-7: Middle styles (facial features). 8-18: Fine styles (color, texture).")
        st.button("Apply Style Mix", on_click=apply_style_mix_callback, args=(crossover,), use_container_width=True)

        st.markdown("---")
        st.download_button(label="💾 Save Image", data=image_to_bytes(st.session_state.image), file_name="generated_face.png", mime="image/png", use_container_width=True)

    st.image(st.session_state.image, caption="Generated by the Backend", use_column_width=True)
