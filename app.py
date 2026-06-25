import os
import re
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
import numpy as np
import gradio as gr
from tqdm import tqdm
import mitdeeplearning as mdl

# 1. Reconstruct configurations and vocabulary
# (Note: In production, hardcode your vocab list or save it to a JSON file next to your model)
# For the MIT Lab, we grab the training text just to dynamically rebuild the exact character map.
songs = mdl.lab1.load_training_data()
songs_joined = "\n\n".join(songs)
songs_joined = re.sub(r'(?m)^%.*\n?', '', songs_joined)
vocab = sorted(set(songs_joined))

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def vectorize_string(string):
    return np.array([char2idx[c] for c in string])

def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True,
    )

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None,), batch_size=batch_size),
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        LSTM(rnn_units),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

# 2. Build the generation model (Batch Size = 1)
checkpoint_prefix = "./my_ckpt.weights.h5"
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

model = build_model(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=1)
model.build(tf.TensorShape([1, None]))

# Load weights if they exist in the repository
if os.path.exists(checkpoint_prefix):
    model.load_weights(checkpoint_prefix)
else:
    print(f"Warning: Checkpoint file not found at {checkpoint_prefix}. App running on untrained weights!")

# 3. Core Text Generation function
def generate_text(model, start_string, generation_length=1000, temperature=1.0):
    input_eval = vectorize_string(start_string)
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    
    for layer in model.layers:
        if hasattr(layer, "reset_states"):
            layer.reset_states()
            
    for i in range(generation_length):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
        
    return (start_string + ''.join(text_generated))

# 4. Gradio Interface Logic
MAX_TRACKS = 10
SAMPLE_RATE = 88200

def generate_tracks_ui(num_tracks, duration_seconds, temperature):
    TARGET_SAMPLE_COUNT = SAMPLE_RATE * int(duration_seconds)
    TRACK_CHARACTER_LENGTH = max(1200, int(duration_seconds) * 110)
    num_tracks_int = int(num_tracks)
    current_state = [None] * MAX_TRACKS
    
    rhythms = ["4/4", "3/4", "2/4", "6/8"]
    keys = ["Dmaj", "Gmaj", "Cmaj", "Amin", "Emid"]
    note_starters = ["|:d2A2", "|:A2BA", "|:E2FG", "|:F2A2", "|:B2cd"]
    
    for track_num in range(1, num_tracks_int + 1):
        selected_rhythm = random.choice(rhythms)
        selected_key = random.choice(keys)
        selected_starter = random.choice(note_starters)
        custom_seed = f"X:{track_num}\nT:AI Melody {track_num}\nM:{selected_rhythm}\nK:{selected_key}\n{selected_starter}"
        current_status = f"⏳ AI is actively composing Track {track_num} of {num_tracks_int}... Please hold."
        
        track_text = generate_text(model, start_string=custom_seed, generation_length=TRACK_CHARACTER_LENGTH, temperature=float(temperature))
        waveform = mdl.lab1.play_song(track_text)
        
        if waveform:
            numeric_data = np.frombuffer(waveform.data, dtype=np.int16)
            if len(numeric_data) > TARGET_SAMPLE_COUNT:
                numeric_data = numeric_data[:TARGET_SAMPLE_COUNT]
            elif len(numeric_data) < TARGET_SAMPLE_COUNT:
                padding_needed = TARGET_SAMPLE_COUNT - len(numeric_data)
                numeric_data = np.pad(numeric_data, (0, padding_needed), 'constant')
            current_state[track_num - 1] = (SAMPLE_RATE, numeric_data.astype(np.int16))
        else:
            current_state[track_num - 1] = None
            
        yield current_state + [current_status]

# [Keep your entire locking/unlocking and UI Block setup here exactly as you designed it]
# --- GRADIO INTERFACE BLOCK ---
with gr.Blocks(title="AI Music Orchestrator") as demo:
    gr.Markdown("# 🎛️ Fully Dynamic AI Orchestration Controls")
    # ... Include your remaining Gradio Layout elements here ...
    # (Leaving your UI definitions intact so it populates properly)
    num_tracks_slider = gr.Slider(minimum=1, maximum=MAX_TRACKS, step=1, value=1, label="Number of Songs")
    duration_slider = gr.Slider(minimum=5, maximum=180, step=5, value=10, label="Song Length (Seconds)")
    temp_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.1, label="Variation / Creativity (Temperature)")
    generate_btn = gr.Button("🎬 Start Synthesis Engines", variant="primary")
    clear_btn = gr.Button("🗑️ Clear Audio Tracks", variant="stop")
    status_box = gr.Textbox(value="💡 Ready to compose...", label="System Status", interactive=False)
    
    audio_components = []
    for idx in range(MAX_TRACKS):
        is_visible = True if idx == 0 else False
        player = gr.Audio(label=f"Track {idx + 1}", interactive=True, visible=is_visible)
        audio_components.append(player)
        
    def lock_ui(): return [gr.update(interactive=False)]*5 + [gr.update(interactive=False)]*MAX_TRACKS
    def unlock_ui(): return [gr.update(interactive=True)]*5 + ["✅ Generation complete!"] + [gr.update(interactive=True)]*MAX_TRACKS
    def update_player_visibility(count): return [gr.update(visible=True if idx < int(count) else False, value=None) for idx in range(MAX_TRACKS)]

    num_tracks_slider.change(fn=update_player_visibility, inputs=[num_tracks_slider], outputs=audio_components)
    generate_click = generate_btn.click(fn=lock_ui, outputs=[num_tracks_slider, duration_slider, temp_slider, generate_btn, clear_btn] + audio_components)
    generate_main = generate_click.then(fn=generate_tracks_ui, inputs=[num_tracks_slider, duration_slider, temp_slider], outputs=audio_components + [status_box])
    generate_main.then(fn=unlock_ui, outputs=[num_tracks_slider, duration_slider, temp_slider, generate_btn, clear_btn, status_box] + audio_components)

demo.launch(server_name="0.0.0.0", server_port=7860)