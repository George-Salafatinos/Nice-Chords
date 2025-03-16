from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import threading
import time
import math
import os
import io
import pygame.mixer

# Initialize the Flask app
app = Flask(__name__)

# Initialize pygame mixer
pygame.mixer.init(frequency=44100, size=-16, channels=2)

# Function to convert MIDI note to frequency
def midi_to_freq(note):
    return 440 * (2 ** ((note - 69) / 12.0))

# Simple MIDI note to note name converter
def note_to_name(note):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = note // 12 - 1
    note_name = note_names[note % 12]
    return f"{note_name}{octave}"

# Create a sine wave sound for a specific frequency
def create_sine_wave(freq, duration=1.0):
    sample_rate = 44100
    amplitude = 4096
    num_samples = int(duration * sample_rate)
    
    buf = np.zeros((num_samples, 2), dtype=np.int16)
    t = np.arange(num_samples) / sample_rate
    
    # Apply a slight fade in/out to reduce clicks
    fade_samples = int(0.05 * sample_rate)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    # Create the main waveform
    wave = np.sin(2 * np.pi * freq * t) * amplitude
    
    # Apply fades
    wave[:fade_samples] *= fade_in
    wave[-fade_samples:] *= fade_out
    
    # Fill both channels
    buf[:, 0] = wave
    buf[:, 1] = wave
    
    return pygame.mixer.Sound(buffer=buf.tobytes())

# Dictionary to cache generated sounds
sound_cache = {}

# PPO Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, num_notes=4):
        super(PolicyNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_notes = num_notes
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output 4 notes (each 0-127 MIDI value)
        self.note_heads = nn.ModuleList([nn.Linear(hidden_dim, 128) for _ in range(num_notes)])
        
        # Value head for PPO
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, exploration_rate=0.2):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Get logits for each note
        note_logits = [head(x) for head in self.note_heads]
        
        # Apply exploration noise to logits
        if exploration_rate > 0:
            note_logits = [logits + torch.randn_like(logits) * exploration_rate * 0.5 
                          for logits in note_logits]
        
        # Get probabilities for each note
        note_probs = [F.softmax(logits, dim=1) for logits in note_logits]
        
        # Value estimate
        value = self.value_head(x)
        
        return note_logits, note_probs, value
    
    def get_chord(self, x, exploration_rate=0.2):
        """Generate a chord from the policy network"""
        with torch.no_grad():
            _, note_probs, _ = self.forward(x, exploration_rate)
            
            # Sample from each distribution
            note_dists = [Categorical(probs) for probs in note_probs]
            notes = [dist.sample().item() for dist in note_dists]
            
            # Constrain notes to middle registers for more pleasant sound
            # MIDI notes 48-84 = C3 to C6 (middle register of piano)
            notes = [n % 36 + 48 for n in notes]  # Map to reasonable octaves
            
            # Log probabilities of chosen notes
            log_probs = [dist.log_prob(torch.tensor(note % 128)) for dist, note in zip(note_dists, notes)]
            
        return notes, log_probs

# PPO Optimizer
class PPOOptimizer:
    def __init__(self, policy_network, lr=0.01, gamma=0.99, eps_clip=0.1, value_coef=0.5):
        self.policy = policy_network
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        
        # Start with higher learning rate for faster initial learning
        self.initial_lr = lr
        self.min_lr = lr / 5
        
        # Memory for optimization
        self.states = []
        self.notes = []
        self.log_probs = []
        self.ratings = []
        self.values = []
        
        # Tracking statistics
        self.best_model_state = None
        self.best_avg_rating = 0
        self.recent_ratings = []
        
    def store_transition(self, state, notes, log_probs, rating, value):
        self.states.append(state)
        self.notes.append(notes)
        self.log_probs.append(log_probs)
        self.ratings.append(rating)
        self.values.append(value)
        
        # Keep track of recent ratings
        self.recent_ratings.append(rating)
        if len(self.recent_ratings) > 10:
            self.recent_ratings.pop(0)
    
    def update(self):
        # Need at least one sample to update
        if len(self.states) == 0:
            return 0.0

        # Convert to tensors
        states = torch.cat(self.states, dim=0)
        ratings = torch.tensor(self.ratings).unsqueeze(1)
        old_values = torch.cat(self.values, dim=0)
        
        # Normalize ratings to zero mean, unit variance for more stable learning
        if len(self.recent_ratings) > 3:
            mean_rating = sum(self.recent_ratings) / len(self.recent_ratings)
            std_rating = max(0.1, np.std(self.recent_ratings))
            normalized_ratings = (ratings - mean_rating) / std_rating
        else:
            normalized_ratings = ratings - 3.0  # Simple normalization around middle rating
        
        # Compute advantages
        advantages = normalized_ratings - old_values
        
        # Current average rating
        current_avg_rating = sum(self.recent_ratings) / len(self.recent_ratings) if self.recent_ratings else 0
        
        # Adaptive learning rate based on progress
        iterations = len(self.recent_ratings)
        if iterations > 0:
            # Decay learning rate as we progress
            decay_factor = max(0.2, min(1.0, 50 / (iterations + 50)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr * decay_factor
        
        # Only update if we're doing reasonably well or just starting
        if len(self.recent_ratings) < 5 or current_avg_rating > self.best_avg_rating * 0.9:
            # Calculate policy loss
            for _ in range(1):  # Just one update per sample for simplicity
                # Get current probabilities
                new_logits, _, new_values = self.policy(states)
                
                policy_loss = 0
                for i in range(len(self.states)):
                    for j in range(self.policy.num_notes):
                        old_log_prob = self.log_probs[i][j]
                        note = self.notes[i][j]
                        
                        # Ensure note is in valid range for indexing
                        note_idx = note % 128
                        new_log_prob = F.log_softmax(new_logits[j][i], dim=0)[note_idx]
                        
                        ratio = torch.exp(new_log_prob - old_log_prob)
                        surr1 = ratio * advantages[i]
                        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages[i]
                        policy_loss -= torch.min(surr1, surr2)
                
                # Value loss
                value_loss = F.mse_loss(new_values, normalized_ratings)
                
                # Add entropy bonus to encourage exploration
                entropy = 0
                for j in range(self.policy.num_notes):
                    probs = F.softmax(new_logits[j], dim=1)
                    log_probs = F.log_softmax(new_logits[j], dim=1)
                    entropy -= (probs * log_probs).sum(dim=1).mean()
                
                # Total loss with entropy bonus
                loss = policy_loss + self.value_coef * value_loss - 0.01 * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Save as best model if appropriate
            if current_avg_rating > self.best_avg_rating:
                self.best_model_state = {k: v.cpu().clone() for k, v in self.policy.state_dict().items()}
                self.best_avg_rating = current_avg_rating
                print(f"New best model! Avg rating: {current_avg_rating:.2f}")
        else:
            # Revert to best model if performance drops significantly
            if current_avg_rating < self.best_avg_rating * 0.7 and self.best_model_state is not None:
                self.policy.load_state_dict(self.best_model_state)
                print(f"Reverting to best model (current: {current_avg_rating:.2f}, best: {self.best_avg_rating:.2f})")
        
        # Clear memory
        self.states = []
        self.notes = []
        self.log_probs = []
        self.ratings = []
        self.values = []
        
        return current_avg_rating

# Function to play a chord
def play_chord(notes, duration=1.5):
    """Play a chord using pygame.mixer"""
    global sound_cache
    
    # Sort notes for better listening experience
    notes.sort()
    note_names = [note_to_name(note) for note in notes]
    
    # Print what we're playing
    print(f"Playing chord: {notes} ({', '.join(note_names)})")
    
    # Stop any currently playing sounds
    pygame.mixer.stop()
    
    # Load or create sounds for each note
    sounds = []
    for note in notes:
        freq = midi_to_freq(note)
        
        # Check if we've already created this sound
        if freq not in sound_cache:
            sound = create_sine_wave(freq, duration)
            sound_cache[freq] = sound
        else:
            sound = sound_cache[freq]
        
        sounds.append(sound)
    
    # Play all notes simultaneously
    channels = []
    for i, sound in enumerate(sounds):
        # Find an available channel
        channel = pygame.mixer.find_channel()
        if channel:
            channel.play(sound)
            channels.append(channel)
    
    # Wait for playback to complete
    time.sleep(duration)

# Initialize model and optimizer
model = PolicyNetwork()
optimizer = PPOOptimizer(model)

# Global variables to track state
current_chord = None
current_state = None
current_log_probs = None
current_value = None
learning_thread = None
exploration_rate = 0.3  # Default exploration rate

def learning_task():
    global current_chord, current_state, current_log_probs, current_value
    
    # Generate random state
    state = torch.randn(1, model.input_dim)
    current_state = state
    
    # Get chord from policy
    chord, log_probs = model.get_chord(state, exploration_rate)
    
    # Get value estimate
    _, _, value = model.forward(state)
    
    current_chord = chord
    current_log_probs = log_probs
    current_value = value
    
    # Sort for better harmony
    chord.sort()
    
    # Play the chord
    play_chord(chord)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play_chord', methods=['POST'])
def generate_chord():
    global learning_thread, current_chord
    
    # Get the action from the request
    data = request.get_json()
    action = data.get('action', 'new')  # 'new' or 'replay'
    
    if action == 'new' or current_chord is None:
        # Generate a new chord
        if learning_thread is None or not learning_thread.is_alive():
            learning_thread = threading.Thread(target=learning_task)
            learning_thread.daemon = True
            learning_thread.start()
            learning_thread.join()  # Wait for chord to be played
    else:
        # Replay the current chord
        play_chord(current_chord)
    
    return jsonify({'chord': current_chord})

@app.route('/rate_chord', methods=['POST'])
def rate_chord():
    global current_chord, current_state, current_log_probs, current_value, exploration_rate
    
    # Get the rating from the request
    data = request.get_json()
    rating = data.get('rating', 3)
    exploration_rate = data.get('exploration', 0.3)
    
    # Store the transition
    if current_chord is not None and current_state is not None:
        optimizer.store_transition(
            current_state, 
            current_chord, 
            current_log_probs, 
            float(rating), 
            current_value
        )
        
        # Update the model
        avg_rating = optimizer.update()
        
        # Reset for next chord
        current_chord = None
        current_state = None
        current_log_probs = None
        current_value = None
        
        return jsonify({'success': True, 'avg_rating': avg_rating})
    
    return jsonify({'success': False, 'message': 'No chord to rate'})

if __name__ == '__main__':
    app.run(debug=True)