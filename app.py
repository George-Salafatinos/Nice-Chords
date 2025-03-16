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

# PPO Policy Network with Improved Representation
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, num_notes=4):
        super(PolicyNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_notes = num_notes
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Base note output (0-127 MIDI value)
        self.base_note_head = nn.Linear(hidden_dim, 128)
        
        # Interval outputs (0-48 semitones above base note)
        # Using 48 as max interval (4 octaves) for reasonable range
        self.interval_heads = nn.ModuleList([nn.Linear(hidden_dim, 49) for _ in range(num_notes-1)])
        
        # Value head for PPO
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, exploration_rate=0.2):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Get logits for base note
        base_note_logits = self.base_note_head(x)
        
        # Apply exploration noise
        if exploration_rate > 0:
            base_note_logits = base_note_logits + torch.randn_like(base_note_logits) * exploration_rate * 0.5
        
        # Get logits for intervals
        interval_logits = []
        for head in self.interval_heads:
            logits = head(x)
            if exploration_rate > 0:
                logits = logits + torch.randn_like(logits) * exploration_rate * 0.5
            interval_logits.append(logits)
        
        # Get probabilities
        base_note_probs = F.softmax(base_note_logits, dim=1)
        interval_probs = [F.softmax(logits, dim=1) for logits in interval_logits]
        
        # Value estimate
        value = self.value_head(x)
        
        return [base_note_logits] + interval_logits, [base_note_probs] + interval_probs, value
    
    def get_chord(self, x, exploration_rate=0.2):
        """Generate a chord from the policy network"""
        with torch.no_grad():
            _, probs, _ = self.forward(x, exploration_rate)
            
            base_note_probs = probs[0]
            interval_probs = probs[1:]
            
            # Sample base note and intervals
            base_note_dist = Categorical(base_note_probs)
            interval_dists = [Categorical(p) for p in interval_probs]
            
            base_note = base_note_dist.sample().item()
            intervals = [dist.sample().item() for dist in interval_dists]
            
            # Constrain base note to a reasonable range (C3-C5)
            base_note = base_note % 24 + 48  # Middle two octaves
            
            # Calculate actual notes
            notes = [base_note] + [base_note + interval for interval in intervals]
            
            # Get log probabilities
            log_probs = [base_note_dist.log_prob(torch.tensor(base_note % 128))] + \
                       [dist.log_prob(torch.tensor(interval)) for dist, interval in zip(interval_dists, intervals)]
            
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
        
        # Extended memory for retraining
        self.history_states = []
        self.history_notes = []
        self.history_ratings = []
        
        # Tracking statistics
        self.best_model_state = None
        self.best_avg_rating = 0
        self.recent_ratings = []
        self.last_retrain_count = 0
        
        # Load saved model if it exists
        self.model_path = "chord_model.pt"
        self.history_path = "chord_history.pt"
        self.load_model()
        self.load_history()
        
    def save_model(self):
        """Save the model and optimizer state"""
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_model_state': self.best_model_state,
            'best_avg_rating': self.best_avg_rating,
            'recent_ratings': self.recent_ratings
        }, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def save_history(self):
        """Save training history"""
        torch.save({
            'states': self.history_states,
            'notes': self.history_notes,
            'ratings': self.history_ratings
        }, self.history_path)
        print(f"History saved to {self.history_path}")
    
    def load_model(self):
        """Load the model if it exists"""
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path)
                self.policy.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.best_model_state = checkpoint['best_model_state']
                self.best_avg_rating = checkpoint['best_avg_rating']
                self.recent_ratings = checkpoint['recent_ratings']
                print(f"Model loaded from {self.model_path}")
                print(f"Best average rating: {self.best_avg_rating:.2f}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def load_history(self):
        """Load training history if it exists"""
        try:
            if os.path.exists(self.history_path):
                history = torch.load(self.history_path)
                self.history_states = history['states']
                self.history_notes = history['notes']
                self.history_ratings = history['ratings']
                print(f"History loaded with {len(self.history_ratings)} samples")
        except Exception as e:
            print(f"Error loading history: {str(e)}")
            
    def store_transition(self, state, notes, log_probs, rating, value):
        """Store the transition for optimization and history"""
        # Make deep copies to avoid modifying tensors in-place
        self.states.append(state.clone().detach())
        self.notes.append(notes)
        # Make copies of log_probs which are scalar tensors
        self.log_probs.append([p.clone().detach() for p in log_probs])
        self.ratings.append(rating)
        self.values.append(value.clone().detach())
        
        # Also store in history for retraining
        self.history_states.append(state.clone().detach())
        self.history_notes.append(notes)
        self.history_ratings.append(rating)
        
        # Keep track of recent ratings
        self.recent_ratings.append(rating)
        if len(self.recent_ratings) > 10:
            self.recent_ratings.pop(0)
            
        # Save after every 5 ratings
        if len(self.history_ratings) % 5 == 0:
            self.save_model()
            self.save_history()
    
    def retrain_on_history(self, epochs=1):
        """Retrain the model on all historical data"""
        if len(self.history_states) < 5:
            return  # Not enough data
            
        print(f"Retraining on {len(self.history_ratings)} historical samples...")
        
        try:
            # Create a completely new computation graph with deep copies
            hist_states = torch.cat([s.clone().detach() for s in self.history_states], dim=0)
            hist_ratings = torch.tensor(self.history_ratings, dtype=torch.float).unsqueeze(1).clone().detach()
            
            # Normalize ratings
            mean_rating = sum(self.history_ratings) / len(self.history_ratings)
            std_rating = max(0.1, np.std(self.history_ratings))
            normalized_ratings = ((hist_ratings - mean_rating) / std_rating).clone().detach()
            
            # Training loop
            for epoch in range(epochs):
                # Make sure to zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass for training
                _, _, new_values = self.policy(hist_states)
                
                # Value loss (mean squared error)
                value_loss = F.mse_loss(new_values, normalized_ratings)
                
                # Update
                value_loss.backward()
                self.optimizer.step()
                
            print("Retraining complete.")
        except Exception as e:
            print(f"Error during retraining: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update(self):
        # Need at least one sample to update
        if len(self.states) == 0:
            return 0.0
            
        # Check if we should do periodic retraining (every 20 samples)
        total_samples = len(self.history_ratings)
        if total_samples % 20 == 0 and total_samples > 0 and total_samples != self.last_retrain_count and total_samples > self.last_retrain_count:
            self.last_retrain_count = total_samples
            try:
                self.retrain_on_history()
            except Exception as e:
                print(f"Error during retraining: {str(e)}")
        
        # Convert to tensors - make deep copies to avoid in-place modifications
        states = torch.cat([s.clone() for s in self.states], dim=0)
        ratings = torch.tensor(self.ratings, dtype=torch.float).unsqueeze(1).clone()
        old_values = torch.cat([v.clone() for v in self.values], dim=0)
        
        # Store old policy parameters for PPO ratio calculation
        old_log_probs = [[p.clone() for p in sublist] for sublist in self.log_probs]
        
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
            # Calculate policy loss - create fresh computation graph
            for _ in range(1):  # Just one update per sample for simplicity
                self.optimizer.zero_grad()
                
                # Forward pass with fresh computation graph
                new_logits, _, new_values = self.policy(states)
                
                policy_loss = 0
                for i in range(len(self.states)):
                    for j in range(self.policy.num_notes):
                        old_log_prob = old_log_probs[i][j]
                        note = self.notes[i][j]
                        
                        # Ensure note is in valid range for indexing
                        note_idx = min(note % 128, 127)  # Clamp to valid range
                        if note_idx >= new_logits[j].size(1):
                            note_idx = new_logits[j].size(1) - 1
                        
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
    current_state = state.clone().detach()  # Make a detached copy
    
    # Get chord from policy
    chord, log_probs = model.get_chord(state, exploration_rate)
    
    # Get value estimate
    with torch.no_grad():  # Use no_grad to avoid computation graph issues
        _, _, value = model.forward(state)
    
    current_chord = chord
    current_log_probs = [p.clone().detach() for p in log_probs]  # Make copies
    current_value = value.clone().detach()  # Make a copy
    
    # Sort for better harmony
    chord.sort()
    
    # Play the chord
    play_chord(chord)

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """Get current stats about the model and training"""
    stats = {
        'chords_rated': len(optimizer.history_ratings),
        'best_avg_rating': optimizer.best_avg_rating,
        'recent_ratings': optimizer.recent_ratings
    }
    return jsonify(stats)

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