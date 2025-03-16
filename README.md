# Chord Preference Learning App

This Flask application learns your chord preferences in real-time using a modified PPO (Proximal Policy Optimization) reinforcement learning algorithm.

## How It Works

The application:
1. Generates chords based on the current policy
2. Plays them using pygame.mixer (allowing true simultaneous notes)
3. Collects ratings from the user (1-5)
4. Updates the model to generate better chords over time

## Technical Details

- **Algorithm:** Modified PPO (optimized for online, single-sample learning)
- **Model:** Small neural network with 2 hidden layers
- **Features:**
  - User-controlled exploration rate
  - Model safeguards to prevent performance collapse
  - Automatic checkpointing and recovery
- **Audio:** Uses pygame.mixer for true chord playback (simultaneous notes)

## Setup Instructions

1. Install dependencies:
```
pip install requirements.txt
```

2. Run the application:
```
python app.py
```

3. Open a web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Click "Play Chord" to generate and hear a chord
2. Click "Replay Chord" to hear the same chord again
3. Click "New Chord" to generate a different chord
4. Rate the chord from 1-5 (Very Bad to Very Good)
5. Adjust the exploration slider to control how much the model explores versus exploits
6. Continue playing and rating chords, and watch as the model learns your preferences

## Requirements

- Windows (this app is optimized for Windows users)
- Python 3.7+
- An audio output device

## File Structure

- `app.py`: Main application file with Flask routes, ML model, and audio playback
- `templates/index.html`: Frontend UI
- `requirements.txt`: Required packages