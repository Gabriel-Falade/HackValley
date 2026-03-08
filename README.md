# FacePlay

A hands-free game controller built for hospital patients and anyone who cannot use their hands. A standard webcam and microphone are the only hardware required. No special devices, no wearables, no physical contact.

---

## The Problem

Hospital patients recovering from surgery, stroke, or injury often face weeks or months of limited mobility. They cannot hold a controller or use a keyboard. Gaming is one of the few sources of entertainment available during long stays, yet it remains inaccessible to them. Existing accessibility controllers are expensive, require setup by a caregiver, and are unavailable in most hospital rooms.

FacePlay solves this with software alone. Any laptop with a built-in webcam and microphone already has everything needed.

---

## What It Does

FacePlay reads your face in real time using a webcam and translates facial gestures into game inputs. You look left to move left. You blink to jump or attack. You raise your eyebrows to hold a key. You say the name of a game and it launches. You say "pause" and the game pauses.

Four games are currently supported: Flappy Bird, Super Mario, OG Snake, and Pac-Man. Each game has its own carefully tuned control scheme.

---

## Demo

Run the launcher:

```
python launcher.py
```

Complete the five-step calibration, then say the name of a game to launch it.

---

## How It Works — System Architecture

```
Webcam feed
    |
    +--> MediaPipe FaceLandmarker --> Eye Aspect Ratio (EAR) --> Blink / Wink detection
    |
    +--> MediaPipe FaceLandmarker --> Brow geometry --> Eyebrow raise detection
    |
    +--> ML head model (scikit-learn, trained on landmark data) --> Head direction (LEFT / RIGHT / UP / DOWN)
    |
    +--> pyautogui / ctypes PostMessageW --> Keyboard inputs to game process

Microphone
    |
    +--> SpeechRecognition (Google STT) --> Voice commands --> game selection / pause / resume

Tkinter Launcher (launcher.py)
    |
    +--> CameraThread (daemon) -- webcam preview + head direction for card navigation
    |
    +--> VoiceThread (daemon) -- speech recognition for game selection
    |
    +--> subprocess.Popen --> game process + main.py (--skip-launcher --mode <mode>)
    |
    +--> MonitorThread (daemon) -- watches game process, restores launcher on close
```

The face controller (main.py) and the game process run as two separate subprocesses launched simultaneously. The webcam is released by the launcher before handing off to main.py, so only one process accesses the camera at a time. Keys are sent from main.py to the game using `pyautogui` for most games and `ctypes.PostMessageW` for Flappy Bird (which required direct window message injection to bypass focus restrictions caused by the OpenCV calibration window).

---

## Calibration

Every session begins with a five-step personal calibration that takes roughly 15 seconds. It measures:

1. Neutral eye openness (baseline EAR for both eyes)
2. Full blink threshold
3. Left wink threshold
4. Right wink threshold
5. Eyebrow raise height

These thresholds are specific to the current user's face and lighting, which means FacePlay adapts to anyone — including people with asymmetric eyes, drooping eyelids, or limited expression range.

---

## Installation

Requirements:
- Windows 10 or 11
- Python 3.11
- Webcam
- Microphone
- Internet connection (Google Speech Recognition requires network access)

Setup:

```
python -m venv .venv
.venv\Scripts\activate
pip install opencv-python mediapipe numpy pyautogui pygame pillow SpeechRecognition scikit-learn
python launcher.py
```

---

## Game Selection (Voice)

After calibration, say the name of a game:

| Say                       | Launches     |
|---------------------------|--------------|
| "flappy" / "flappy bird"  | Flappy Bird  |
| "mario" / "super mario"   | Super Mario  |
| "snake" / "og snake"      | OG Snake     |
| "pac man" / "pacman"      | Pac-Man      |

Say "select" or "go" to launch the currently highlighted card. You can also navigate cards by tilting your head left, right, up, or down.

---

## Controls

### Flappy Bird

| Gesture          | Action  |
|------------------|---------|
| Blink            | Flap    |
| Raise eyebrows   | Flap    |

### Super Mario

| Gesture                    | Action                      |
|----------------------------|-----------------------------|
| Tilt head left             | Run left                    |
| Tilt head right            | Run right                   |
| Raise eyebrows             | Jump (hold for taller jump) |
| Tilt up-right              | Jump right simultaneously   |
| Tilt up-left               | Jump left simultaneously    |
| Blink                      | Run boost (Left Shift)      |
| Left wink                  | Pause / unpause             |

### OG Snake

| Gesture               | Action              |
|-----------------------|---------------------|
| Tilt head left        | Turn left           |
| Tilt head right       | Turn right          |
| Tilt head up          | Turn up             |
| Tilt head down        | Turn down           |
| Blink                 | Dash burst          |
| Hold eyebrow raise    | Slow motion         |

### Pac-Man

| Gesture                | Action                      |
|------------------------|-----------------------------|
| Tilt head left         | Move left                   |
| Tilt head right        | Move right                  |
| Tilt head up           | Move up                     |
| Tilt head down         | Move down                   |
| Hold eyebrow raise     | Slow all ghosts             |
| Blink                  | Restart after game over     |
| Hold blink for 3 sec   | Close the game              |

---

## Voice Commands (all games)

| Say      | Action         |
|----------|----------------|
| "pause"  | Pause game     |
| "resume" | Resume game    |

Not available in Flappy Bird (no pause mechanic).

---

## Accessibility Features

- **Personal calibration** — thresholds adapt to any face, any lighting, any expression range
- **Auto-pause** — if the camera loses the face for several seconds, the game pauses automatically
- **Hold-to-exit** — holding a blink for 3 seconds in Pac-Man closes the game cleanly without needing a keyboard
- **Slow mode** — holding eyebrows raised in Pac-Man or Snake slows the game to give more reaction time
- **Voice-only launcher** — no head movement needed to choose a game; just say the name

---

## Hardware Components

| Component   | Purpose                                             |
|-------------|-----------------------------------------------------|
| Webcam      | Captures face for MediaPipe landmark detection      |
| Microphone  | Captures voice for game selection and pause/resume  |

Both are standard built-in components on any modern laptop. No external hardware is required.

---

## Tech Stack

| Technology                         | Role                                              |
|------------------------------------|---------------------------------------------------|
| Python 3.11                        | Primary language                                  |
| MediaPipe FaceLandmarker           | 478-point facial landmark detection               |
| OpenCV                             | Webcam capture, calibration HUD overlay           |
| scikit-learn                       | ML model for head direction classification        |
| pyautogui                          | Keyboard input simulation                         |
| ctypes / Windows User32 API        | Direct window message injection (PostMessageW)    |
| Pygame / SDL                       | Game rendering and input for all four games       |
| Tkinter                            | Game launcher UI                                  |
| Pillow                             | Webcam frames rendered in Tkinter                 |
| SpeechRecognition                  | Voice command detection                           |
| Google Speech Recognition API      | Cloud STT backend                                 |
| Python threading / queue           | Concurrent camera, voice, and monitor threads     |
| subprocess                         | Isolated game and controller processes            |

---

## Project Structure

```
HackValley/
    main.py                     Face controller (calibration + gesture loop)
    launcher.py                 Tkinter game launcher
    head_model.py               Head direction ML model wrapper
    head_model.pkl              Trained model weights
    README.md
    Games/
        Flappy-bird-python/
            flappy.py
            assets/
        super-mario-python/
            main.py
            ...
        og-snake.py
        PacMan/
            pacman.py
            ...
```

---

## Asset Attribution

| Asset                  | Source / License                                          |
|------------------------|-----------------------------------------------------------|
| Flappy Bird assets     | Sourced from open Flappy Bird clone repositories          |
| Super Mario sprites    | super-mario-python open source project (GitHub)           |
| Pac-Man sprites        | Open source Pac-Man Python implementation                 |
| OG Snake               | Custom Python implementation                              |

All game assets used are from open-source educational repositories. FacePlay does not distribute proprietary Nintendo or Namco assets. Sprites are community-recreated resources used under fair educational use.

---

## Troubleshooting

**Camera not found** — Make sure no other application is using the webcam before launching.

**Voice commands not recognised** — Set your microphone as the Windows default recording device. An internet connection is required for Google STT.

**Game does not respond to gestures** — Ensure your face is well lit and centred in the frame. Poor lighting during calibration will produce inaccurate thresholds. Restart and recalibrate.

**Blink detected but no response in game** — Keys are sent directly to the game window via PostMessageW, bypassing focus. If this still occurs, ensure the game window has fully loaded before blinking.
