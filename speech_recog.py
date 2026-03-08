import speech_recognition as sr
import pyautogui

r = sr.Recognizer()

# ── Accuracy settings ────────────────────────────────────────────
r.energy_threshold = 300          # minimum volume to register as speech
r.pause_threshold = 0.5           # seconds of silence before speech is done, lower = faster
r.dynamic_energy_threshold = False # stops auto-adjusting which can cause missed words

# ── Commands ─────────────────────────────────────────────────────
key_phrases = ["pause", "resume", "fight", "interact"]

# ── Calibrate to room noise at startup ───────────────────────────
print("Calibrating to room noise...")
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source, duration=1)
print(f"Calibrated. Energy threshold set to {r.energy_threshold}")
print("Listening for: " + ", ".join(key_phrases))

# ── Listen loop ──────────────────────────────────────────────────
while True:
    with sr.Microphone() as source:
        print("\nSay something!")
        try:
            # timeout=3 - stops waiting after 3 seconds if nothing heard
            # phrase_time_limit=3 - cuts off after 3 seconds of speaking
            audio = r.listen(source, timeout=3, phrase_time_limit=3)
        except sr.WaitTimeoutError:
            print("No speech detected, listening again...")
            continue

    try:
        # call recognize_google once and store result
        text = r.recognize_google(audio).lower()
        print("Heard: " + text)

        # check if any key phrase is in the text
        # using 'in' instead of == catches "pause the game" not just "pause"
        if "pause" in text:
            print("COMMAND: pause")
            # pyautogui.press('escape')  # uncomment when ready to hook into game
        elif "resume" in text:
            print("COMMAND: resume")
            # pyautogui.press('escape')  # uncomment when ready to hook into game
        elif "fight" in text:
            print("COMMAND: fight mode")
            # switch to fight mode
        elif "interact" in text:
            print("COMMAND: interact mode")
            # switch to interact mode
        else:
            print("Not a command, ignoring")

    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not connect to Google: {0}".format(e))