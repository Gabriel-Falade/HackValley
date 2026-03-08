"""
FacePlay — Tkinter Game Launcher
Dark gaming dashboard with webcam preview, voice nav, and head-tilt nav.
"""

import os, sys, math, time, queue, threading, subprocess
import tkinter as tk

# _PYTHON can contain embedded quotes on some Windows installations,
# which corrupts subprocess.Popen argument lists. Strip them.
_PYTHON = sys.executable.strip('"').strip("'")

# Optional imports — graceful fallback if not installed
try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    from PIL import Image, ImageTk
    _PIL = True
except ImportError:
    _PIL = False

try:
    import speech_recognition as sr
    _SR = True
except ImportError:
    _SR = False

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(HERE, "Games")
MAIN_PY   = os.path.join(HERE, "main.py")

# ── Colours ───────────────────────────────────────────────────────────────────
C_BG      = "#0a0a14"
C_CYAN    = "#00ffff"
C_PURPLE  = "#bf00ff"
C_WHITE   = "#e8e8f0"
C_DIM     = "#555566"
C_CARD_BG = "#12121e"

# ── Layout ────────────────────────────────────────────────────────────────────
WIN_W, WIN_H   = 1280, 800
CARD_W, CARD_H = 330,  240
CAM_W,  CAM_H  = 240,  180

# ── Timing ────────────────────────────────────────────────────────────────────
WEBCAM_MS  = 33      # ~30 fps webcam refresh
PULSE_MS   = 45      # border pulse tick
POLL_MS    = 200     # queue poll
NAV_COOL   = 0.65    # seconds between head-nav moves
NAV_STABLE = 5       # frames same direction required before acting
COUNTDOWN  = 3       # countdown seconds before launch

# ── Games ─────────────────────────────────────────────────────────────────────
GAMES = [
    {
        "name":  "Flappy Bird",
        "mode":  "Flappy_Bird",
        "accent": C_CYAN,
        "cmd":   [_PYTHON, os.path.join(GAMES_DIR, "Flappy-bird-python", "flappy.py")],
        "cwd":   os.path.join(GAMES_DIR, "Flappy-bird-python"),
        "voice": ["flappy", "flappy bird", "bird"],
    },
    {
        "name":  "Super Mario",
        "mode":  "Mario",
        "accent": "#ff5050",
        "cmd":   [_PYTHON, os.path.join(GAMES_DIR, "super-mario-python", "main.py")],
        "cwd":   os.path.join(GAMES_DIR, "super-mario-python"),
        "voice": ["mario", "super mario"],
    },
    {
        "name":  "OG Snake",
        "mode":  "Snake",
        "accent": "#50ff80",
        "cmd":   [_PYTHON, os.path.join(GAMES_DIR, "og-snake.py")],
        "cwd":   GAMES_DIR,
        "voice": ["snake", "og snake"],
    },
    {
        "name":  "Pac-Man",
        "mode":  "Pacman",
        "accent": "#ffdd00",
        "cmd":   [_PYTHON, os.path.join(GAMES_DIR, "PacMan", "pacman.py")],
        "cwd":   os.path.join(GAMES_DIR, "PacMan"),
        "voice": ["pac man", "pacman", "pac"],
    },
]


# ── Camera / direction thread ──────────────────────────────────────────────────
class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._lock        = threading.Lock()
        self._frame       = None    # latest BGR frame
        self._direction   = None    # "LEFT"|"RIGHT"|"UP"|"DOWN"|None
        self._stop_event  = threading.Event()
        self._predict     = None

        # Try to import head_model predict function
        try:
            sys.path.insert(0, HERE)
            from head_model import predict as _predict
            self._predict = _predict
        except Exception:
            pass

    def run(self):
        if not _CV2:
            return
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)

            direction = None
            if self._predict is not None:
                try:
                    direction = self._predict(frame)
                except Exception:
                    pass

            with self._lock:
                self._frame     = frame.copy()
                self._direction = direction

        cap.release()

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_direction(self):
        with self._lock:
            return self._direction

    def stop(self):
        self._stop_event.set()


# ── Voice thread ───────────────────────────────────────────────────────────────
class VoiceThread(threading.Thread):
    def __init__(self, cmd_queue: queue.Queue):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self._q = cmd_queue

    def run(self):
        if not _SR:
            return
        r   = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
        while not self._stop_event.is_set():
            try:
                with mic as source:
                    audio = r.listen(source, timeout=2, phrase_time_limit=3)
                text = r.recognize_google(audio).lower().strip()
                self._q.put(("voice", text))
            except Exception:
                pass

    def stop(self):
        self._stop_event.set()


# ── Colour helpers ─────────────────────────────────────────────────────────────
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Launcher UI ────────────────────────────────────────────────────────────────
class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("FacePlay")
        root.geometry(f"{WIN_W}x{WIN_H}")
        root.resizable(False, False)
        root.configure(bg=C_BG)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.selected      = 0
        self.cmd_queue     = queue.Queue()
        self._pulse_phase  = 0.0
        self._nav_dir_buf  = []
        self._nav_last_t   = 0.0
        self._countdown_n  = 0
        self._cam_thread   = None
        self._voice_thread = None
        self._game_proc    = None
        self._main_proc    = None

        self._build_ui()
        self._start_threads()
        self._schedule_loops()

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        # Title
        title_fr = tk.Frame(self.root, bg=C_BG)
        title_fr.pack(fill="x", pady=(28, 0))
        tk.Label(title_fr, text="FacePlay", bg=C_BG, fg=C_CYAN,
                 font=("Segoe UI", 40, "bold")).pack()
        tk.Label(title_fr,
                 text='Say a game name to launch  •  Say "select" / "go" to confirm',
                 bg=C_BG, fg=C_DIM, font=("Segoe UI", 13)).pack(pady=(4, 0))

        # 2×2 card grid
        cards_fr = tk.Frame(self.root, bg=C_BG)
        cards_fr.pack(expand=True)

        self._canvases = []
        for idx, (row, col) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            c = self._build_card(cards_fr, idx)
            c.grid(row=row, column=col, padx=22, pady=18)
            self._canvases.append(c)

        # Bottom bar
        bot_fr = tk.Frame(self.root, bg=C_BG)
        bot_fr.pack(fill="x", side="bottom", pady=8)

        # Webcam preview (bottom-right)
        self._cam_label = tk.Label(bot_fr, bg="#111122",
                                   width=CAM_W, height=CAM_H)
        self._cam_label.pack(side="right", padx=18, pady=4)

        # Voice status (bottom-left)
        self._voice_lbl = tk.Label(bot_fr, text="Mic ready", bg=C_BG,
                                   fg=C_DIM, font=("Segoe UI", 11))
        self._voice_lbl.pack(side="left", padx=18)

        # Full-screen countdown overlay (hidden initially)
        self._overlay = tk.Frame(self.root, bg=C_BG)
        self._cd_lbl  = tk.Label(self._overlay, text="", bg=C_BG,
                                  fg=C_CYAN, font=("Segoe UI", 100, "bold"))
        self._cd_lbl.pack(expand=True)
        self._cd_sub  = tk.Label(self._overlay, text="", bg=C_BG,
                                  fg=C_WHITE, font=("Segoe UI", 22))
        self._cd_sub.pack(pady=(0, 60))

        # Key bindings
        self.root.bind("<Left>",   lambda e: self._move_h(-1))
        self.root.bind("<Right>",  lambda e: self._move_h(+1))
        self.root.bind("<Up>",     lambda e: self._move_v(-1))
        self.root.bind("<Down>",   lambda e: self._move_v(+1))
        self.root.bind("<space>",  lambda e: self._start_countdown())
        self.root.bind("<Return>", lambda e: self._start_countdown())
        self.root.bind("<Escape>", lambda e: self._on_close())

    def _build_card(self, parent, idx):
        g = GAMES[idx]
        c = tk.Canvas(parent, width=CARD_W, height=CARD_H,
                      bg=C_CARD_BG, highlightthickness=0)
        # Border
        c.create_rectangle(3, 3, CARD_W-3, CARD_H-3,
                            outline=g["accent"], width=3, tags="border")
        # Top colour stripe
        c.create_rectangle(3, 3, CARD_W-3, 10,
                            fill=g["accent"], outline="", tags="stripe")
        # Game name
        c.create_text(CARD_W//2, 55, text=g["name"],
                      fill=g["accent"], font=("Segoe UI", 20, "bold"))
        # Mode tag
        c.create_text(CARD_W//2, 90, text=f"Mode: {g['mode']}",
                      fill=C_DIM, font=("Segoe UI", 11))
        # Voice triggers
        voice_hint = "  /  ".join(f'"{v}"' for v in g["voice"][:2])
        c.create_text(CARD_W//2, 140, text=f"Say: {voice_hint}",
                      fill=C_WHITE, font=("Segoe UI", 12))
        # SELECT label — shown only when selected
        c.create_text(CARD_W//2, CARD_H-36, text="[ SELECT ]",
                      fill=g["accent"], font=("Segoe UI", 14, "bold"),
                      tags="launch_hint", state="hidden")
        c.bind("<Button-1>", lambda e, i=idx: self._card_clicked(i))
        return c

    # ── Thread management ──────────────────────────────────────────────────────
    def _start_threads(self):
        self._cam_thread = CameraThread()
        self._cam_thread.start()
        self._voice_thread = VoiceThread(self.cmd_queue)
        self._voice_thread.start()

    def _stop_threads(self):
        if self._cam_thread:
            self._cam_thread.stop()
            self._cam_thread = None
        if self._voice_thread:
            self._voice_thread.stop()
            self._voice_thread = None

    # ── Animation / polling loops ──────────────────────────────────────────────
    def _schedule_loops(self):
        self._pulse_tick()
        self._webcam_tick()
        self._poll_tick()

    def _pulse_tick(self):
        self._pulse_phase += 0.12
        v = int(128 + 127 * math.sin(self._pulse_phase))
        for idx, c in enumerate(self._canvases):
            g = GAMES[idx]
            if idx == self.selected:
                r, gr, b = _hex_to_rgb(g["accent"])
                mixed = _rgb_to_hex(
                    max(r // 2, int(r * v / 255)),
                    max(gr // 2, int(gr * v / 255)),
                    max(b // 2, int(b * v / 255)),
                )
                c.itemconfig("border", outline=mixed, width=4)
                c.itemconfig("launch_hint", state="normal")
                c.configure(bg="#1a1a2e")
            else:
                c.itemconfig("border", outline=C_DIM, width=2)
                c.itemconfig("launch_hint", state="hidden")
                c.configure(bg=C_CARD_BG)
        self.root.after(PULSE_MS, self._pulse_tick)

    def _webcam_tick(self):
        if self._cam_thread and _CV2 and _PIL:
            frame = self._cam_thread.get_frame()
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb).resize((CAM_W, CAM_H))
                ph  = ImageTk.PhotoImage(img)
                self._cam_label.configure(image=ph)
                self._cam_label._photo = ph  # keep reference
        self.root.after(WEBCAM_MS, self._webcam_tick)

    def _poll_tick(self):
        try:
            while True:
                msg = self.cmd_queue.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass
        self._handle_head_nav()
        self.root.after(POLL_MS, self._poll_tick)

    # ── Navigation ─────────────────────────────────────────────────────────────
    def _handle_head_nav(self):
        if not self._cam_thread:
            return
        d = self._cam_thread.get_direction()
        self._nav_dir_buf.append(d)
        if len(self._nav_dir_buf) > NAV_STABLE:
            self._nav_dir_buf.pop(0)
        if len(self._nav_dir_buf) < NAV_STABLE:
            return
        if d is None or not all(x == d for x in self._nav_dir_buf):
            return
        now = time.time()
        if now - self._nav_last_t < NAV_COOL:
            return
        self._nav_last_t = now
        if   d == "RIGHT": self._move_h(+1)
        elif d == "LEFT":  self._move_h(-1)
        elif d == "DOWN":  self._move_v(+1)
        elif d == "UP":    self._move_v(-1)

    def _move_h(self, delta):
        """Horizontal move within current row (wraps)."""
        row, col = divmod(self.selected, 2)
        self.selected = row * 2 + (col + delta) % 2

    def _move_v(self, delta):
        """Vertical move within current column (wraps)."""
        row, col = divmod(self.selected, 2)
        self.selected = ((row + delta) % 2) * 2 + col

    def _card_clicked(self, idx):
        self.selected = idx
        self._start_countdown()

    def _handle_msg(self, msg):
        kind, payload = msg
        if kind == "voice":
            self._on_voice(payload)
        elif kind == "game_done":
            self._restore_launcher()

    def _on_voice(self, text):
        self._voice_lbl.configure(text=f'Heard: "{text}"', fg=C_CYAN)
        # "select" / "go" triggers launch of currently highlighted game
        if any(w in text for w in ("select", "go", "launch", "play")):
            self._start_countdown()
            return
        # Game name triggers selection + launch
        for idx, g in enumerate(GAMES):
            for trigger in g["voice"]:
                if trigger in text:
                    self.selected = idx
                    self._start_countdown()
                    return
        self._voice_lbl.configure(fg=C_DIM)

    # ── Countdown ──────────────────────────────────────────────────────────────
    def _start_countdown(self):
        self._overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._overlay.lift()
        self._countdown_n = COUNTDOWN
        self._cd_sub.configure(text=f"Launching: {GAMES[self.selected]['name']}")
        self._countdown_step()

    def _countdown_step(self):
        if self._countdown_n > 0:
            self._cd_lbl.configure(text=str(self._countdown_n))
            self._countdown_n -= 1
            self.root.after(1000, self._countdown_step)
        else:
            self._cd_lbl.configure(text="GO!")
            self.root.after(400, self._launch_game)

    # ── Launch ─────────────────────────────────────────────────────────────────
    def _launch_game(self):
        self._overlay.place_forget()
        # Release webcam before spawning processes that need it
        self._stop_threads()

        g = GAMES[self.selected]
        self._game_proc = subprocess.Popen(g["cmd"], cwd=g["cwd"])
        self._main_proc = subprocess.Popen(
            [_PYTHON, MAIN_PY, "--skip-launcher", "--mode", g["mode"]],
            cwd=HERE,
        )
        self.root.iconify()
        threading.Thread(target=self._monitor_game, daemon=True).start()

    def _monitor_game(self):
        """Wait for game to close, then signal restore via queue."""
        if self._game_proc:
            self._game_proc.wait()
        if self._main_proc and self._main_proc.poll() is None:
            self._main_proc.terminate()
        self.cmd_queue.put(("game_done", None))

    def _restore_launcher(self):
        self._game_proc = None
        self._main_proc = None
        self._start_threads()
        self.root.deiconify()
        self.root.lift()
        self._voice_lbl.configure(text="Mic ready", fg=C_DIM)

    # ── Close ───────────────────────────────────────────────────────────────────
    def _on_close(self):
        self._stop_threads()
        self.root.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────
def run_launcher():
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    # If running standalone (not via main.py), just show the launcher
    run_launcher()
