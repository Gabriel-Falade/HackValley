import pygame
import random

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION BLOCK
# ═══════════════════════════════════════════════════════════════
CONFIG = {
    "SCREEN_WIDTH":   800,
    "SCREEN_HEIGHT":  600,
    "CELL":           30,       # grid cell size in pixels (bigger = easier to see)
    "FPS_NORMAL":     6,        # ticks per second at normal speed (was 10 — slowed down)
    "FPS_TURBO":      11,       # ticks per second during blink-boost (X/E)
    "TURBO_DURATION": 4,        # snake ticks the boost lasts
    "SLOW_DIVISOR":   2,        # eyebrow hold (Z) slows game to FPS/SLOW_DIVISOR
    "BONUS_TICKS":    20,       # snake ticks before bonus food vanishes
    "BONUS_POINTS":   5,
}

# Colors
COLOR_BG        = (15,  15,  20)
COLOR_GRID      = (25,  25,  35)
COLOR_SNAKE     = (0,   210, 100)
COLOR_HEAD      = (0,   255, 130)
COLOR_FOOD      = (220, 50,  50)
COLOR_BONUS     = (255, 210, 0)
COLOR_TEXT      = (255, 255, 255)
COLOR_DIMTEXT   = (120, 120, 140)
COLOR_TURBO     = (180, 80,  255)
COLOR_SLOW      = (80,  160, 255)

COLS = CONFIG["SCREEN_WIDTH"]  // CONFIG["CELL"]
ROWS = CONFIG["SCREEN_HEIGHT"] // CONFIG["CELL"]

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def cell_rect(col, row):
    c = CONFIG["CELL"]
    return pygame.Rect(col * c + 1, row * c + 1, c - 2, c - 2)


# ═══════════════════════════════════════════════════════════════
# GAME
# ═══════════════════════════════════════════════════════════════

class OGSnake:
    """
    Classic grid-based snake, tuned for FacePlay.

    FacePlay key behaviour (important for input design):
      • LEFT / RIGHT / UP / DOWN  — pyautogui.keyDown() while head is turned,
        keyUp() when head returns.  Pygame sees these as a held key, so both
        KEYDOWN (fires once on entry) and get_pressed() (true while held) work.
        We use KEYDOWN for direction changes — one tap = one turn, no repeating.

      • X / E (blink)  — pyautogui.press() = keydown + keyup in ~1 ms.
        get_pressed() MISSES these entirely.  Must use KEYDOWN event.

      • C (left wink)  — same as blink: KEYDOWN event only.

      • Z (eyebrow hold) — pyautogui.keyDown() while raised, keyUp() on release.
        get_pressed() works correctly.

      • ESC — pyautogui.press('escape').  Use KEYDOWN event.
    """

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (CONFIG["SCREEN_WIDTH"], CONFIG["SCREEN_HEIGHT"]))
        pygame.display.set_caption("FacePlay OG Snake  ◄ CLICK HERE to focus")
        self.font   = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_s = pygame.font.SysFont("Consolas", 15)
        self.clock  = pygame.time.Clock()
        self.reset()

    # ── State ───────────────────────────────────────────────────
    def reset(self):
        mid = (COLS // 2, ROWS // 2)
        self.body      = [mid, (mid[0] - 1, mid[1]), (mid[0] - 2, mid[1])]
        self.direction = (1, 0)     # (col_delta, row_delta); start moving right
        self.queued    = (1, 0)     # next direction, buffered from input

        self.score      = 0
        self.game_over  = False
        self.paused     = False

        # Tick counter — snake moves on every Nth pygame clock tick
        self.tick_accum = 0

        # Blink boost (X/E): turbo speed for N snake-ticks
        self.turbo_ticks = 0

        # Bonus food (C wink)
        self.bonus_cell  = None
        self.bonus_ticks = 0   # snake-ticks remaining

        self.spawn_food()

    def spawn_food(self):
        occupied = set(self.body)
        if self.bonus_cell:
            occupied.add(self.bonus_cell)
        options = [(c, r) for c in range(COLS) for r in range(ROWS)
                   if (c, r) not in occupied]
        self.food_cell = random.choice(options) if options else (0, 0)

    def spawn_bonus(self):
        occupied = set(self.body) | {self.food_cell}
        options  = [(c, r) for c in range(COLS) for r in range(ROWS)
                    if (c, r) not in occupied]
        if options:
            self.bonus_cell  = random.choice(options)
            self.bonus_ticks = CONFIG["BONUS_TICKS"]

    # ── Input ───────────────────────────────────────────────────
    def handle_events(self):
        """
        Process all KEYDOWN events.
        Direction changes are queued here; they apply on the next snake tick
        so rapid inputs between ticks aren't lost.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()

            if event.type != pygame.KEYDOWN:
                continue

            k = event.key

            # ── Pause / resume (ESC = voice command in FacePlay) ──
            if k == pygame.K_ESCAPE:
                if self.game_over:
                    self.reset()
                else:
                    self.paused = not self.paused

            # ── Restart on game over ──────────────────────────────
            if self.game_over and k in (pygame.K_x, pygame.K_e,
                                        pygame.K_SPACE, pygame.K_RETURN):
                self.reset()
                return

            if self.paused or self.game_over:
                continue

            # ── Direction (LEFT/RIGHT/UP/DOWN head tilt) ──────────
            # Block 180° reversals — snake can't turn back into itself.
            dx, dy = self.direction
            if k in (pygame.K_LEFT, pygame.K_a)  and dx == 0:
                self.queued = (-1, 0)
            elif k in (pygame.K_RIGHT, pygame.K_d) and dx == 0:
                self.queued = (1, 0)
            elif k in (pygame.K_UP, pygame.K_w)   and dy == 0:
                self.queued = (0, -1)
            elif k in (pygame.K_DOWN, pygame.K_s) and dy == 0:
                self.queued = (0, 1)

            # ── Blink both eyes (X = Attack, E = Interact) ────────
            # Activates turbo speed for TURBO_DURATION snake-ticks.
            elif k in (pygame.K_x, pygame.K_e):
                self.turbo_ticks = CONFIG["TURBO_DURATION"]

            # ── Left wink (C) → spawn gold bonus food ─────────────
            elif k == pygame.K_c:
                self.spawn_bonus()

    def handle_held(self):
        """Z (eyebrow hold) is a held key — get_pressed works correctly."""
        keys = pygame.key.get_pressed()
        self.slow_mode = bool(keys[pygame.K_z])

    # ── Update (tick-based) ─────────────────────────────────────
    def advance_snake(self):
        """Move the snake one grid cell. Called once per snake tick."""
        self.direction = self.queued

        head = (self.body[0][0] + self.direction[0],
                self.body[0][1] + self.direction[1])

        # Wrap walls — never die from touching an edge
        head = (head[0] % COLS, head[1] % ROWS)

        # Self-collision → game over
        if head in self.body:
            self.game_over = True
            return

        self.body.insert(0, head)

        ate = False
        if head == self.food_cell:
            self.score += 1
            ate = True
            self.spawn_food()
        elif head == self.bonus_cell:
            self.score    += CONFIG["BONUS_POINTS"]
            self.bonus_cell  = None
            self.bonus_ticks = 0
            ate = True

        if not ate:
            self.body.pop()

        # Tick bonus food countdown
        if self.bonus_cell is not None:
            self.bonus_ticks -= 1
            if self.bonus_ticks <= 0:
                self.bonus_cell  = None
                self.bonus_ticks = 0

        # Tick turbo countdown
        if self.turbo_ticks > 0:
            self.turbo_ticks -= 1

    def update(self, dt_ms):
        if self.paused or self.game_over:
            return

        # Choose target milliseconds per snake tick
        if self.turbo_ticks > 0:
            ms_per_tick = 1000 / CONFIG["FPS_TURBO"]
        elif self.slow_mode:
            ms_per_tick = 1000 / (CONFIG["FPS_NORMAL"] / CONFIG["SLOW_DIVISOR"])
        else:
            ms_per_tick = 1000 / CONFIG["FPS_NORMAL"]

        self.tick_accum += dt_ms
        while self.tick_accum >= ms_per_tick:
            self.tick_accum -= ms_per_tick
            self.advance_snake()
            if self.game_over:
                break

    # ── Draw ────────────────────────────────────────────────────
    def draw(self):
        self.screen.fill(COLOR_BG)
        self._draw_grid()
        self._draw_food()
        self._draw_snake()
        self._draw_hud()

        if self.paused:
            self._overlay("PAUSED", 'Say "resume"  or  press ESC')
        elif self.game_over:
            self._overlay("GAME OVER",
                          f"Score: {self.score}   |   Blink (X/E) or ESC to restart")

        pygame.display.flip()

    def _draw_grid(self):
        c = CONFIG["CELL"]
        for col in range(COLS):
            for row in range(ROWS):
                pygame.draw.rect(self.screen, COLOR_GRID,
                                 cell_rect(col, row), 1)

    def _draw_food(self):
        pygame.draw.rect(self.screen, COLOR_FOOD,
                         cell_rect(*self.food_cell))

        if self.bonus_cell:
            # Blink the bonus cell when < 1/3 of time remains
            frac = self.bonus_ticks / CONFIG["BONUS_TICKS"]
            if frac > 0.33 or (pygame.time.get_ticks() // 200) % 2 == 0:
                pygame.draw.rect(self.screen, COLOR_BONUS,
                                 cell_rect(*self.bonus_cell))
                # Small "+5" label
                lbl = self.font_s.render("+5", True, COLOR_BG)
                r   = cell_rect(*self.bonus_cell)
                self.screen.blit(lbl, lbl.get_rect(center=r.center))

    def _draw_snake(self):
        for i, cell in enumerate(self.body):
            if i == 0:
                color = COLOR_TURBO if self.turbo_ticks > 0 else COLOR_HEAD
            else:
                color = COLOR_TURBO if self.turbo_ticks > 0 else COLOR_SNAKE
                # Fade tail toward background
                fade  = max(0, 1.0 - i / len(self.body))
                color = tuple(int(c * fade + bg * (1 - fade))
                              for c, bg in zip(color, COLOR_BG))
            pygame.draw.rect(self.screen, color, cell_rect(*cell))

    def _draw_hud(self):
        # Score
        self.screen.blit(
            self.font.render(f"Score: {self.score}", True, COLOR_TEXT),
            (8, 6))

        # Active gesture indicators
        x = CONFIG["SCREEN_WIDTH"] - 8
        indicators = []
        if self.turbo_ticks > 0:
            indicators.append((f"TURBO  ({self.turbo_ticks})", COLOR_TURBO))
        if self.slow_mode:
            indicators.append(("SLOW-MO  (eyebrow)", COLOR_SLOW))
        if self.bonus_cell:
            secs = self.bonus_ticks  # counted in snake-ticks, small number
            indicators.append((f"BONUS FOOD  ({secs} ticks)", COLOR_BONUS))

        for i, (txt, col) in enumerate(indicators):
            surf = self.font_s.render(txt, True, col)
            self.screen.blit(surf, surf.get_rect(topright=(x, 8 + i * 22)))

        # Controls cheat-sheet — bottom of screen
        hints = [
            "Arrows/WASD → steer",
            "Blink (X/E) → turbo speed",
            "Left wink (C) → bonus food",
            "Eyebrow hold (Z) → slow-mo",
            'Say "pause" / "resume"  or  ESC → pause',
        ]
        for i, h in enumerate(hints):
            surf = self.font_s.render(h, True, COLOR_DIMTEXT)
            self.screen.blit(surf,
                surf.get_rect(bottomleft=(8,
                    CONFIG["SCREEN_HEIGHT"] - 4 - i * 18)))

    def _overlay(self, title, subtitle):
        W, H = CONFIG["SCREEN_WIDTH"], CONFIG["SCREEN_HEIGHT"]
        s = pygame.Surface((W, H), pygame.SRCALPHA)
        s.fill((0, 0, 0, 170))
        self.screen.blit(s, (0, 0))

        t = self.font.render(title, True, COLOR_TEXT)
        self.screen.blit(t, t.get_rect(center=(W // 2, H // 2 - 18)))

        sub = self.font_s.render(subtitle, True, (180, 180, 180))
        self.screen.blit(sub, sub.get_rect(center=(W // 2, H // 2 + 16)))

    # ── Main loop ───────────────────────────────────────────────
    def run(self):
        while True:
            dt = self.clock.tick(120)   # render at up to 120 fps; dt drives snake speed
            self.handle_events()        # KEYDOWN events — catches fast blinks/winks
            self.handle_held()          # get_pressed — held eyebrow Z
            self.update(dt)
            self.draw()


if __name__ == "__main__":
    OGSnake().run()
