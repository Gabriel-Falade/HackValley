import pygame
import random
import math

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION BLOCK
# ═══════════════════════════════════════════════════════════════
CONFIG = {
    "SCREEN_WIDTH":    1000,
    "SCREEN_HEIGHT":   700,
    "FPS":             60,
    "SNAKE_SPEED":     2.2,   # pixels per frame at neutral (was 4 — slowed down)
    "BOOST_MULT":      1.6,   # UP arrow/W (head UP) → speed multiplier
    "BRAKE_MULT":      0.4,   # DOWN arrow/S (head DOWN) → speed multiplier
    "DASH_MULT":       2.8,   # X/E blink burst multiplier
    "DASH_FRAMES":     22,    # frames the dash burst lasts after a blink
    "TURN_SPEED":      3,     # degrees per frame (was 4 — slightly less sensitive)
    "SNAKE_SIZE":      18,    # head radius in pixels (bigger)
    "FOOD_SIZE":       10,
    "BONUS_FOOD_SIZE": 14,
    "BONUS_DURATION":  300,   # frames bonus food stays before vanishing
    "BONUS_POINTS":    5,
}

# Colors
COLOR_BG         = (20,  20,  30)
COLOR_SNAKE      = (0,   255, 127)
COLOR_SHIELD     = (0,   191, 255)
COLOR_FOOD       = (255, 69,  0)
COLOR_BONUS      = (255, 215, 0)    # gold
COLOR_TEXT       = (255, 255, 255)
COLOR_DASH       = (180, 100, 255)
COLOR_BOOST      = (255, 220, 60)
COLOR_BRAKE      = (120, 180, 255)
COLOR_WARN       = (255, 60,  60)

# ═══════════════════════════════════════════════════════════════
# GAME
# ═══════════════════════════════════════════════════════════════

class FaceSnake:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (CONFIG["SCREEN_WIDTH"], CONFIG["SCREEN_HEIGHT"]))
        pygame.display.set_caption("FacePlay Snake  ◄ CLICK HERE to focus")
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_s = pygame.font.SysFont("Arial", 18)
        self.reset_game()

    # ── State reset ────────────────────────────────────────────
    def reset_game(self):
        cx, cy = CONFIG["SCREEN_WIDTH"] // 2, CONFIG["SCREEN_HEIGHT"] // 2
        self.x, self.y = float(cx), float(cy)
        self.angle   = 0.0          # degrees, 0 = right
        self.body    = [(self.x, self.y)]
        self.length  = 20
        self.score   = 0
        self.game_over = False
        self.paused    = False

        # Gesture states
        self.shield_active = False  # Z held → wall + self immunity
        self.dash_frames   = 0      # countdown after blink fires
        self.boost_active  = False  # UP arrow/W held
        self.brake_active  = False  # DOWN arrow/S held

        # Bonus food (C left-wink)
        self.bonus_pos   = None
        self.bonus_timer = 0

        self.spawn_food()

    def spawn_food(self):
        margin = 40
        self.food_pos = (
            random.randint(margin, CONFIG["SCREEN_WIDTH"]  - margin),
            random.randint(margin, CONFIG["SCREEN_HEIGHT"] - margin),
        )

    def spawn_bonus(self):
        """Left wink (C) drops a bonus food worth 5 pts for ~5 seconds."""
        margin = 40
        self.bonus_pos   = (
            random.randint(margin, CONFIG["SCREEN_WIDTH"]  - margin),
            random.randint(margin, CONFIG["SCREEN_HEIGHT"] - margin),
        )
        self.bonus_timer = CONFIG["BONUS_DURATION"]

    # ── Input ───────────────────────────────────────────────────
    def handle_events(self):
        """
        FacePlay fires blink (X/E) and wink (C) as pyautogui.press() —
        a keydown + keyup in microseconds. Using get_pressed() misses them.
        All single-press gestures must be caught in the KEYDOWN event loop.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()

            if event.type == pygame.KEYDOWN:
                # ESC — voice "pause" / "resume"
                if event.key == pygame.K_ESCAPE:
                    if self.game_over:
                        self.reset_game()       # ESC also restarts
                    else:
                        self.paused = not self.paused

                # BLINK both eyes (X = Attack, E = Interact)
                # → burst of speed for DASH_FRAMES frames
                # → while game_over: restart
                elif event.key in (pygame.K_x, pygame.K_e):
                    if self.game_over:
                        self.reset_game()
                    else:
                        self.dash_frames = CONFIG["DASH_FRAMES"]

                # LEFT WINK (C) → spawn bonus food
                elif event.key == pygame.K_c:
                    if not self.game_over and not self.paused:
                        self.spawn_bonus()

    def handle_held(self):
        """
        Held keys for continuous steering.
        LEFT/RIGHT head tilt → steer.
        UP head tilt          → boost (1.6×).
        DOWN head tilt        → brake (0.45×) — useful for tight turns.
        Z eyebrow hold        → shield (wall wrap + self immunity).
        """
        keys = pygame.key.get_pressed()

        # Steer — Attack: arrows, Interact: WASD (both work simultaneously)
        if keys[pygame.K_LEFT]  or keys[pygame.K_a]:
            self.angle -= CONFIG["TURN_SPEED"]
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.angle += CONFIG["TURN_SPEED"]

        # UP head tilt → boost speed
        self.boost_active = bool(keys[pygame.K_UP]   or keys[pygame.K_w])
        # DOWN head tilt → brake
        self.brake_active = bool(keys[pygame.K_DOWN] or keys[pygame.K_s])

        # Eyebrow hold → shield
        self.shield_active = bool(keys[pygame.K_z])

    # ── Update ──────────────────────────────────────────────────
    def update(self):
        if self.paused or self.game_over:
            return

        # Tick bonus food timer
        if self.bonus_timer > 0:
            self.bonus_timer -= 1
            if self.bonus_timer == 0:
                self.bonus_pos = None

        # Resolve speed — priority: dash > brake > boost > neutral
        speed = CONFIG["SNAKE_SPEED"]
        if self.dash_frames > 0:
            speed *= CONFIG["DASH_MULT"]
            self.dash_frames -= 1
        elif self.brake_active:
            speed *= CONFIG["BRAKE_MULT"]
        elif self.boost_active:
            speed *= CONFIG["BOOST_MULT"]

        # Move head
        rad     = math.radians(self.angle)
        self.x += speed * math.cos(rad)
        self.y += speed * math.sin(rad)

        # Wall behaviour — WRAP instead of death.
        # Much more forgiving for face control: you never die from drift.
        W, H = CONFIG["SCREEN_WIDTH"], CONFIG["SCREEN_HEIGHT"]
        self.x %= W
        self.y %= H

        # Update body trail
        self.body.insert(0, (self.x, self.y))
        if len(self.body) > self.length:
            self.body.pop()

        sz = CONFIG["SNAKE_SIZE"]

        # Collect regular food
        if math.hypot(self.x - self.food_pos[0], self.y - self.food_pos[1]) < sz:
            self.score  += 1
            self.length += 10
            self.spawn_food()

        # Collect bonus food
        if (self.bonus_pos is not None
                and math.hypot(self.x - self.bonus_pos[0],
                               self.y - self.bonus_pos[1]) < sz + 4):
            self.score      += CONFIG["BONUS_POINTS"]
            self.bonus_pos   = None
            self.bonus_timer = 0

        # Self-collision (shield disables it)
        if not self.shield_active:
            for seg in self.body[20:]:      # skip head segments
                if math.hypot(self.x - seg[0], self.y - seg[1]) < sz * 0.6:
                    self.game_over = True
                    break

    # ── Draw ────────────────────────────────────────────────────
    def draw(self):
        self.screen.fill(COLOR_BG)

        # Food
        pygame.draw.circle(self.screen, COLOR_FOOD,
                           self.food_pos, CONFIG["FOOD_SIZE"])

        # Bonus food (pulsing ring to stand out)
        if self.bonus_pos is not None:
            bsz  = CONFIG["BONUS_FOOD_SIZE"]
            # pulse size based on remaining timer
            pulse = int(3 * math.sin(self.bonus_timer * 0.15))
            pygame.draw.circle(self.screen, COLOR_BONUS,
                               self.bonus_pos, bsz + pulse)
            # draw countdown ring (thins as timer falls)
            frac = self.bonus_timer / CONFIG["BONUS_DURATION"]
            arc_rect = pygame.Rect(self.bonus_pos[0] - bsz - 6,
                                   self.bonus_pos[1] - bsz - 6,
                                   (bsz + 6) * 2, (bsz + 6) * 2)
            pygame.draw.arc(self.screen, COLOR_WARN, arc_rect,
                            0, math.tau * frac, 3)

        # Snake body
        if self.shield_active:
            snake_color = COLOR_SHIELD
        elif self.dash_frames > 0:
            snake_color = COLOR_DASH
        elif self.boost_active:
            snake_color = COLOR_BOOST
        elif self.brake_active:
            snake_color = COLOR_BRAKE
        else:
            snake_color = COLOR_SNAKE

        for i, seg in enumerate(self.body):
            size = max(4, CONFIG["SNAKE_SIZE"] - i // 8)
            pygame.draw.circle(self.screen, snake_color,
                               (int(seg[0]), int(seg[1])), size)

        # HUD — score
        self.screen.blit(
            self.font.render(f"Score: {self.score}", True, COLOR_TEXT), (20, 16))

        # HUD — active gesture labels
        hud_y = 50
        if self.shield_active:
            self._hud(f"SHIELD  (hold eyebrow)", COLOR_SHIELD, hud_y); hud_y += 26
        if self.dash_frames > 0:
            self._hud(f"DASH  ({self.dash_frames}f)", COLOR_DASH, hud_y); hud_y += 26
        if self.boost_active and self.dash_frames == 0:
            self._hud("BOOST  (head UP)", COLOR_BOOST, hud_y); hud_y += 26
        if self.brake_active and self.dash_frames == 0:
            self._hud("BRAKE  (head DOWN)", COLOR_BRAKE, hud_y); hud_y += 26
        if self.bonus_pos is not None:
            secs = math.ceil(self.bonus_timer / CONFIG["FPS"])
            self._hud(f"BONUS FOOD  {secs}s", COLOR_BONUS, hud_y)

        # Controls cheat-sheet — bottom left
        hints = [
            "LEFT/RIGHT head  → steer",
            "UP head          → boost",
            "DOWN head        → brake",
            "Blink (X/E)      → dash burst",
            "Left wink (C)    → bonus food",
            "Eyebrow hold (Z) → shield",
            'Say "pause" / "resume"  or  ESC → pause',
        ]
        for i, h in enumerate(hints):
            surf = self.font_s.render(h, True, (110, 110, 130))
            self.screen.blit(surf, (16, CONFIG["SCREEN_HEIGHT"] - 24 - i * 22))

        # Overlays
        if self.paused:
            self._overlay('PAUSED',        'Say "resume"  or  press ESC')
        elif self.game_over:
            self._overlay('GAME OVER',
                          f"Score: {self.score}   |   Blink (X/E) or ESC to restart")

        pygame.display.flip()

    def _hud(self, text, color, y):
        self.screen.blit(self.font_s.render(text, True, color), (20, y))

    def _overlay(self, title, subtitle):
        W, H = CONFIG["SCREEN_WIDTH"], CONFIG["SCREEN_HEIGHT"]
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 160))
        self.screen.blit(surf, (0, 0))

        t = self.font.render(title, True, COLOR_TEXT)
        self.screen.blit(t, t.get_rect(center=(W // 2, H // 2 - 20)))

        s = self.font_s.render(subtitle, True, (190, 190, 190))
        self.screen.blit(s, s.get_rect(center=(W // 2, H // 2 + 18)))

    # ── Main loop ───────────────────────────────────────────────
    def run(self):
        while True:
            self.handle_events()    # KEYDOWN events first (catches fast blinks)
            self.handle_held()      # then held keys
            self.update()
            self.draw()
            self.clock.tick(CONFIG["FPS"])


if __name__ == "__main__":
    FaceSnake().run()
