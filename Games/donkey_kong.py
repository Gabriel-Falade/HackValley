#!/usr/bin/env python3
"""
FacePlay Donkey Kong — face-gesture controlled.
No external assets required; all graphics drawn with pygame primitives.
"""
import pygame
import sys
import random
import math

# ═══════════════════════════════════════════════════════════════
# FacePlay key mapping
# ═══════════════════════════════════════════════════════════════
#
#   HEAD LEFT / RIGHT  → pyautogui.keyDown() while turned, keyUp() on return
#                        → LEFT / RIGHT arrow or A / D
#                        Mario walks continuously in that direction.
#
#   HEAD UP            → pyautogui.keyDown() while tilted up → UP arrow or W
#                        On platform : jump.
#                        On / near ladder : climb up.
#
#   HEAD DOWN          → pyautogui.keyDown() while tilted down → DOWN arrow or S
#                        On ladder : climb down.
#
#   BLINK both (X / E) → pyautogui.press() ≈ 1 ms KEYDOWN only
#                        Jump (alternative). Restart when game over.
#                        Must catch as KEYDOWN event — get_pressed misses it.
#
#   EYEBROW hold (Z)   → pyautogui.keyDown() while raised
#                        Halves game speed (more reaction time for patients).
#
#   ESC / voice        → pause / unpause
#   SPACE              → restart on game over / win
#
# IMPORTANT: Click the game window first to give it keyboard focus.

W, H = 900, 700
FPS  = 60

GRAVITY    = 0.46
MARIO_SPD  = 3.2
JUMP_VY    = -13.5
LADDER_SPD = 2.6
BARREL_SPD = 2.4
BARREL_R   = 11
MARIO_W    = 20
MARIO_H    = 28

SPAWN_INTERVAL = 160   # frames between barrel throws (~2.7 s at 60 fps)

# ── Colors ────────────────────────────────────────────────────
C_BG       = (8,   8,   24)
C_PLAT     = (190, 130, 60)
C_PLAT_HI  = (215, 165, 85)
C_LADDER   = (195, 155, 65)
C_RUNG     = (162, 128, 48)
C_MARIO    = (195, 55,  30)
C_SKIN     = (225, 180, 130)
C_HAT      = (160, 40,  20)
C_OVERALL  = (55,  75,  200)
C_DK_BODY  = (110, 55,  10)
C_DK_FACE  = (155, 100, 40)
C_BARREL   = (148, 78,  18)
C_BAR_HI   = (198, 128, 55)
C_PAULINE  = (255, 128, 175)
C_HEART    = (255, 50,  75)
C_TEXT     = (255, 255, 255)
C_DIM      = (108, 108, 128)
C_GOLD     = (255, 215, 0)
C_SLOW     = (80,  160, 255)
C_GREEN    = (80,  225, 115)
C_RED      = (255, 70,  70)

# ═══════════════════════════════════════════════════════════════
# Level geometry
# ═══════════════════════════════════════════════════════════════
# Each platform: (x1, y1_at_left, x2, y2_at_right)
# Higher y = lower on screen (pygame convention).
# The direction barrels roll is always downhill.

PLAT_DATA = [
    (50,  118, 860, 156),   # P0: DK's top platform, slopes DOWN to right
    (50,  256, 860, 218),   # P1: slopes DOWN to left
    (50,  376, 860, 414),   # P2: slopes DOWN to right
    (50,  496, 860, 458),   # P3: slopes DOWN to left
    (0,   640, 900, 640),   # P4: floor, flat
]

# Ladders: (center_x, lower_platform_idx, upper_platform_idx)
LADDER_DEF = [
    (680, 4, 3),   # Floor → P3  (Mario climbs right side of floor)
    (160, 3, 2),   # P3 → P2     (Mario climbs left side of P3)
    (700, 2, 1),   # P2 → P1     (Mario climbs right side of P2)
    (200, 1, 0),   # P1 → P0     (Mario climbs left side of P1)
]

DK_CX     = 105    # DK center x on P0 (left / high side)
PAULINE_X = 795    # Pauline x on P0  (right side, near the end)
WIN_X     = 730    # Mario x threshold to trigger win (on P0)

MARIO_START_X    = 80
MARIO_START_PLAT = 4   # starts on floor


# ── Geometry helpers ─────────────────────────────────────────

def plat_y(pidx: int, x: float) -> float:
    """Surface y-coordinate of platform pidx at the given x."""
    x1, y1, x2, y2 = PLAT_DATA[pidx]
    if x2 == x1:
        return float(y1)
    t = max(0.0, min(1.0, (x - x1) / (x2 - x1)))
    return y1 + t * (y2 - y1)


def plat_contains(pidx: int, x: float) -> bool:
    x1, y1, x2, y2 = PLAT_DATA[pidx]
    return x1 <= x <= x2


def plat_barrel_dir(pidx: int) -> int:
    """Direction barrels naturally roll on this platform (+1=right, -1=left)."""
    x1, y1, x2, y2 = PLAT_DATA[pidx]
    return 1 if y2 >= y1 else -1


# Pre-compute ladder geometry once
# Each entry: (cx, y_bottom, y_top, lower_pidx, upper_pidx)
LADDERS = []
for _cx, _lp, _up in LADDER_DEF:
    _yb = plat_y(_lp, _cx)
    _yt = plat_y(_up, _cx)
    LADDERS.append((_cx, _yb, _yt, _lp, _up))


# ═══════════════════════════════════════════════════════════════
# Mario
# ═══════════════════════════════════════════════════════════════

class Mario:
    def __init__(self):
        self.lives       = 3
        self.score       = 0
        self.jump_queued = False   # set on KEYDOWN (blink / UP)
        self.respawn()

    def respawn(self):
        sy = plat_y(MARIO_START_PLAT, MARIO_START_X)
        self.x          = float(MARIO_START_X)
        self.y          = sy - MARIO_H
        self.vx         = 0.0
        self.vy         = 0.0
        self.on_ground  = True
        self.on_ladder  = False
        self.lad_idx    = -1
        self.facing     = 1       # +1=right, -1=left
        self.alive      = True
        self.dying      = False
        self.die_timer  = 0
        self.invincible = 0       # flash frames after respawn
        self.walk_frame = 0       # animation counter

    @property
    def cx(self):    return self.x + MARIO_W / 2
    @property
    def cy(self):    return self.y + MARIO_H / 2
    @property
    def feet(self):  return self.y + MARIO_H

    def kill(self):
        if self.dying or self.invincible > 0:
            return
        self.dying     = True
        self.die_timer = 80
        self.vx = self.vy = 0.0

    def update(self, keys, slow: bool):
        if self.invincible > 0:
            self.invincible -= 1

        if self.dying:
            self.die_timer -= 1
            if self.die_timer <= 0:
                self.alive = False
            return

        spd  = MARIO_SPD   * (0.5 if slow else 1.0)
        lspd = LADDER_SPD  * (0.5 if slow else 1.0)
        grav = GRAVITY     * (0.5 if slow else 1.0)

        lft = keys[pygame.K_LEFT]  or keys[pygame.K_a]
        rgt = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        up  = keys[pygame.K_UP]    or keys[pygame.K_w]
        dn  = keys[pygame.K_DOWN]  or keys[pygame.K_s]

        # ── Ladder mode ───────────────────────────────────────
        if self.on_ladder:
            cx_lad, yb, yt, lp, uppidx = LADDERS[self.lad_idx]
            self.vx = 0.0
            if up:
                self.vy = -lspd
                self.walk_frame += 1
            elif dn:
                self.vy = lspd
                self.walk_frame += 1
            else:
                self.vy = 0.0

            self.x  = cx_lad - MARIO_W / 2   # stay snapped to ladder
            self.y += self.vy

            # Climbed off the top
            if self.feet <= yt:
                self.y         = yt - MARIO_H
                self.vy        = 0.0
                self.on_ladder = False
                self.on_ground = True
                self.lad_idx   = -1

            # Descended off the bottom
            elif self.feet >= yb + 6:
                self.y         = yb - MARIO_H
                self.vy        = 0.0
                self.on_ladder = False
                self.on_ground = True
                self.lad_idx   = -1

        # ── Walking / jumping mode ────────────────────────────
        else:
            if lft:
                self.vx = -spd
                self.facing = -1
                if self.on_ground:
                    self.walk_frame += 1
            elif rgt:
                self.vx = spd
                self.facing = 1
                if self.on_ground:
                    self.walk_frame += 1
            else:
                self.vx = 0.0

            # Jump
            if (up or self.jump_queued) and self.on_ground:
                self.vy        = JUMP_VY
                self.on_ground = False

            self.jump_queued = False

            # Grab ladder (only when standing on a platform)
            if self.on_ground and (up or dn):
                for i, (cx_lad, yb, yt, lp, uppidx) in enumerate(LADDERS):
                    if abs(self.cx - cx_lad) < 20:
                        if up and abs(self.feet - yb) < 28:
                            # Grab from below, climb up
                            self.on_ladder = True
                            self.on_ground = False
                            self.lad_idx   = i
                            self.x         = cx_lad - MARIO_W / 2
                            self.vy        = -lspd
                            self.jump_queued = False
                            break
                        elif dn and abs(self.feet - yt) < 28:
                            # Grab from above, climb down
                            self.on_ladder = True
                            self.on_ground = False
                            self.lad_idx   = i
                            self.x         = cx_lad - MARIO_W / 2
                            self.vy        = lspd
                            break

            # Gravity
            if not self.on_ground:
                self.vy += grav

            self.x += self.vx
            self.y += self.vy

            # Screen edge clamp
            self.x = max(0.0, min(W - MARIO_W, self.x))

            # Platform collision
            self.on_ground = False
            for pidx in range(len(PLAT_DATA)):
                if plat_contains(pidx, self.cx):
                    sy = plat_y(pidx, self.cx)
                    if 0 <= self.feet - sy <= 14 and self.vy >= -1:
                        self.y         = sy - MARIO_H
                        self.vy        = 0.0
                        self.on_ground = True
                        break

            # Fell off screen
            if self.y > H + 40:
                self.kill()

    def draw(self, surf, frame: int):
        ix, iy = int(self.x), int(self.y)

        # Death spin
        if self.dying:
            t = 1.0 - self.die_timer / 80.0
            r = max(3, int(14 * (1 - t)))
            col = (
                int(C_MARIO[0]),
                int(C_MARIO[1] * (1 - t)),
                int(C_MARIO[2] * (1 - t)),
            )
            pygame.draw.circle(surf, col, (int(self.cx), int(self.cy)), r)
            return

        # Flicker when invincible
        if self.invincible > 0 and (self.invincible // 5) % 2:
            return

        # Walk animation: bob slightly
        bob = 1 if (self.walk_frame // 6) % 2 == 0 else 0

        # Overalls
        pygame.draw.rect(surf, C_OVERALL, (ix + 3, iy + 14 + bob, MARIO_W - 6, 14))
        # Shirt
        pygame.draw.rect(surf, C_MARIO,   (ix + 2, iy + 8  + bob, MARIO_W - 4, 8))
        # Head
        pygame.draw.circle(surf, C_SKIN, (ix + MARIO_W // 2, iy + 6 + bob), 7)
        # Hat brim
        pygame.draw.rect(surf, C_HAT, (ix + 1,  iy + 2 + bob, MARIO_W - 2, 3))
        # Hat top
        pygame.draw.rect(surf, C_HAT, (ix + 4,  iy      + bob, MARIO_W - 8, 3))
        # Boot
        bx = ix + 10 if self.facing == 1 else ix
        pygame.draw.ellipse(surf, (50, 30, 5), (bx, iy + MARIO_H - 5 + bob, 10, 5))


# ═══════════════════════════════════════════════════════════════
# Barrel
# ═══════════════════════════════════════════════════════════════

class Barrel:
    ROLL = 0
    FALL = 1
    DONE = 2

    def __init__(self, x: float, pidx: int):
        self.x     = x
        self.y     = plat_y(pidx, x)
        self.pidx  = pidx
        self.dir   = plat_barrel_dir(pidx)
        self.vx    = BARREL_SPD * self.dir
        self.vy    = 0.0
        self.state = self.ROLL
        self.rot   = 0.0
        self.scoreable = True

    @property
    def cx(self):  return self.x
    @property
    def cy(self):  return self.y - BARREL_R

    def update(self, slow: bool):
        if self.state == self.DONE:
            return

        spd  = BARREL_SPD * (0.5 if slow else 1.0)
        grav = GRAVITY    * (0.5 if slow else 1.0)
        self.rot += 6.0 * self.dir * (0.5 if slow else 1.0)

        if self.state == self.ROLL:
            self.x += self.vx
            self.y  = plat_y(self.pidx, self.x)
            x1, y1, x2, y2 = PLAT_DATA[self.pidx]

            if self.x > x2 or self.x < x1:
                # Roll off the edge → start falling straight down
                self.x    = max(x1, min(x2, self.x))   # clamp to edge
                self.state = self.FALL
                self.vy    = 1.0
                self.vx    = 0.0   # fall nearly vertically

        elif self.state == self.FALL:
            self.vy += grav
            self.x  += self.vx
            self.y  += self.vy

            # Check landing on any platform
            for pidx in range(len(PLAT_DATA)):
                if pidx == self.pidx:
                    continue
                if plat_contains(pidx, self.x):
                    sy = plat_y(pidx, self.x)
                    if self.y >= sy and self.vy > 0:
                        # Land: nudge slightly inside platform to avoid edge re-fall
                        self.pidx  = pidx
                        self.y     = sy
                        self.vy    = 0.0
                        self.dir   = plat_barrel_dir(pidx)
                        self.vx    = spd * self.dir
                        # Small nudge inward so we don't immediately roll off
                        if self.dir == 1:
                            self.x += 4
                        else:
                            self.x -= 4
                        self.state = self.ROLL
                        break

            # Off-screen → remove
            if self.x < -30 or self.x > W + 30 or self.y > H + 60:
                self.state = self.DONE

    def draw(self, surf):
        if self.state == self.DONE:
            return
        x, y = int(self.x), int(self.y)
        r = BARREL_R
        # Barrel body (concentric circles for a wood-plank look)
        pygame.draw.circle(surf, C_BARREL,  (x, y - r), r)
        pygame.draw.circle(surf, C_BAR_HI,  (x, y - r), r - 3)
        pygame.draw.circle(surf, C_BARREL,  (x, y - r), r - 6)
        # Rotation indicator line
        ang = math.radians(self.rot)
        ex  = int(x + (r - 2) * math.cos(ang))
        ey  = int((y - r) + (r - 2) * math.sin(ang))
        pygame.draw.line(surf, C_BAR_HI, (x, y - r), (ex, ey), 2)


# ═══════════════════════════════════════════════════════════════
# Game
# ═══════════════════════════════════════════════════════════════

class DKGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("FacePlay Donkey Kong  ◄ CLICK HERE to focus")
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont("Arial", 30, bold=True)
        self.font_s = pygame.font.SysFont("Arial", 20)
        self.font_h = pygame.font.SysFont("Arial", 15)
        self.reset()

    def reset(self):
        self.mario       = Mario()
        self.barrels     = []
        self.spawn_timer = SPAWN_INTERVAL // 2   # first barrel arrives sooner
        self.frame       = 0
        self.paused      = False
        self.slow_mode   = False
        self.game_over   = False
        self.you_win     = False
        self.dk_anim     = 0   # countdown for throw animation

    # ── Input ────────────────────────────────────────────────

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                k = event.key

                if k == pygame.K_ESCAPE:
                    if self.game_over or self.you_win:
                        self.reset()
                    else:
                        self.paused = not self.paused

                elif k in (pygame.K_x, pygame.K_e, pygame.K_SPACE, pygame.K_RETURN):
                    if self.game_over or self.you_win:
                        self.reset()
                    else:
                        # Blink → jump
                        self.mario.jump_queued = True

                elif k in (pygame.K_UP, pygame.K_w):
                    # Head-tilt UP also queues a jump (catches fast tilt entry)
                    self.mario.jump_queued = True

    def handle_held(self):
        keys = pygame.key.get_pressed()
        self.slow_mode = bool(keys[pygame.K_z])
        return keys

    # ── Update ───────────────────────────────────────────────

    def update(self, keys):
        if self.paused or self.game_over or self.you_win:
            return

        self.frame += 1

        # Mario
        self.mario.update(keys, self.slow_mode)

        if not self.mario.alive:
            self.mario.lives -= 1
            if self.mario.lives <= 0:
                self.game_over = True
            else:
                self.mario.respawn()
                self.mario.invincible = 120   # 2 s of flash immunity

        # Barrel spawning
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self.spawn_timer = SPAWN_INTERVAL
            self.dk_anim     = 35
            bx = float(DK_CX + 22)
            self.barrels.append(Barrel(bx, 0))

        if self.dk_anim > 0:
            self.dk_anim -= 1

        # Barrels update
        for b in self.barrels:
            b.update(self.slow_mode)

        # Score: award 100 pts when Mario is airborne above a passing barrel
        if not self.mario.on_ground and not self.mario.on_ladder:
            for b in self.barrels:
                if b.scoreable and b.state == Barrel.ROLL:
                    if abs(self.mario.cx - b.cx) < 34 and self.mario.feet < b.cy + 10:
                        self.mario.score += 100
                        b.scoreable = False

        # Remove finished barrels
        self.barrels = [b for b in self.barrels if b.state != Barrel.DONE]

        # Barrel-Mario collision
        if not self.mario.dying:
            hw = MARIO_W // 2 + BARREL_R - 3
            hh = MARIO_H // 2 + BARREL_R - 3
            for b in self.barrels:
                if b.state == Barrel.DONE:
                    continue
                if abs(self.mario.cx - b.cx) < hw and abs(self.mario.cy - b.cy) < hh:
                    self.mario.kill()
                    break

        # Win condition: Mario reaches Pauline on P0
        if (not self.mario.dying
                and self.mario.cx > WIN_X
                and plat_contains(0, self.mario.cx)):
            sy = plat_y(0, self.mario.cx)
            if abs(self.mario.feet - sy) < 24:
                self.you_win = True
                self.mario.score += 1000

    # ── Draw ─────────────────────────────────────────────────

    def draw(self):
        self.screen.fill(C_BG)

        self._draw_platforms()
        self._draw_ladders()
        self._draw_dk()
        self._draw_pauline()

        for b in self.barrels:
            b.draw(self.screen)

        self.mario.draw(self.screen, self.frame)
        self._draw_hud()

        if self.paused:
            self._overlay("PAUSED", 'Say "resume"  or  press ESC')
        elif self.game_over:
            self._overlay(
                "GAME OVER",
                f"Score: {self.mario.score}   |   Blink (X/E) or ESC to restart",
                color=C_RED,
            )
        elif self.you_win:
            self._overlay(
                "YOU WIN!",
                f"Score: {self.mario.score}   |   Blink (X/E) or ESC to play again",
                color=C_GREEN,
            )

        pygame.display.flip()

    def _draw_platforms(self):
        THICK = 10
        for pidx, (x1, y1, x2, y2) in enumerate(PLAT_DATA):
            pts = [
                (x1, y1),
                (x2, y2),
                (x2, y2 + THICK),
                (x1, y1 + THICK),
            ]
            pygame.draw.polygon(self.screen, C_PLAT, pts)
            pygame.draw.line(self.screen, C_PLAT_HI, (x1, y1), (x2, y2), 3)

    def _draw_ladders(self):
        for (cx, yb, yt, lp, uppidx) in LADDERS:
            yb_i, yt_i = int(yb), int(yt)
            pygame.draw.line(self.screen, C_LADDER, (cx - 8, yt_i), (cx - 8, yb_i), 3)
            pygame.draw.line(self.screen, C_LADDER, (cx + 8, yt_i), (cx + 8, yb_i), 3)
            y = yb_i
            while y > yt_i:
                pygame.draw.line(self.screen, C_RUNG, (cx - 7, y), (cx + 7, y), 2)
                y -= 16

    def _draw_dk(self):
        x  = DK_CX
        fy = int(plat_y(0, DK_CX))  # feet y (platform surface)
        throwing = self.dk_anim > 18

        # Body
        pygame.draw.ellipse(self.screen, C_DK_BODY, (x - 22, fy - 58, 44, 48))
        # Head
        pygame.draw.circle(self.screen, C_DK_BODY, (x, fy - 63), 20)
        # Face patch
        pygame.draw.ellipse(self.screen, C_DK_FACE, (x - 12, fy - 74, 24, 18))
        # Eyes
        pygame.draw.circle(self.screen, (255, 255, 255), (x - 6,  fy - 68), 5)
        pygame.draw.circle(self.screen, (255, 255, 255), (x + 6,  fy - 68), 5)
        pygame.draw.circle(self.screen, (10,  10,  10),  (x - 5,  fy - 68), 2)
        pygame.draw.circle(self.screen, (10,  10,  10),  (x + 7,  fy - 68), 2)

        if throwing:
            # Right arm raised — throwing pose
            pygame.draw.line(self.screen, C_DK_BODY, (x + 20, fy - 48), (x + 50, fy - 75), 11)
            pygame.draw.circle(self.screen, C_DK_BODY, (x + 50, fy - 75), 11)
            # Barrel in hand
            pygame.draw.circle(self.screen, C_BARREL, (x + 58, fy - 83), BARREL_R - 1)
            pygame.draw.circle(self.screen, C_BAR_HI, (x + 58, fy - 83), BARREL_R - 4)
        else:
            # Arms at sides / hanging
            pygame.draw.line(self.screen, C_DK_BODY, (x + 20, fy - 45), (x + 38, fy - 20), 11)
            pygame.draw.circle(self.screen, C_DK_BODY, (x + 38, fy - 20), 11)
            pygame.draw.line(self.screen, C_DK_BODY, (x - 20, fy - 45), (x - 38, fy - 20), 11)
            pygame.draw.circle(self.screen, C_DK_BODY, (x - 38, fy - 20), 11)

    def _draw_pauline(self):
        x  = PAULINE_X
        fy = int(plat_y(0, PAULINE_X))

        # Dress
        pts = [(x - 12, fy), (x + 12, fy), (x + 16, fy - 32), (x - 16, fy - 32)]
        pygame.draw.polygon(self.screen, C_PAULINE, pts)
        # Torso
        pygame.draw.rect(self.screen, C_PAULINE, (x - 9, fy - 55, 18, 24))
        # Head
        pygame.draw.circle(self.screen, (225, 185, 135), (x, fy - 62), 12)
        # Hair
        pygame.draw.arc(self.screen, (100, 55, 15),
                        (x - 13, fy - 76, 26, 26), 0, math.pi, 5)
        # HELP! text (blinking)
        if (self.frame // 28) % 2 == 0:
            hs = self.font_s.render("HELP!", True, C_GOLD)
            self.screen.blit(hs, (x - hs.get_width() // 2, fy - 100))

    def _draw_hud(self):
        # Score
        sc = self.font.render(f"Score: {self.mario.score}", True, C_TEXT)
        self.screen.blit(sc, (20, 10))

        # Lives — mini Marios
        for i in range(self.mario.lives):
            mx = 300 + i * 34
            pygame.draw.circle(self.screen, (225, 180, 130), (mx + 8, 17), 5)
            pygame.draw.rect(self.screen, C_MARIO,   (mx + 2, 21, 12, 8))
            pygame.draw.rect(self.screen, C_HAT,     (mx + 3, 12, 10, 5))

        # Slow mode indicator
        if self.slow_mode:
            sl = self.font_s.render("SLOW  (eyebrow hold)", True, C_SLOW)
            self.screen.blit(sl, (20, 48))

        # Controls cheat-sheet (two lines at the bottom)
        hints = [
            "Left/Right head  → walk   |   Head up → jump / climb   |   Head down → descend",
            'Blink (X/E) → jump / restart   |   Eyebrow hold (Z) → slow   |   ESC → pause',
        ]
        for i, h in enumerate(hints):
            s = self.font_h.render(h, True, C_DIM)
            self.screen.blit(s, s.get_rect(center=(W // 2, H - 28 + i * 18)))

    def _overlay(self, title: str, subtitle: str, color=C_TEXT):
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 165))
        self.screen.blit(surf, (0, 0))
        t = self.font.render(title, True, color)
        self.screen.blit(t, t.get_rect(center=(W // 2, H // 2 - 22)))
        s = self.font_s.render(subtitle, True, (190, 190, 190))
        self.screen.blit(s, s.get_rect(center=(W // 2, H // 2 + 18)))

    # ── Main loop ────────────────────────────────────────────

    def run(self):
        while True:
            self.handle_events()        # KEYDOWN (catches fast blinks)
            keys = self.handle_held()   # get_pressed (held keys)
            self.update(keys)
            self.draw()
            self.clock.tick(FPS)


if __name__ == "__main__":
    DKGame().run()
