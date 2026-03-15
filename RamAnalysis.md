# RAM Feature Augmentation — Todo

Missing RAM features useful for the RL agent, ordered by priority.

---

## High Priority

- [x] `0x0057` — Mario horizontal speed (momentum, wall jumps, gap traversal)
- [x] `0x009F` — Mario vertical velocity (ascending vs descending jump arc)
- [x] `0x001D` — Float state: ground / jumping / sliding flagpole
- [x] `0x0756` — Power state: small / big / fire (uses 0x0756 which maps 0/1/2+ cleanly)
- [x] `0x07F8–0x07FA` — Game timer BCD digits (time pressure, avoid timeouts)
- [x] `0x079E` — Invincibility timer after collision (changes optimal behavior)

---

## Medium Priority

- [ ] `0x0016–0x001A` — Enemy types per slot (Goomba vs Koopa vs Bowser; currently all = -1)
- [ ] `0x001E–0x0022` — Enemy states: alive / stomped / falling
- [ ] `0x0058–0x005C` — Enemy horizontal speeds (stomp timing)
- [x] `0x0039` — Powerup type: mushroom / flower / star / 1UP (currently all encoded as 3)
- [ ] `0x006D` + `0x03AD` — Mario absolute X position in level (level progress reward signal)
- [ ] `0x079F` — Star timer (starman invincibility)

---

## Lower Priority

- [ ] `0x075E` — Coin count
- [ ] `0x075A` — Lives
- [ ] `0x07DD–0x07E2` — Score in BCD
- [ ] `0x0704` — Swimming flag
- [ ] `0x0723` — Scroll lock (level end / pipe transition)
- [ ] `0x0024–0x0025` + `0x008D` / `0x00D5` — Fireball positions (fire Mario offense)

---

## Implementation Plan

Add an `AugmentedRAMFeatures` wrapper in `src/wrappers/ram_wrappers.py` that appends a
scalar feature vector alongside (or concatenated to) the existing grid observation.

Implement in this order:
1. Mario speed (x, y) + float state + power state + invincibility + game timer  ← batch 1
2. Enemy types + enemy states + enemy speeds  ← batch 2
3. Powerup type + absolute Mario X + star timer  ← batch 3
4. Coins + lives + score + swimming + scroll lock + fireballs  ← batch 4
