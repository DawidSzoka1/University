from load_img import load_images

player_walk = load_images("playerImages/move/sprite_", 10)
player_idle = load_images("playerImages/idle/sprite_", 7)
player_special_attack = load_images("playerImages/attack_special/sprite_", 10)
player_basic_attack = load_images("playerImages/attack_basic/sprite_", 10)
HEIGHT = 600
WIDTH = 800
STAGE = 1
EnemyPerStage = STAGE * 5
EXPPerStage = STAGE * 5 + 25