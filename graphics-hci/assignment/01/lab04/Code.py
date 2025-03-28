import pygame
from pygame.locals import *
import sys
import random
import time
from tkinter import filedialog
from tkinter import *


pygame.init()  # Begin pygame

# Declaring variables to be used through the program
vec = pygame.math.Vector2
HEIGHT = 350
WIDTH = 700
ACC = 0.3
FRIC = -0.10
FPS = 60
FPS_CLOCK = pygame.time.Clock()
COUNT = 0

# Create the display
displaysurface = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Game")


# light shade of the button 
color_light = (170,170,170)
color_dark = (100,100,100)
color_white = (255,255,255) 
  
# defining a font
headingfont = pygame.font.SysFont("Verdana", 40)
regularfont = pygame.font.SysFont('Corbel',25)
smallerfont = pygame.font.SysFont('Corbel',16) 
text = regularfont.render('LOAD' , True , color_light)



# Run animation for the RIGHT
run_ani_R = [pygame.image.load("movement_animations/Player_Sprite_R.png"), pygame.image.load("movement_animations/Player_Sprite2_R.png"),
             pygame.image.load("movement_animations/Player_Sprite3_R.png"),pygame.image.load("movement_animations/Player_Sprite4_R.png"),
             pygame.image.load("movement_animations/Player_Sprite5_R.png"),pygame.image.load("movement_animations/Player_Sprite6_R.png"),
             pygame.image.load("movement_animations/Player_Sprite_R.png")]

# Run animation for the LEFT
run_ani_L = [pygame.image.load("movement_animations/Player_Sprite_L.png"), pygame.image.load("movement_animations/Player_Sprite2_L.png"),
             pygame.image.load("movement_animations/Player_Sprite3_L.png"),pygame.image.load("movement_animations/Player_Sprite4_L.png"),
             pygame.image.load("movement_animations/Player_Sprite5_L.png"),pygame.image.load("movement_animations/Player_Sprite6_L.png"),
             pygame.image.load("movement_animations/Player_Sprite_L.png")]

# Attack animation for the RIGHT
attack_ani_R = [pygame.image.load("movement_animations/Player_Sprite_R.png"), pygame.image.load("attack_animations/Player_Attack_R.png"),
                pygame.image.load("attack_animations/Player_Attack2_R.png"),pygame.image.load("attack_animations/Player_Attack2_R.png"),
                pygame.image.load("attack_animations/Player_Attack3_R.png"),pygame.image.load("attack_animations/Player_Attack3_R.png"),
                pygame.image.load("attack_animations/Player_Attack4_R.png"),pygame.image.load("attack_animations/Player_Attack4_R.png"),
                pygame.image.load("attack_animations/Player_Attack5_R.png"),pygame.image.load("attack_animations/Player_Attack5_R.png"),
                pygame.image.load("movement_animations/Player_Sprite_R.png")]

# Attack animation for the LEFT
attack_ani_L = [pygame.image.load("movement_animations/Player_Sprite_L.png"), pygame.image.load("attack_animations/Player_Attack_L.png"),
                pygame.image.load("attack_animations/Player_Attack2_L.png"),pygame.image.load("attack_animations/Player_Attack2_L.png"),
                pygame.image.load("attack_animations/Player_Attack3_L.png"),pygame.image.load("attack_animations/Player_Attack3_L.png"),
                pygame.image.load("attack_animations/Player_Attack4_L.png"),pygame.image.load("attack_animations/Player_Attack4_L.png"),
                pygame.image.load("attack_animations/Player_Attack5_L.png"),pygame.image.load("attack_animations/Player_Attack5_L.png"),
                pygame.image.load("movement_animations/Player_Sprite_L.png")]

# Animations for the Health Bar
health_ani = [pygame.image.load("heart/heart0.png"), pygame.image.load("heart/heart.png"),
              pygame.image.load("heart/heart2.png"), pygame.image.load("heart/heart3.png"),
              pygame.image.load("heart/heart4.png"), pygame.image.load("heart/heart5.png")]


class Background(pygame.sprite.Sprite):
      def __init__(self):
            super().__init__()
            self.bgimage = pygame.image.load("Background.png")
            self.rectBGimg = self.bgimage.get_rect()        
            self.bgY = 0
            self.bgX = 0

      def render(self):
            displaysurface.blit(self.bgimage, (self.bgX, self.bgY))      


class Ground(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("Ground.png")
        self.rect = self.image.get_rect(center = (350, 350))
        self.bgX1 = 0
        self.bgY1 = 285

    def render(self):
        displaysurface.blit(self.image, (self.bgX1, self.bgY1)) 


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("movement_animations/Player_Sprite_R.png")
        self.rect = self.image.get_rect()

        # Position and direction
        self.vx = 0
        self.pos = vec((340, 240))
        self.vel = vec(0,0)
        self.acc = vec(0,0)
        self.direction = "RIGHT"

        # Movement 
        self.jumping = False
        self.running = False
        self.move_frame = 0

        #Combat
        self.attacking = False
        self.cooldown = False
        self.immune = False
        self.special = False
        self.experiance = 0
        self.attack_frame = 0
        self.health = 5
        self.magic_cooldown = 1
        self.mana = 0


    def move(self):
          # Keep a constant acceleration of 0.5 in the downwards direction (gravity)
          self.acc = vec(0,0.5)

          # Will set running to False if the player has slowed down to a certain extent
          if abs(self.vel.x) > 0.3:
                self.running = True
          else:
                self.running = False

          # Returns the current key presses
          pressed_keys = pygame.key.get_pressed()

          # Accelerates the player in the direction of the key press
          if pressed_keys[K_LEFT]:
                self.acc.x = -ACC
          if pressed_keys[K_RIGHT]:
                self.acc.x = ACC 

          # Formulas to calculate velocity while accounting for friction
          self.acc.x += self.vel.x * FRIC
          self.vel += self.acc
          self.pos += self.vel + 0.5 * self.acc  # Updates Position with new values

          # This causes character warping from one point of the screen to the other
          if self.pos.x > WIDTH:
                self.pos.x = 0
          if self.pos.x < 0:
                self.pos.x = WIDTH
        
          self.rect.midbottom = self.pos  # Update rect with new pos            

    def gravity_check(self):
          hits = pygame.sprite.spritecollide(player ,ground_group, False)
          if self.vel.y > 0:
              if hits:
                  lowest = hits[0]
                  if self.pos.y < lowest.rect.bottom:
                      self.pos.y = lowest.rect.top + 1
                      self.vel.y = 0
                      self.jumping = False


    def update(self):
          # Return to base frame if at end of movement sequence 
          if self.move_frame > 6:
                self.move_frame = 0
                return

          # Move the character to the next frame if conditions are met 
          if self.jumping == False and self.running == True:  
                if self.vel.x > 0:
                      self.image = run_ani_R[self.move_frame]
                      self.direction = "RIGHT"
                else:
                      self.image = run_ani_L[self.move_frame]
                      self.direction = "LEFT"
                self.move_frame += 1

          # Returns to base frame if standing still and incorrect frame is showing
          if abs(self.vel.x) < 0.2 and self.move_frame != 0:
                self.move_frame = 0
                if self.direction == "RIGHT":
                      self.image = run_ani_R[self.move_frame]
                elif self.direction == "LEFT":
                      self.image = run_ani_L[self.move_frame]

    def attack(self):        
          # If attack frame has reached end of sequence, return to base frame      
          if self.attack_frame > 10:
                self.attack_frame = 0
                self.attacking = False

          # Check direction for correct animation to display  
          if self.direction == "RIGHT":
                 self.image = attack_ani_R[self.attack_frame]
          elif self.direction == "LEFT":
                 self.correction()
                 self.image = attack_ani_L[self.attack_frame] 

          # Update the current attack frame  
          self.attack_frame += 1
          

    def jump(self):
        self.rect.x += 1

        # Check to see if payer is in contact with the ground
        hits = pygame.sprite.spritecollide(self, ground_group, False)
        
        self.rect.x -= 1

        # If touching the ground, and not currently jumping, cause the player to jump.
        if hits and not self.jumping:
           self.jumping = True 
           self.vel.y = -12

    def correction(self):
          # Function is used to correct an error
          # with character position on left attack frames
          if self.attack_frame == 1:
                self.pos.x -= 20
          if self.attack_frame == 10:
                self.pos.x += 20
                
    def player_hit(self):
        if self.cooldown == False:      
            self.cooldown = True # Enable the cooldown
            pygame.time.set_timer(hit_cooldown, 1000) # Resets cooldown in 1 second

            self.health = self.health - 1
            health.image = health_ani[self.health]
            
            if self.health <= 0:
                self.kill()
                pygame.display.update()

      
class Enemy(pygame.sprite.Sprite):
      def __init__(self):
        super().__init__()
        self.image = pygame.image.load("Enemy.png")
        self.rect = self.image.get_rect()     
        self.pos = vec(0,0)
        self.vel = vec(0,0)

        self.direction = random.randint(0,1) # 0 for Right, 1 for Left
        self.vel.x = random.randint(2,6) / 2  # Randomised velocity of the generated enemy

        # Sets the intial position of the enemy
        if self.direction == 0:
            self.pos.x = 0
            self.pos.y = 235
        if self.direction == 1:
            self.pos.x = 700
            self.pos.y = 235


      def move(self):
        # Causes the enemy to change directions upon reaching the end of screen    
        if self.pos.x >= (WIDTH-20):
              self.direction = 1
        elif self.pos.x <= 0:
              self.direction = 0

        # Updates positon with new values     
        if self.direction == 0:
            self.pos.x += self.vel.x
        if self.direction == 1:
            self.pos.x -= self.vel.x
            
        self.rect.center = self.pos # Updates rect
               
      def update(self):
            # Checks for collision with the Player
            hits = pygame.sprite.spritecollide(self, Playergroup, False)
            #print("fff")

            # Activates upon either of the two expressions being true
            if hits and player.attacking == True:
                  print("Enemy Killed")
                  self.kill()

            # If collision has occured and player not attacking, call "hit" function            
            elif hits and player.attacking == False:
                  player.player_hit()
                  
      def render(self):
            # Displayed the enemy on screen
            displaysurface.blit(self.image, (self.pos.x, self.pos.y))


class Castle(pygame.sprite.Sprite):
      def __init__(self):
            super().__init__()
            self.hide = False
            self.image = pygame.image.load("castle.png")

      def update(self):
            if self.hide == False:
                  displaysurface.blit(self.image, (400, 80))


class EventHandler():
      def __init__(self):
            self.enemy_count = 0
            self.battle = False
            self.enemy_generation = pygame.USEREVENT + 1
            self.stage = 1

            self.stage_enemies = []
            for x in range(1, 21):
                  self.stage_enemies.append(int((x ** 2 / 2) + 1))
            
      def stage_handler(self):
            # Code for the Tkinter stage selection window
            self.root = Tk()
            self.root.geometry('200x170')
            
            button1 = Button(self.root, text = "Twilight Dungeon", width = 18, height = 2,
                            command = self.world1)
            button2 = Button(self.root, text = "Skyward Dungeon", width = 18, height = 2,
                            command = self.world2)
            button3 = Button(self.root, text = "Hell Dungeon", width = 18, height = 2,
                            command = self.world3)
             
            button1.place(x = 40, y = 15)
            button2.place(x = 40, y = 65)
            button3.place(x = 40, y = 115)
            
            self.root.mainloop()
      
      def world1(self):
            self.root.destroy()
            pygame.time.set_timer(self.enemy_generation, 2000)
            castle.hide = True
            self.battle = True

      def world2(self):
            self.battle = True
            

      def world3(self):
            self.battle = True
 
      def next_stage(self):  # Code for when the next stage is clicked            
            self.stage += 1
            print("Stage: "  + str(self.stage))
            self.enemy_count = 0
            pygame.time.set_timer(self.enemy_generation, 1500 - (50 * self.stage))      


class HealthBar(pygame.sprite.Sprite):
      def __init__(self):
            super().__init__()
            self.image = pygame.image.load("heart/heart5.png")

      def render(self):
            displaysurface.blit(self.image, (10,10))

Enemies = pygame.sprite.Group()

player = Player()
Playergroup = pygame.sprite.Group()
Playergroup.add(player)

background = Background()

ground = Ground()
ground_group = pygame.sprite.Group()
ground_group.add(ground)

castle = Castle()
handler = EventHandler()
health = HealthBar()

hit_cooldown = pygame.USEREVENT + 1

while True:
    player.gravity_check()
  
    for event in pygame.event.get():
        if event.type == hit_cooldown:
            player.cooldown = False
        # Will run when the close window button is clicked    
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == handler.enemy_generation:
            if handler.enemy_count < handler.stage_enemies[handler.stage - 1]:
                  #print(handler.enemy_count)
                  #print(handler.stage_enemies[handler.stage - 1])
                  enemy = Enemy()
                  Enemies.add(enemy)
                  handler.enemy_count += 1     
            
        # For events that occur upon clicking the mouse (left click) 
        if event.type == pygame.MOUSEBUTTONDOWN:
              pass


        # Event handling for a range of different key presses    
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:
                  if handler.battle == True and len(Enemies) == 0:
                        handler.next_stage() 
            if event.key == pygame.K_q and 450 < player.rect.x < 550:
                handler.stage_handler()
            if event.key == pygame.K_SPACE:
                player.jump()
            if event.key == pygame.K_RETURN:
                if player.attacking == False:
                    player.attack()
                    player.attacking = True      

    
    

    # Player related functions
    player.update()
    if player.attacking == True:
          player.attack() 
    player.move()                

    # Display and Background related functions         
    background.render()
    ground.render()

    # Rendering Sprites
    castle.update()
    if player.health > 0:
        displaysurface.blit(player.image, player.rect)
    health.render()

    for entity in Enemies:
          entity.update()
          entity.move()
          entity.render()
          
    

    pygame.display.update()      
    FPS_CLOCK.tick(FPS)

