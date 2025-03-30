from config import *


def load_animations(scale_factor, sprite_sheets):
    animations = {}
    for action, (sprite_sheet, rows, cols) in sprite_sheets.items():
        frame_width = sprite_sheet.get_width() // cols
        frame_height = sprite_sheet.get_height() // rows
        frames = []
        for row in range(rows):
            for col in range(cols):
                frame = sprite_sheet.subsurface(
                    pygame.Rect(col * frame_width, row * frame_height, frame_width, frame_height))
                scaled_frame = pygame.transform.scale(frame, (frame_width * scale_factor, frame_height * scale_factor))
                frames.append(scaled_frame)

        animations[f"{action}_right"] = frames
        animations[f"{action}_left"] = [pygame.transform.flip(img, True, False) for img in frames]
    return animations
