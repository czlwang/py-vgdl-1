import pygame
from pygame.math import Vector2

from vgdl.render import SpriteLibrary
from vgdl.ontology.constants import RIGHT, BLACK, WHITE, GOLD

import os
import numpy as np


class PygameRenderer:
    def __init__(self, game, block_size, render_sprites=True, visualize_diag=False):
        #import pdb; pdb.set_trace()
        self.game = game
        # In pixels
        self.block_size = block_size
        self.screen_dims = (game.width * self.block_size, game.height * self.block_size)
        self.render_sprites = render_sprites
        self.visualize_diag = visualize_diag
        if self.render_sprites:
            self.sprite_cache = SpriteLibrary.default()


    def init_screen(self, headless, title=None, visualize_diag=False):
        self.headless = headless
        # Right now display_dims and screen_dims are the same,
        # Likewise screen and display are interchangeable, for now.
        # I think it'd be good to allow resizing, just keep screen the same
        # and scale onto the resized display
        #self.display_dims = self.screen_dims

        w, h = self.screen_dims
        #TODO change hardcode
        diagram_height = h
        self.display_dims = (w, h)
        if self.visualize_diag:
            self.display_dims = (w, h + diagram_height)

        # The screen surface will be used for drawing on
        # It will be displayed on the `display` surface, possibly magnified
        # The background is currently solely used for clearing away sprites
        if headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            self.screen = pygame.display.set_mode(self.display_dims)
            #self.display = pygame.display.set_mode(self.display_dims, pygame.RESIZABLE, 32)
            self.display = pygame.display.set_mode((self.display_dims))
            self.background = pygame.Surface(self.screen_dims)
        else:
            self.screen = pygame.Surface(self.screen_dims)
            self.screen.fill((255, 255, 255))
            self.background = self.screen.copy()
            self.big_dims = self.display_dims
            self.big_screen = pygame.Surface(self.big_dims) 
            #self.display = pygame.display.set_mode(self.display_dims, pygame.RESIZABLE, 32)
            self.display = pygame.display.set_mode((self.display_dims))
            title_prefix = 'VGDL'
            title = title_prefix + ' ' + title if title else title_prefix
            if title:
                pygame.display.set_caption(title)


    def draw_all(self):
        for s in self.game.kill_list:
            self.clear_sprite(s)

        # This is for games where a sprite can disappear and leave black
        # background, mainly. Other games do not need clearing
        for s in self.game.sprite_registry.sprites():
            self.clear_sprite_last_if_necessary(s)

        for s in self.game.sprite_registry.sprites():
            self.draw_sprite(s)


    def update_display(self):
        # TODO this could be quicker for headless
        #pygame.transform.scale(self.screen, , self.display)
        screen_height = self.big_dims[1]
        import pdb; pdb.set_trace()
        if self.visualize_diag:
            screen_height = int(self.big_dims[1]/2)

        if self.visualize_diag:
            graphImg = pygame.image.load('test.gv.png')
            graphImg = pygame.transform.scale(graphImg, (self.big_dims[0], screen_height))
            g_rect = graphImg.get_rect()
            g_rect = g_rect.move((0, screen_height))
            self.big_screen.blit(graphImg, g_rect)

        temp_screen = pygame.transform.scale(self.screen, (self.big_dims[0], screen_height))
        s_rect = temp_screen.get_rect()
        self.big_screen.blit(temp_screen, s_rect)

        pygame.transform.scale(self.big_screen, self.display_dims, self.display)
        pygame.display.update()


    def calculate_render_rect(self, rect, shrinkfactor=0):
        displacement_factor = self.block_size / max(rect.size)
        sprite_rect = pygame.Rect(Vector2(rect.topleft) * displacement_factor,
                                  (self.block_size, self.block_size))
        if shrinkfactor != 0:
            sprite_rect = sprite_rect.inflate(*(-Vector2(sprite_rect.size) * shrinkfactor))

        return sprite_rect


    def draw_sprite(self, sprite):
        sprite_rect = self.calculate_render_rect(sprite.rect, sprite.shrinkfactor)

        #import pdb; pdb.set_trace()
        if self.render_sprites and sprite.img:
            # assert sprite.shrinkfactor == 0, 'TODO implement shrinking sprites'
            block_size = int((1-sprite.shrinkfactor) * self.block_size)
            img = self.sprite_cache.get_sprite_of_size(sprite.img, block_size)

            if hasattr(sprite, 'orientation'):
                # Assume by default images face right
                img_orientation = sprite.img_orient or RIGHT
                angle = img_orientation.angle_to(sprite.orientation)
                if abs(angle / 180) == 1:
                    # A flip will likely look nicer than a rotate
                    img = pygame.transform.flip(img, True, False)
                else:
                    img = pygame.transform.rotate(img, -angle)

            self.screen.blit(img, sprite_rect)
        else:
            self.screen.fill(sprite.color, sprite_rect)

        if sprite.resources:
            self.draw_resources(sprite, sprite_rect)


    def draw_resources(self, sprite, rect):
        """ Draw progress bars on the bottom third of the sprite """
        tot = len(sprite.resources)
        barheight = rect.height/3.5/tot
        offset = rect.top+2*rect.height/3.
        for r in sorted(sprite.resources.keys()):
            wiggle = rect.width/10.
            limit = self.game.domain.resources_limits.get(r, 1)
            prop = max(0,min(1,sprite.resources[r] / float(limit)))
            if prop != 0:
                filled = pygame.Rect(rect.left+wiggle/2, offset, prop*(rect.width-wiggle), barheight)
                rest   = pygame.Rect(rect.left+wiggle/2+prop*(rect.width-wiggle), offset, (1-prop)*(rect.width-wiggle), barheight)
                self.screen.fill(self.game.domain.resources_colors.get(r, GOLD), filled)
                self.screen.fill(BLACK, rest)
                offset += barheight


    def clear_sprite(self, sprite):
        # Shrunk objects clear non-shrunk rectangles, I think that's alright
        rect = self.calculate_render_rect(sprite.rect)
        self.screen.blit(self.background, rect, rect)


    def clear_sprite_last(self, sprite):
        rect = self.calculate_render_rect(sprite.lastrect)
        self.screen.blit(self.background, rect, rect)

    def clear_sprite_last_if_necessary(self, s):
        if s.rect != s.lastrect:
            self.clear_sprite_last(s)


    def clear(self):
        # TODO properly draw background
        # self.screen.blit()
        self.screen.fill((0,0,0))


    def force_display(self):
        self.clear()
        self.draw_all()
        self.update_display()


    #def _resize_display(self, target_size):
    #    # Doesn't actually work on quite a few systems
    #    # https://github.com/pygame/pygame/issues/201
    #    w_factor = target_size[0] / self.display_dims[0]
    #    h_factor = target_size[1] / self.display_dims[1]
    #    factor = min(w_factor, h_factor)

    #    self.display_dims = (int(self.display_dims[0] * factor),
    #                         int(self.display_dims[1] * factor))
    #    self.display = pygame.display.set_mode(self.display_dims, pygame.RESIZABLE, 32)


    def get_image(self):
        return np.flipud(np.rot90(pygame.surfarray.array3d(
            self.screen).astype(np.uint8)))


    def close(self):
        pygame.display.quit()
