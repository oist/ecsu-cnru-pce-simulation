"""
Implements 2D animation of experimental simulation.
"""
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import os
from dataclasses import dataclass
from ranges import Range, RangeSet
import tempfile
import numpy as np
import pygame
from tqdm import tqdm
import pygame.freetype  # Import the freetype module.
from pce.simulation import Simulation, export_data_trial_to_tsv, test_simulation
from pce.utils import modulo_radians, unit_vector, TWO_PI
from pce import params


pygame.init()        
pygame.freetype.init() 
pygame.key.set_repeat(100)

CANVAS_SIZE = 800
RENDER_STEP_NUM = True

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
RED = (190, 18, 27)
BLUE = (25, 82, 184)
GREEN = (0, 153, 0)
YELLOW = (228, 213, 29)
PINK = (255, 51, 255)
ORANGE = (255, 130, 0)

PRINT_MODE = True
BG_COLOR = WHITE  if PRINT_MODE else BLACK
FG_COLOR = BLACK  if PRINT_MODE else WHITE

agents_colors = [GREEN, BLUE, RED, PINK, ORANGE] 

GAME_FONT = pygame.freetype.Font(None, 24)

@dataclass
class Frame:

    sim: Simulation    
    canvas_size: int
    env_radius: float
    zoom_factor: float
    data_record: dict
    trial_idx: int
    
    def __post_init__(self):
        self.surface = pygame.Surface((self.canvas_size, self.canvas_size))
        
        self.canvas_center = np.full(2, self.canvas_size / 2)

        self.sim.prepare_trial(self.trial_idx)        
        
        self.agents_angle = self.data_record['agents_pos'][self.trial_idx] / self.env_radius
        self.shadows_angle = self.data_record['shadows_pos'][self.trial_idx] / self.env_radius
        self.objs_angle = self.data_record['objs_pos'][self.trial_idx] / self.env_radius
        self.agents_signal = self.data_record['signal'][self.trial_idx]        

        self.agent_radius = self.sim.agent_width/2
        self.ring_padding = self.sim.agent_width * 3/4

    def draw_agent(self, center, inside, ang_rotation, color, signal):
        sens_color = YELLOW if signal else color
        dim = self.scale(np.array([self.sim.agent_width, self.sim.agent_width]))
        mid = dim/2
        agent_surf = pygame.Surface(dim, pygame.SRCALPHA)
        rect = pygame.Rect(0,0,self.scale(self.sim.agent_width/3),self.scale(self.sim.agent_width))        
        agent_radius = self.scale(self.agent_radius)
        wheel_radius = self.scale(self.agent_radius/3)
        pygame.draw.circle(agent_surf, color, mid, radius=agent_radius, width=0) # agent
        pygame.draw.circle(agent_surf, FG_COLOR, mid, radius=agent_radius, width=1) # agent border
        pygame.draw.rect(agent_surf, sens_color, rect, border_radius=5) # sensor
        pygame.draw.rect(agent_surf, FG_COLOR, rect, border_radius=5, width=1) # sensor border
        pygame.draw.circle(agent_surf, color, (dim[0]-wheel_radius, wheel_radius), radius=wheel_radius, width=0) # wheel 1
        pygame.draw.circle(agent_surf, FG_COLOR, (dim[0]-wheel_radius, wheel_radius), radius=wheel_radius, width=1) # wheel 1 border
        pygame.draw.circle(agent_surf, color, (dim[0]-wheel_radius, dim[0]-wheel_radius), radius=wheel_radius, width=0) # wheel 2
        pygame.draw.circle(agent_surf, FG_COLOR, (dim[0]-wheel_radius, dim[0]-wheel_radius), radius=wheel_radius, width=1) # wheel 2 border
        ang_degree = np.degrees(-ang_rotation)
        if inside: 
            ang_degree += 180
        agent_surf = pygame.transform.rotate(agent_surf, ang_degree)
        blit_rect_size = np.array(agent_surf.get_size())
        center_scaled_transposed = self.scale_transpose(center)
        center_scaled_transposed -= blit_rect_size/2
        self.surface.blit(agent_surf, dest=center_scaled_transposed) #
        # pygame.draw.rect(self.surface, FG_COLOR, blit_rect, width=1)


    def scale_transpose(self, p):
        return self.zoom_factor * p + self.canvas_center        

    def scale(self, p):
        return self.zoom_factor * p
    
    def scale(self, p):
        return self.zoom_factor * p

    def draw_line(self, x1y1, theta, length, color, width=1):
        x2y2 = (
            int(x1y1[0] + length * np.cos(theta)),
            int(x1y1[1] + length * np.sin(theta))
        )
        pygame.draw.line(self.surface, color, x1y1, x2y2, width=width)

    def draw_circle(self, color, center, radius, width):
        # if width is 0 circle will be filled
        center = self.scale_transpose(center)
        radius = self.scale(radius)
        pygame.draw.circle(self.surface, color, center, radius, width=width)
    
    def final_tranform_main_surface(self):
        '''
        final transformations:
        - shift coordinates to conventional x=0, y=0 in bottom left corner
        - zoom...
        '''
        self.surface.blit(pygame.transform.flip(self.surface, False, True), dest=(0, 0))


    def run_step(self, s):
        # update values at current time steps

        # reset canvas
        self.surface.fill(BG_COLOR)

        signals = self.agents_signal[s]

        # draw environment
        self.draw_circle(FG_COLOR, np.zeros(2), self.env_radius, 4)        
            
        # draw objects
        for i, o_ang in enumerate(self.objs_angle[s]):
            ang_unit_vector = np.array([np.cos(o_ang), np.sin(o_ang)])
            obj_dst = self.env_radius
            if self.sim.objects_facing_agents:
                if self.sim.agents_reverse_motors[i]:
                    # object should be outside the circle (facing reversed agent facing out)
                    obj_dst += self.ring_padding
                else:
                    obj_dst -= self.ring_padding
            pos = obj_dst * ang_unit_vector            
            if self.sim.objects_facing_agents:
                # color = agents_colors[i%len(agents_colors)]
                self.draw_circle(GRAY, pos, self.agent_radius, 0)
            self.draw_circle(FG_COLOR, pos, self.agent_radius, 1)

        # draw shadows
        if not self.sim.no_shadow:
            for i, s_ang in enumerate(self.shadows_angle[s]):
                shadow_inside = self.sim.agents_reverse_motors[i]
                shadow_dst = self.env_radius 
                if shadow_inside:
                    shadow_dst -= self.ring_padding
                else:
                    shadow_dst += self.ring_padding
                shadow_pos = shadow_dst * np.array([np.cos(s_ang), np.sin(s_ang)])
                color = agents_colors[i%len(agents_colors)]
                self.draw_circle(color, shadow_pos, self.agent_radius, 3)

        # draw agents        
        for i, a_ang in enumerate(self.agents_angle[s]):
            agent_inside = self.sim.agents_reverse_motors[i]
            ang_unit_vector = np.array([np.cos(a_ang), np.sin(a_ang)])
            agent_dst = self.env_radius
            if agent_inside:
                agent_dst -= self.ring_padding
            else:
                agent_dst += self.ring_padding
            agent_pos = agent_dst * ang_unit_vector 
            color = agents_colors[i%len(agents_colors)]
            self.draw_agent(agent_pos, agent_inside, a_ang, color, signals[i])
            

        # final traformations
        self.final_tranform_main_surface()


@dataclass
class Visualization:

    sim: Simulation
    canvas_size: int = CANVAS_SIZE
    fps: float = 20    
    video_path: str = None

    def __post_init__(self):    
        self.video_mode = self.video_path is not None        
        
        self.env_radius = self.sim.env_length / TWO_PI
        env_width = int(np.ceil(self.env_radius * 5 / 4))
        self.zoom_factor = self.canvas_size / 2 / env_width
        
                
        if self.video_mode:
            self.video_tmp_dir = tempfile.mkdtemp(dir=os.path.dirname(self.video_path))
            self.surface = pygame.Surface((self.canvas_size, self.canvas_size))
        else:
            self.surface = pygame.display.set_mode((self.canvas_size, self.canvas_size))         

    def export_video(self):
        import ffmpeg
        import shutil
        (
            ffmpeg
            .input(f'{self.video_tmp_dir}/*.png', pattern_type='glob', framerate=self.fps)
            .output(self.video_path, pix_fmt='yuv420p')
            .overwrite_output()
            .run(quiet=True)
        )        
        shutil.rmtree(self.video_tmp_dir)

    def start(self, data_record, trial_idx):

        main_frame = Frame(
            sim=self.sim,
            canvas_size=self.canvas_size,
            env_radius=self.env_radius,
            zoom_factor=self.zoom_factor,
            data_record=data_record,
            trial_idx=trial_idx
        )
        
        step_text_pos = (self.canvas_size-120, 10) # if cp[0] else (10, 10)

        num_zeros = int(np.ceil(np.log10(self.sim.num_steps)))

        running = True
        pause = False

        clock = pygame.time.Clock()        

        pbar = tqdm(total=self.sim.num_steps)

        s = 0

        while running and s < self.sim.num_steps:

            if not self.video_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        ekey = event.key
                        if ekey == pygame.K_ESCAPE:
                            running = False
                        elif ekey == pygame.K_p:
                            pause = not pause
                        elif pause:
                            if ekey == pygame.K_LEFT and s>0:                                
                                s -= 1
                            elif ekey == pygame.K_RIGHT and s < self.sim.num_steps-1:
                                s += 1

                

            self.surface.fill(BG_COLOR)
            
            # render one step
            main_frame.run_step(s)

            self.surface.blit(main_frame.surface, dest=(0, 0))
            
            step = str(s+1).zfill(num_zeros)
            
            if RENDER_STEP_NUM:
                GAME_FONT.render_to(self.surface, step_text_pos, f"Step: {step}", FG_COLOR)            

            if self.video_mode:
                filepath = os.path.join(self.video_tmp_dir, f"{step}.png")
                pygame.image.save(self.surface, filepath)                
            else:                
                pygame.display.update()
                clock.tick(self.fps)

            if not pause:
                pbar.update()
                s += 1            

        if self.video_mode:
            self.export_video()
        
        pygame.quit()


def test_visual_sim(seed=1233442, trial_idx = 0):    
    sim, data_record = test_simulation(
        num_agents=2,
        num_neurons=2,
        num_steps=300,
        env_length=150,                
        shadow_delta=20,
        seed=seed,
        performance_function = 'SHANNON_ENTROPY',
        no_shadow = False
    )
    # export_data_trial_to_tsv(
    #     tsv_file='data/test/test.tsv',
    #     data_record=data_record,
    #     trial_idx=trial_idx
    # )    
    viz = Visualization(
        sim,
        fps=20,
        # video_path='test.mp4'
    )
    viz.start(data_record, trial_idx=trial_idx)

if __name__ == "__main__":
    test_visual_sim()
