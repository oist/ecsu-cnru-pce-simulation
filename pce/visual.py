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

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (190, 18, 27)
BLUE = (25, 82, 184)
GREEN = (82, 240, 63)
YELLOW = (228, 213, 29)
PINK = (255, 51, 255)
ORANGE = (255, 130, 0)

agents_colors = [GREEN, BLUE, YELLOW, PINK, ORANGE] 

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
        self.surface.fill(BLACK)

        signals = self.agents_signal[s]

        # draw environment
        self.draw_circle(WHITE, np.zeros(2), self.env_radius, 4)        
            
        # draw objects
        for i, o_ang in enumerate(self.objs_angle[s]):
            pos = self.env_radius * np.array([np.cos(o_ang), np.sin(o_ang)])
            self.draw_circle(WHITE, pos, self.sim.agent_width/2, 0)

        # draw shadows
        if not self.sim.no_shadow:
            for i, s_ang in enumerate(self.shadows_angle[s]):
                pos = self.env_radius * np.array([np.cos(s_ang), np.sin(s_ang)])
                color = agents_colors[i%len(agents_colors)]
                self.draw_circle(color, pos, self.sim.agent_width/2, 3)

        # draw agents
        for i, a_ang in enumerate(self.agents_angle[s]):
            ang_unit_vector = np.array([np.cos(a_ang), np.sin(a_ang)])
            pos = self.env_radius * ang_unit_vector
            color = agents_colors[i%len(agents_colors)]
            self.draw_circle(color, pos, self.sim.agent_width/2, 0)
            # draw signal
            if signals[i]:
                x1y1 = self.scale_transpose(pos)
                # self.draw_line(x1y1, a_ang, 50, RED, 3)
                self.draw_circle(RED, pos, self.sim.agent_width/4, 0)
            # drow direction (head position)
            head_pos =  self.sim.agent_width/2 * ang_unit_vector
            if self.sim.agents_reverse_motors[i]:
                head_pos = - head_pos
            head_pos = pos - head_pos
            self.draw_circle(color, head_pos, self.sim.agent_width/4, 0)
            

        # final traformations
        self.final_tranform_main_surface()


@dataclass
class Visualization:

    sim: Simulation
    canvas_size: int = 800
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

                

            self.surface.fill(BLACK)
            
            # render one step
            main_frame.run_step(s)

            self.surface.blit(main_frame.surface, dest=(0, 0))
            
            step = str(s+1).zfill(num_zeros)
            
            GAME_FONT.render_to(self.surface, step_text_pos, f"Step: {step}", WHITE)            

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


def test_visual_sim(seed=663587459, trial_idx = 0):    
    sim, data_record = test_simulation(
        num_agents=1,
        num_neurons=1,
        num_steps=300,                
        seed=seed,
        performance_function = 'SHANNON_ENTROPY',
        no_shadow = True
    )
    export_data_trial_to_tsv(
        tsv_file='data/test/test.tsv',
        data_record=data_record,
        trial_idx=trial_idx
    )    
    viz = Visualization(
        sim,
        fps=5
        # video_path='video/test.mp4'å
    )
    viz.start(data_record, trial_idx=trial_idx)

if __name__ == "__main__":
    test_visual_sim()
