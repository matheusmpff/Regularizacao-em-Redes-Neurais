from manim import *
from models import run_models

BLUE_COLOR = "#253DA8"
RED_COLOR = "#C02525"

class Regularizacao(Scene):
    def construct(self):
        results,data = run_models()
        
        dots = VGroup()
        
        for x,y in zip(data.X, data.y):
            dot = Dot(
                point=np.array(
                    [x[0],x[1],0]
                ),
                color= BLUE_C if y == 0 else RED_C ,
                radius=0.04
            )
            dots.add(dot)
        
        dots.move_to(ORIGIN)
        self.play(FadeIn(dots))
        self.wait(1.5)
        