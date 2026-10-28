from manim import *
from models import run_models
from data import ExperimentResult

BLUE_COLOR = "#253DA8"
RED_COLOR = "#C02525"

def introdutory_animation(self,data):
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
    title= Text(
        "Regularização em MLP's",
        font_size=38,
        weight=BOLD
    ).to_edge(UP*0.6)
    
    subtitle = Text(
        "Impacto da regularização L2 na capacidade de generalização das MLPs",
        font_size=20,
        color=GRAY_B
    ).next_to(title,DOWN,buff=0.3)
    
    info = VGroup(
        Dot(color=BLUE_C),
        Text("Classe 0",font_size=20),
        Dot(color=RED_C),
        Text("Classe 1",font_size=20)
    ).arrange(RIGHT).next_to(dots,DOWN,buff=0.3)
    
    self.play(Write(title),FadeIn(subtitle))
    self.wait(1)
    self.play(FadeIn(dots), FadeIn(info))        
    self.wait(1)
    self.play(FadeOut(title),FadeOut(subtitle),FadeOut(dots),FadeOut(info))
    self.wait(1)

def train_explanation(self):
    explicacao = Text(
        "Vamos treinar duas MLP's com os mesmos dados, uma usando regularização e outra não",
        font_size=24
    ).to_edge(UP*0.6)
    self.play(Write(explicacao))
    self.wait(2)
    self.play(FadeOut(explicacao))
            
    status = Text(
        "Treinando as MLP's ...",
        font_size=24
    ).to_edge(UP*0.6)
    self.play(FadeIn(status))
    self.wait(2)
    self.play(FadeOut(status))

def scatterplot_animation(self,results:ExperimentResult):
    scatter1 = ImageMobject("media/imgs/scatterplot.png")
    scatter2 = ImageMobject("media/imgs/scatterplot2.png")
    
    scatter1.scale_to_fit_width(6.0)
    scatter2.scale_to_fit_width(6.0)
    
    scatter1.move_to(LEFT * 3.15 + DOWN)
    scatter2.move_to(RIGHT * 3.15 + DOWN)
    
    title= Text(
        "Comparando as fronteiras",
        font_size=32,
        weight=BOLD
    ).to_edge(UP*0.6)
    
    metrics_with_l2 = VGroup(
        Text(
            f"Acurácia Treino:{results.acc_train_with_l2:.1%}",
            font_size=20,
            color=GRAY_B
        ),
        Text(
            f"Acurácia Teste:{results.acc_test_with_l2:.1%}",
            font_size=20,
            color=GRAY_B
        ),
    ).arrange(DOWN, aligned_edge=LEFT)
    
    metrics_without_l2 = VGroup(
        Text(
            f"Acurácia Treino:{results.acc_train_without_l2:.1%}",
            font_size=20,
            color=GRAY_B
        ),
        Text(
            f"Acurácia Teste:{results.acc_test_without_l2:.1%}",
            font_size=20,
            color=GRAY_B
        ),
    ).arrange(DOWN, aligned_edge=LEFT)
        
    self.play(Write(title), FadeIn(scatter1),FadeIn(scatter2))
    self.play(
        scatter1.animate.scale(0.88).shift(UP * 0.6),
        scatter2.animate.scale(0.88).shift(UP * 0.6),
        run_time=0.7,
    )
    metrics_without_l2.next_to(scatter1,DOWN*0.6)
    metrics_with_l2.next_to(scatter2,DOWN*0.6)
    self.play(FadeIn(metrics_without_l2,metrics_with_l2))
    self.wait(2)
    self.play(FadeOut(metrics_with_l2,metrics_without_l2,scatter1,scatter2,title))

def weight_changes(self,results:ExperimentResult):
    title= Text(
        "O que mudou nos pesos da rede?",
        font_size=32,
        weight=BOLD
    ).to_edge(UP*0.6)
    
    hist = ImageMobject("media/imgs/comparacao_pesos.png")
    
    hist.scale_to_fit_width(6.0)
    hist.move_to(ORIGIN)
    auxiliar_text = Text(
        "A penalização incentiva a rede a utilizar pesos de menor magnitude, em vez de pesos elevados",
        font_size=18,
        
    ).next_to(hist,DOWN*0.6)
    self.play(Write(title),FadeIn(hist),FadeIn(auxiliar_text))
    self.wait(2)
    self.play(FadeOut(title,hist,auxiliar_text))
    self.wait(1.5)
    
    individual_weights = Text(
        "Comparação de Pesos individuais",
        font_size=32,
        weight=BOLD,
    ).to_edge(UP*0.6)
    
    positions = np.argsort(np.abs(results.weights_without_l2))[-10:][::-1]
    
    header = VGroup(
        Text("Peso",font_size=18,weight=BOLD),
        Text("Sem L2",weight=BOLD,font_size=18),
        Text("Com L2",weight=BOLD,font_size=18)
    )
    header.arrange(RIGHT, buff=0.85)

    lines = VGroup(header)

    for i in positions:
        line = VGroup(
            Text(f"w[{i}]", font_size=18),
            Text(f"{results.weights_without_l2[i]:.2f}", font_size=18),
            Text(f"{results.weights_with_l2[i]:.2f}", font_size=18),
        )

        # Alinha cada coluna à posição correspondente do cabeçalho
        for j, item in enumerate(line):
            item.move_to(header[j].get_center())
            item.set_y(0)  # será ajustado pelo posicionamento da linha

        lines.add(line)

    # Organiza as linhas verticalmente
    lines.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
    lines.move_to(ORIGIN + UP*0.2)
    
    self.play(Write(individual_weights),FadeIn(lines))
    
    self.wait(4)
    self.play(FadeOut(individual_weights),FadeOut(lines))
    

class Regularizacao(Scene):
    def construct(self):
        results,data = run_models()

        introdutory_animation(self,data)
        train_explanation(self)
        scatterplot_animation(self,results)
        weight_changes(self,results)
        