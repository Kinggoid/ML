from typing import List

from ML.Perceptron.perceptronen import Perceptron


class PerceptronLaag:
    def __init__(self, perceptrons: List[Perceptron], name: str):
        self.perceptrons = perceptrons  # In de lagen van de netwerken hoef ik alleen maar een lijst te hebben met alle
                                        # perceptrons die in deze laag voorkomen.
        self.name = name

    def laag_forward(self, inputs: List[int]):
        """Bereken van elke perceptron, in deze laag, zijn output en geef alle outputs terug."""
        return [perceptron.calculate_output(inputs) for perceptron in self.perceptrons]

    def __str__(self):
        """Informatie van de perceptronlaag"""
        perceptronen = ''
        for i in range(0, len(self.perceptrons) - 1):
            perceptronen += str(self.perceptrons[i].name) + ', '
        perceptronen += self.perceptrons[-1].name

        return 'Ik ben de perceptronlaag {} en ik bevat de volgende perceptronen: {}'.format(self.name, perceptronen)

