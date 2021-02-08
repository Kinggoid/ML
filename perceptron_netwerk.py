from ML.Perceptron_laag import PerceptronLaag
from ML.perceptronen import Perceptron


class PerceptronNetwork:
    def __init__(self, lagen: list):
        self.lagen = lagen

    def feed_forward(self, inputs: list):

        for i in self.lagen:
            antwoorden_laag = []
            for j in i.perceptrons:
                antwoorden_laag.append(j.calc(inputs))
            inputs = antwoorden_laag
        return antwoorden_laag
