from typing import List

class Perceptron:
    def __init__(self, weights: List[float], bias: float, name: str):
        self.weights = weights
        self.threshold = 0
        self.name = name
        self.bias = bias

    def calculate_output(self, inputs: List[int]):
        """In deze definitie kijken we bij een perceptron welke inputs hij krijgt en hoe de weights en de bias dit
        beïnvloeden. Vervolgens kijken we of het antwoord de threshold haalt."""
        inputs_met_weight = [self.weights[i] * inputs[i] for i in range(0, len(inputs))]  # Inputs * weights
        if sum(inputs_met_weight) - self.bias >= self.threshold:  # De som van die resultaten tellen we op met de bias en dan kijken we of het de threshold haalt.
            return 1
        return 0

    def __str__(self):
        """Informatie van de perceptron"""
        return 'Mijn naam is {} en ik heb {} input variabelen. Mijn threshold is {} en mijn bias is {}.'.format(self.name, str(len(self.weights)), str(self.threshold), str(self.bias))



