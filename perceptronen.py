class Perceptron:
    def __init__(self, weights: list, treshold: float, name: str):
        self.weights = weights
        self.treshold = treshold
        self.name = name
        self.bias = 0

    def caclulate_output(self, inputs):
        """In deze definitie kijken we bij een perceptron welke inputs hij krijgt en hoe de weights en de bias dit
        beïnvloeden. Vervolgens kijken we of het antwoord de treshold haalt."""
        inputs_met_weight = [self.weights[i] * inputs[i] for i in range(0, len(inputs))]  # Inputs * weights
        if sum(inputs_met_weight) + self.bias >= self.treshold:  # De som van die resultaten tellen we op met de bias en dan kijken we of het de treshold haalt.
            return True
        return False

    def __str__(self):
        """Informatie van de perceptron"""
        return 'Mijn naam is {} en ik heb {} input variabelen. Mijn treshold is {} en mijn bias is {}.'.format(self.name, str(len(self.weights)), str(self.treshold), str(self.bias))



