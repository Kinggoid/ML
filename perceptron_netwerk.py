class PerceptronNetwork:
    def __init__(self, lagen: list):
        self.lagen = lagen

    def feed_forward(self, inputs: list):
        """In deze definitie geven we de inputs van een perceptronlaag door naar de volgende laag tot we uiteindelijk
        de outputlaag bereiken."""
        for laag in self.lagen:  # Per laag
            antwoorden_laag = []  # Ik maak hier de laag aan zodat ik alleen maar de outputs van de laatste laag opsla
            for perceptron in laag.perceptrons:  # Calculeer de outputs van elke perceptron
                antwoorden_laag.append(perceptron.caclulate_output(inputs))
            inputs = antwoorden_laag
        return antwoorden_laag
