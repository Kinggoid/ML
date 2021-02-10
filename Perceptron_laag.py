class PerceptronLaag:
    def __init__(self, perceptrons: list):
        self.perceptrons = perceptrons  # In de lagen van de netwerken hoef ik alleen maar een lijst te hebben met alle
                                        # perceptrons die in deze laag voorkomen.

    def feed_forward(self, inputs):
        """Bereken van elke perceptron, in deze laag, zijn output en geef alle outputs terug."""
        return [perceptron.caclulate_output(inputs) for perceptron in self.perceptrons]
