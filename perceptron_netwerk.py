class PerceptronNetwork:
    def __init__(self, lagen: list):
        self.lagen = lagen

    def network(self, inputs: list):
        """In deze definitie geven we de inputs van een perceptronlaag door naar de volgende laag tot we uiteindelijk
        de outputlaag bereiken."""

        alle_outputs = [inputs]
        for laag in self.lagen:  # Per laag
            alle_outputs.append(laag.feed_forward(alle_outputs[-1]))  # Pak de output van de laatste laag en calculeer de output van deze laag voor alle lagen
        return alle_outputs[-1]  # Pak de output van de laatste laag
