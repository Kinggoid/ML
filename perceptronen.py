class Perceptron:
    def __init__(self, weights: list, bias: float, name: str):
        self.weights = weights
        self.bias = bias
        self.name = name

    def calc(self, inputs):
        x = [self.weights[i] * inputs[i] for i in range(0, len(inputs))]
        if sum(x) - self.bias >= 0:
            return True
        return False

    def __str__(self):
        return ('Mijn naam is ' + self.name + ', en ik heb ' + str(len(self.weights))
              + ' input variabelen. En mijn bias is ' + str(self.bias) + '.')



