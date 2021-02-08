import unittest

from ML.Perceptron_laag import PerceptronLaag
from ML.perceptron_netwerk import PerceptronNetwork
from ML.perceptronen import Perceptron


class test_perceptronen(unittest.TestCase):
    def test_AND_GATE(self):
        antwoorden = []
        for i in range(0, 2):
            for j in range(0, 2):
                AND = Perceptron([1, 1], 2, 'AND gate')
                antwoorden.append(AND.calc([i, j]))

        self.assertEqual(antwoorden, [0,0,0,1])


    def test_NOT_GATE(self):
        antwoorden = []
        for i in range(0, 2):
            NOT = Perceptron([-1], 0, 'NOT gate')
            antwoorden.append(NOT.calc([i]))

        self.assertEqual(antwoorden, [1, 0])


    def test_OR_GATE(self):
        antwoorden = []
        for i in range(0, 2):
            for j in range(0, 2):
                OR = Perceptron([1, 1], 1, 'OR gate')
                antwoorden.append(OR.calc([i, j]))

        self.assertEqual(antwoorden, [0, 1, 1, 1])


    def test_NOR_GATE(self):
        antwoorden = []
        for i in range(0, 2):
            for j in range(0, 2):
                for u in range(0, 2):
                    NOR = Perceptron([-1, -1, -1], 0, 'NOR gate')
                    antwoorden.append(NOR.calc([i, j, u]))

        self.assertEqual(antwoorden, [1, 0, 0, 0, 0, 0, 0, 0])


    def test_3_WAY_GATE(self):
        antwoorden = []
        for i in range(0, 2):
            for j in range(0, 2):
                for u in range(0, 2):
                    NOR = Perceptron([0.6, 0.3, 0.2], 0.4, 'NOR gate')
                    antwoorden.append(NOR.calc([i, j, u]))

        self.assertEqual(antwoorden, [0, 0, 0, 1, 1, 1, 1, 1])


    def test_str(self):
        percep = Perceptron([1, 1], 2, 'AND gate')
        self.assertEqual(percep.__str__(), 'Mijn naam is AND gate, en ik heb 2 input variabelen. En mijn bias is 2.')


class Test_Network(unittest.TestCase):
    def test_XOR(self):
        x1_1 = Perceptron([1, 1], 1, 'x1_1')
        x1_2 = Perceptron([-1, -1], -1.5, 'x1_1')

        laag_een = PerceptronLaag([x1_1, x1_2])

        x2_1 = Perceptron([1, 1], 2, 'x2_1')

        laag_twee = PerceptronLaag([x2_1])

        XOR = PerceptronNetwork([laag_een, laag_twee])

        antwoorden = []
        for i in range(0, 2):
            for j in range(0, 2):
                antwoorden.append(XOR.feed_forward([i, j]))

        self.assertEqual(antwoorden, [[0], [1], [1], [0]])


    def test_half_adder(self):
        x1_1 = Perceptron([1, 1], 1, 'x1_1')
        x1_2 = Perceptron([-1, -1], -1.5, 'x1_1')
        x1_3 = Perceptron([1, 1], 2, 'x1_3')

        laag_een = PerceptronLaag([x1_1, x1_2, x1_3])

        x2_1 = Perceptron([1, 1, 0], 2, 'x2_1')
        x2_2 = Perceptron([0, 0, 1], 1, 'x2_2')

        laag_twee = PerceptronLaag([x2_1, x2_2])

        Half_adder = PerceptronNetwork([laag_een, laag_twee])

        antwoorden = []
        for i in range(0, 2):
            for j in range(0, 2):
                antwoorden.append(Half_adder.feed_forward([i, j]))

        self.assertEqual(antwoorden, [[0, 0], [1, 0], [1, 0], [0, 1]])


if __name__ == '__main__':
    unittest.main()