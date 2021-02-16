import unittest

from ML.Perceptron.Perceptron_laag import PerceptronLaag
from ML.Perceptron.perceptron_netwerk import PerceptronNetwork
from ML.Perceptron.perceptronen import Perceptron


class test_perceptronen(unittest.TestCase):
    def test_AND_GATE(self):
        """test de AND gate."""
        antwoorden = []  # Kijk per input wat het antwoord zou zijn.
        for i in range(0, 2):
            for j in range(0, 2):
                AND = Perceptron([1, 1], 2, 'AND gate')
                antwoorden.append(AND.calculate_output([i, j]))

        self.assertEqual(antwoorden, [0,0,0,1])  # Kijk of de outputs goed zijn

    def test_NOT_GATE(self):
        antwoorden = []  # Kijk per input wat het antwoord zou zijn.
        for i in range(0, 2):
            NOT = Perceptron([-1], 0, 'NOT gate')
            antwoorden.append(NOT.calculate_output([i]))

        self.assertEqual(antwoorden, [1, 0])  # Kijk of de outputs goed zijn

    def test_OR_GATE(self):
        antwoorden = []  # Kijk per input wat het antwoord zou zijn.
        for i in range(0, 2):
            for j in range(0, 2):
                OR = Perceptron([1, 1], 1, 'OR gate')
                antwoorden.append(OR.calculate_output([i, j]))

        self.assertEqual(antwoorden, [0, 1, 1, 1])  # Kijk of de outputs goed zijn

    def test_NOR_GATE(self):
        antwoorden = []  # Kijk per input wat het antwoord zou zijn.
        for i in range(0, 2):
            for j in range(0, 2):
                for u in range(0, 2):
                    NOR = Perceptron([-1, -1, -1], 0, 'NOR gate')
                    antwoorden.append(NOR.calculate_output([i, j, u]))

        self.assertEqual(antwoorden, [1, 0, 0, 0, 0, 0, 0, 0])  # Kijk of de outputs goed zijn

    def test_3_WAY_GATE(self):
        antwoorden = []  # Kijk per input wat het antwoord zou zijn.
        for i in range(0, 2):
            for j in range(0, 2):
                for u in range(0, 2):
                    NOR = Perceptron([0.6, 0.3, 0.2], 0.4, 'NOR gate')
                    antwoorden.append(NOR.calculate_output([i, j, u]))

        self.assertEqual(antwoorden, [0, 0, 0, 1, 1, 1, 1, 1])  # Kijk of de outputs goed zijn

    def test_str_perceptron(self):
        percep = Perceptron([1, 1], 2, 'AND gate')
        self.assertEqual(percep.__str__(), 'Mijn naam is AND gate en ik heb 2 input variabelen. Mijn threshold is 0 en mijn bias is 2.')  # Kijk of de outputs goed zijn

    def test_str_perceptronlaag(self):
        x1_1 = Perceptron([1, 1], 2, 'x1_1')
        x1_2 = Perceptron([1, 1], 2, 'x1_2')

        laag_een = PerceptronLaag([x1_1, x1_2], 'L1')
        self.assertEqual(laag_een.__str__(), 'Ik ben de perceptronlaag L1 en ik bevat de volgende perceptronen: x1_1, x1_2')  # Kijk of de outputs goed zijn

    def test_str_perceptronnetwork(self):
        x1_1 = Perceptron([1, 1], 2, 'x1_1')
        x1_2 = Perceptron([1, 1], 2, 'x1_2')
        x2_1 = Perceptron([1, 1], 2, 'x2_1')
        x2_2 = Perceptron([1, 1], 2, 'x2_2')

        laag_een = PerceptronLaag([x1_1, x1_2], 'L1')
        laag_twee = PerceptronLaag([x2_1, x2_2], 'L2')

        network = PerceptronNetwork([laag_een, laag_twee])
        self.assertEqual(network.__str__(), 'In dit network zitten de volgende lagen: L1, L2')  # Kijk of de outputs goed zijn


class Test_Network(unittest.TestCase):
    def test_XOR(self):
        # Maak het network
        x1_1 = Perceptron([1, 1], 1, 'x1_1')
        x1_2 = Perceptron([-1, -1], -1.5, 'x1_2')

        laag_een = PerceptronLaag([x1_1, x1_2], 'L1')

        x2_1 = Perceptron([1, 1], 2, 'x2_1')

        laag_twee = PerceptronLaag([x2_1], 'L2')

        XOR = PerceptronNetwork([laag_een, laag_twee])

        antwoorden = []  # Kijk per input wat het antwoord zou zijn.
        for i in range(0, 2):
            for j in range(0, 2):
                antwoorden.append(XOR.feed_forward([i, j]))

        self.assertEqual(antwoorden, [[0], [1], [1], [0]])  # Kijk of de outputs goed zijn

    def test_half_adder(self):
        # Maak het network
        x1_1 = Perceptron([1, 1], 1, 'x1_1')
        x1_2 = Perceptron([-1, -1], -1.5, 'x1_1')
        x1_3 = Perceptron([1, 1], 2, 'x1_3')

        laag_een = PerceptronLaag([x1_1, x1_2, x1_3], 'L1')

        x2_1 = Perceptron([1, 1, 0], 2, 'x2_1')
        x2_2 = Perceptron([0, 0, 1], 1, 'x2_2')

        laag_twee = PerceptronLaag([x2_1, x2_2], 'L2')

        Half_adder = PerceptronNetwork([laag_een, laag_twee])

        antwoorden = []  # Kijk per input wat het antwoord zou zijn.
        for i in range(0, 2):
            for j in range(0, 2):
                antwoorden.append(Half_adder.feed_forward([i, j]))

        self.assertEqual(antwoorden, [[0, 0], [1, 0], [1, 0], [0, 1]])  # Kijk of de outputs goed zijn


if __name__ == '__main__':
    unittest.main()
