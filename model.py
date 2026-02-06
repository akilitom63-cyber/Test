"""Modele d'apprentissage supervisé simple (régression linéaire)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class LinearRegressionGD:
    """Régression linéaire 1D entraînée par descente de gradient."""

    learning_rate: float = 0.01
    epochs: int = 1000
    weight: float = 0.0
    bias: float = 0.0

    def fit(self, x_values: Iterable[float], y_values: Iterable[float]) -> None:
        x_list = list(x_values)
        y_list = list(y_values)
        if len(x_list) != len(y_list):
            raise ValueError("x_values et y_values doivent avoir la même taille.")
        if not x_list:
            raise ValueError("Les données d'entraînement sont vides.")

        n = len(x_list)
        for _ in range(self.epochs):
            weight_grad = 0.0
            bias_grad = 0.0
            for x_val, y_val in zip(x_list, y_list):
                prediction = self.weight * x_val + self.bias
                error = prediction - y_val
                weight_grad += (2 / n) * error * x_val
                bias_grad += (2 / n) * error
            self.weight -= self.learning_rate * weight_grad
            self.bias -= self.learning_rate * bias_grad

    def predict(self, x_values: Iterable[float]) -> List[float]:
        return [self.weight * x_val + self.bias for x_val in x_values]


def generate_training_data() -> Tuple[List[float], List[float]]:
    """Génère un jeu de données synthétique y = 3x + 2 avec un léger bruit."""
    x_values = [x for x in range(1, 21)]
    y_values = [3 * x + 2 + ((x % 3) - 1) for x in x_values]
    return x_values, y_values


def main() -> None:
    x_train, y_train = generate_training_data()
    model = LinearRegressionGD(learning_rate=0.01, epochs=2000)
    model.fit(x_train, y_train)

    x_test = [0, 5, 10, 15, 20, 25]
    predictions = model.predict(x_test)

    print("Paramètres appris :")
    print(f"  poids = {model.weight:.4f}")
    print(f"  biais = {model.bias:.4f}")
    print("\nPrédictions :")
    for x_val, y_val in zip(x_test, predictions):
        print(f"  x={x_val:>2} -> y≈{y_val:.2f}")


if __name__ == "__main__":
    main()
