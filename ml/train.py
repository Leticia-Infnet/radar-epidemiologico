"""Treino dos modelos de tendência, executado apenas no CI (workflow `train.yml`).

Progressão prevista: baseline sazonal → Ridge com Fourier e lags → GLM Poisson/binomial
negativa ou SARIMAX. Validação por `TimeSeriesSplit`, nunca k-fold aleatório.
"""


def main() -> None:
    """Lê os agregados do Hub, treina, avalia e publica modelo e métricas."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
