"""O deploy no Community Cloud não pode levar torch: o wheel do PyPI puxa as bibliotecas CUDA."""

import re
from pathlib import Path

REQUIREMENTS = Path(__file__).parents[1] / "app" / "requirements.txt"


def test_requirements_sem_torch():
    pesados = re.findall(r"^(torch|transformers|nvidia-[\w-]+)==", REQUIREMENTS.read_text(), re.M)
    assert pesados == []
