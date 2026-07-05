"""
Smoke test do solver CFD: roda uma simulação pequena de ponta a ponta
e valida a estrutura do resultado.

Execução direta (sem pytest):
    python tests/test_smoke.py

Ou via pytest:
    pytest tests/
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfd.simulation import CFDSimulation


def test_solve_steady_state_small_mesh():
    """Simulação em malha pequena deve rodar sem erros e retornar campos coerentes"""
    # Mesma resolução que o app usa em CPU (create_gpu_optimized_simulation)
    nx, ny = 30, 15
    sim = CFDSimulation(domain_size=(4.0, 2.0), resolution=(nx, ny))
    sim.inlet_velocity = 30.0

    results = sim.solve_steady_state(max_iterations=5, tolerance=1e-8)

    # Chaves esperadas pela camada de visualização
    for key in ('u', 'v', 'velocity_magnitude', 'pressure', 'x', 'y',
                'residuals', 'drag_coefficient', 'iterations',
                'object_mask', 'converged', 'final_residual'):
        assert key in results, f"chave ausente no resultado: {key}"

    # Contrato de formas: eixos com nx/ny pontos, campos achatados nx*ny
    assert results['x'].shape == (nx,)
    assert results['y'].shape == (ny,)
    assert results['u'].shape == (nx * ny,)
    assert results['velocity_magnitude'].shape == (nx * ny,)
    assert results['pressure'].shape == (nx * ny,)
    assert results['object_mask'].shape == (nx * ny,)

    # Campos devem ser finitos
    assert np.all(np.isfinite(results['u']))
    assert np.all(np.isfinite(results['v']))
    assert np.all(np.isfinite(results['pressure']))

    # A máscara do objeto deve conter o aerofólio (algumas células sólidas)
    assert results['object_mask'].any(), "máscara do objeto vazia"

    # Velocidade nula dentro do objeto (no-slip)
    assert np.allclose(results['u'][results['object_mask']], 0.0)
    assert np.allclose(results['v'][results['object_mask']], 0.0)

    # Iterações registradas
    assert results['iterations'] == len(results['residuals']) > 0


def test_calculate_streamlines():
    """Streamlines devem ser geradas a partir do campo resolvido"""
    sim = CFDSimulation(domain_size=(4.0, 2.0), resolution=(30, 15))
    sim.solve_steady_state(max_iterations=3, tolerance=1e-8)

    streamlines = sim.calculate_streamlines(n_streamlines=5)
    assert isinstance(streamlines, list)
    for line in streamlines:
        assert line.ndim == 2 and line.shape[1] == 2


if __name__ == '__main__':
    test_solve_steady_state_small_mesh()
    print("test_solve_steady_state_small_mesh: OK")
    test_calculate_streamlines()
    print("test_calculate_streamlines: OK")
    print("\nSmoke test passou ✔")
