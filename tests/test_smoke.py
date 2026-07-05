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


def test_glb_load_canonicalization_and_mask():
    """GLB deve carregar, ser reorientado (maior eixo -> X) e virar
    obstáculo real na grade do solver"""
    import io
    import trimesh
    from cfd.geometry import process_model_for_cfd

    # caixa alongada no eixo Y: simula modelo com orientação "errada"
    box = trimesh.creation.box(extents=(1.0, 4.0, 0.5))
    glb_bytes = trimesh.Scene(box).export(file_type='glb')
    buf = io.BytesIO(glb_bytes)
    buf.name = 'test.glb'

    processor, sim_points, mesh_info = process_model_for_cfd(
        buf, file_type='glb')

    # canonicalizado: X (fluxo) >= Y (largura) >= Z (altura), maior dim = 1
    ext = processor.mesh_3d.extents
    assert ext[0] >= ext[1] >= ext[2], f"orientação não canônica: {ext}"
    assert np.isclose(ext[0], 1.0), f"malha não normalizada: {ext}"

    # pontos de simulação dentro do domínio padrão (4 x 2)
    assert sim_points.shape[1] == 2
    assert sim_points[:, 0].min() >= 0 and sim_points[:, 0].max() <= 4.0
    assert sim_points[:, 1].min() >= 0 and sim_points[:, 1].max() <= 2.0

    # máscara de ocupação rasterizada na grade
    nx, ny = 60, 30
    mask = processor.build_occupancy_mask(nx, ny)
    assert mask.shape == (nx * ny,)
    assert mask.any(), "máscara vazia — geometria não foi rasterizada"

    # bordas do túnel nunca podem ser sólidas
    mask_2d = mask.reshape(ny, nx)
    assert not mask_2d[:, 0].any() and not mask_2d[:, -1].any()
    assert not mask_2d[0, :].any() and not mask_2d[-1, :].any()

    # solver deve usar a geometria real como obstáculo (no-slip)
    sim = CFDSimulation(domain_size=(4.0, 2.0), resolution=(nx, ny))
    sim.set_object_mask(mask)
    results = sim.solve_steady_state(max_iterations=3, tolerance=1e-8)

    assert np.array_equal(results['object_mask'], mask.astype(bool))
    assert np.allclose(results['u'][results['object_mask']], 0.0)
    assert np.allclose(results['v'][results['object_mask']], 0.0)
    for key in ('lift_coefficient', 'drag_force', 'lift_force', 'frontal_area'):
        assert key in results, f"chave ausente no resultado: {key}"
    assert np.isfinite(results['drag_coefficient'])


def test_stl_watertight_example():
    """O exemplo RB16B (watertight) deve carregar e gerar máscara coerente"""
    stl_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'examples', 'RB16B_FIXED!.stl')
    if not os.path.exists(stl_path):
        return  # exemplo não disponível neste checkout

    from cfd.geometry import process_model_for_cfd

    processor, sim_points, mesh_info = process_model_for_cfd(stl_path)
    assert mesh_info['is_watertight']

    ext = processor.mesh_3d.extents
    assert ext[0] >= ext[1] >= ext[2]

    mask = processor.build_occupancy_mask(120, 60)
    frac = mask.mean()
    # silhueta de um carro deve ocupar fração pequena mas não nula do domínio
    assert 0.005 < frac < 0.30, f"fração sólida suspeita: {frac:.3f}"


if __name__ == '__main__':
    test_solve_steady_state_small_mesh()
    print("test_solve_steady_state_small_mesh: OK")
    test_calculate_streamlines()
    print("test_calculate_streamlines: OK")
    test_glb_load_canonicalization_and_mask()
    print("test_glb_load_canonicalization_and_mask: OK")
    test_stl_watertight_example()
    print("test_stl_watertight_example: OK")
    print("\nSmoke test passou (OK)")
