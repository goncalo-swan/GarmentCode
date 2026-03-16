"""
Run pants pipeline for muses bodies (nina, carol, mayke, simone).
Each body runs on 3 sizes: one below, matching, one above.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
import run_custom_pants as rcp
from pygarment.data_config import Properties


BODIES = {
    'nina':   {'yaml': './assets/bodies/muses/nina.yaml',   'name': 'muses/nina',   'sizes': [44, 46, 48]},
    'carol':  {'yaml': './assets/bodies/muses/carol.yaml',  'name': 'muses/carol',  'sizes': [38, 40, 42]},
    'mayke':  {'yaml': './assets/bodies/muses/mayke.yaml',  'name': 'muses/mayke',  'sizes': [36, 38, 40]},
    'simone': {'yaml': './assets/bodies/muses/simone.yaml', 'name': 'muses/simone', 'sizes': [40, 42, 44]},
}


if __name__ == '__main__':
    sys_props = Properties('./system.json')
    output_base = sys_props['output']

    for person, cfg in BODIES.items():
        body_yaml = cfg['yaml']
        body_name = cfg['name']
        sizes = cfg['sizes']

        # Patch module globals so simulate_pattern / verify_measurements use the right body
        rcp.BODY_YAML = body_yaml
        rcp.BODY_NAME = body_name

        body_obj_path = f'./assets/bodies/{body_name}.obj'
        rcp.normalize_body_mesh(body_obj_path)

        print("\n" + "=" * 70)
        print(f"  {person.upper()} — body: {body_name}, sizes: {sizes}")
        print(f"  GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print("=" * 70)

        # Step 1: Generate patterns
        print(f"\n--- {person}: Generating patterns ---")
        generated = []
        for size in sizes:
            prod = rcp.PRODUCTION_DATA[size]
            print(f'\nSize {size}: Waist={prod["Waist"]}, Hip={prod["Low_Hip"]}, '
                  f'Inseam={prod["Inseam"]}, Ankle={prod["Ankle"]}')
            design = rcp.map_production_to_design(prod, body_yaml)
            print(f'  Design: width={design["pants"]["width"]["v"]:.3f}, '
                  f'length={design["pants"]["length"]["v"]:.3f}, '
                  f'flare={design["pants"]["flare"]["v"]:.3f}, '
                  f'thigh={design["pants"]["thigh"]["v"]:.3f}, '
                  f'knee={design["pants"]["knee"]["v"]:.3f}')
            folder, name = rcp.generate_pattern(size, design, body_yaml, output_base,
                                                   name_prefix=f'{person}_')
            generated.append((folder, name, size))

        # Step 2: Verify measurements
        print(f"\n--- {person}: Verifying pattern measurements ---")
        for folder, name, size in generated:
            spec_files = list(folder.glob('*_specification.json'))
            if spec_files:
                rcp.verify_measurements(spec_files[0], size, rcp.PRODUCTION_DATA[size])

        # Step 3: Simulate
        print(f"\n--- {person}: Running simulations ---")
        sim_results = []
        for folder, name, size in generated:
            print(f'\nSimulating {person} size {size}...')
            try:
                sim_folder = rcp.simulate_pattern(folder, name, output_base)
                if sim_folder:
                    sim_results.append((sim_folder, size))
            except Exception as e:
                print(f'  Simulation failed for {person} size {size}: {e}')
                import traceback
                traceback.print_exc()

        # Step 4: Combined meshes
        print(f"\n--- {person}: Creating combined body+garment meshes ---")
        for sim_folder, size in sim_results:
            print(f'\nCombining {person} size {size}...')
            try:
                rcp.save_combined_mesh(sim_folder, body_obj_path=body_obj_path)
            except Exception as e:
                print(f'  Combined mesh failed for {person} size {size}: {e}')
                import traceback
                traceback.print_exc()

        # Summary for this body
        print(f"\n--- {person}: Done ---")
        for folder, name, size in generated:
            print(f"  Size {size}: {folder}")
        for sim_folder, size in sim_results:
            print(f"  Sim {size}: {sim_folder}")

    print("\n" + "=" * 70)
    print("  All bodies complete!")
    print("=" * 70)
