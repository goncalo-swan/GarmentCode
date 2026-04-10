"""
Generic Garment Generation & Simulation Pipeline.

Usage:
    python run_garment.py <config.yaml>
    python run_garment.py assets/garment_configs/hm_wide_high_waist_jeans.yaml
    python run_garment.py assets/garment_configs/hm_wide_high_waist_jeans.yaml --sizes 36,38,40
    python run_garment.py assets/garment_configs/hm_wide_high_waist_jeans.yaml --gpu 0
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

from pygarment.data_config import Properties


def _apply_collar_fold_rotations(folder):
    """Rotate collar fold panels in the spec JSON after serialization.

    Panels are serialized flat to avoid stitching vertex collapse cycles.
    This applies the 3D rotations needed for the fold to work in simulation.
    Only affects panels with 'collar_front_stand', 'collar_front_fall',
    'collar_back_stand', or 'collar_back_fall' in their name.
    """
    import json
    spec_files = list(folder.glob('*_specification.json'))
    if not spec_files:
        return
    spec_file = spec_files[0]
    with open(spec_file) as f:
        spec = json.load(f)

    panels = spec.get('pattern', {}).get('panels', {})

    # First pass: collect back_stand info so back_fall fold line can be aligned
    stand_info = {}
    for pname, panel in panels.items():
        if 'collar_back_stand' in pname:
            side = pname.split('_collar_back_stand')[0]
            fold_y = max(abs(v[1]) for v in panel['vertices'])
            stand_info[side] = {
                'translation': list(panel['translation']),
                'fold_y': fold_y,
            }

    modified = False
    for pname, panel in panels.items():
        rot = panel.get('rotation', [0, 0, 0])
        if 'collar_front_stand' in pname:
            rot[0] -= 90
            panel['translation'][2] = 3
        elif 'collar_front_fall' in pname:
            rot[0] += 60
            panel['translation'][2] = 15
        elif 'collar_back_stand' in pname:
            # Vertical against neck: no X rotation (local Y → 3D Y upward)
            panel['translation'][2] = -5
        elif 'collar_back_fall' in pname:
            side = pname.split('_collar_back_fall')[0]
            # Fold backward and downward from fold line
            rot[0] = -120
            if side in stand_info:
                si = stand_info[side]
                # Align fall's fold_line (local y=0) with stand's fold_line
                # (local y=fold_y, which with 0° rotation is at stand_Y + fold_y)
                panel['translation'][0] = si['translation'][0]
                panel['translation'][1] = si['translation'][1] + si['fold_y']
            panel['translation'][2] = -5
        else:
            continue
        panel['rotation'] = rot
        modified = True

    if modified:
        with open(spec_file, 'w') as f:
            json.dump(spec, f, indent=2)


def _deep_merge(base, overrides):
    """Recursively merge overrides into base dict."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def load_garment_config(config_path):
    """Load garment configuration from YAML file.

    Returns a dict with keys: name, garment_type, garment_prefix,
    body (yaml, name), sim_props (sim, render), sizes (all, run), production_data.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Ensure production_data keys are integers
    if 'production_data' in cfg:
        cfg['production_data'] = {int(k): v for k, v in cfg['production_data'].items()}
    if 'sizes' in cfg:
        cfg['sizes']['all'] = [int(s) for s in cfg['sizes'].get('all', [])]
        cfg['sizes']['run'] = [int(s) for s in cfg['sizes'].get('run', [])]
    return cfg


def run_pipeline(config):
    """Run the full garment pipeline from a loaded config dict."""
    garment_type = config.get('garment_type', 'pants')

    # Import shared utilities from pants module (simulate, mesh, normalize)
    from run_custom_pants import (
        simulate_pattern,
        save_combined_mesh,
        normalize_body_mesh,
    )

    # Import garment-type-specific functions
    if garment_type == 'shirt':
        from run_custom_tshirt import (
            map_production_to_design,
            generate_pattern,
        )
    else:
        from run_custom_pants import (
            map_production_to_design,
            generate_pattern,
            verify_measurements,
        )

    body_yaml = config['body']['yaml']
    body_name = config['body']['name']
    normalize_body_mesh(f'./assets/bodies/{body_name}.obj')
    sim_props_dict = config['sim_props']
    garment_prefix = config.get('garment_prefix', 'garment')
    # When body was overridden, append body name to prefix to avoid output collisions
    if config.get('_body_overridden'):
        body_short = body_name.replace('/', '_')
        garment_prefix = f'{garment_prefix}_{body_short}'
    sizes_to_run = config['sizes']['run']
    production_data = config['production_data']

    sys_props = Properties('./system.json')
    output_base = sys_props['output']

    print("=" * 60)
    print(f"  Garment Pipeline: {config.get('name', garment_prefix)}")
    print(f"  Type: {garment_type}")
    print(f"  Body: {body_name}")
    print(f"  Sizes: {sizes_to_run}")
    print(f"  GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 60)

    # Step 1: Generate patterns
    print("\n--- Step 1: Generating patterns ---")
    generated = []
    for size in sizes_to_run:
        prod = production_data[size]
        summary = ', '.join(f'{k}={v}' for k, v in prod.items())
        print(f'\nSize {size}: {summary}')

        if garment_type == 'shirt':
            design = map_production_to_design(prod, body_yaml)
        else:
            design = map_production_to_design(
                prod, body_yaml,
                elastic_waistband=config.get('elastic_waistband', False),
            )
        # Apply design overrides from config (e.g. upper type, collar, placket)
        if 'design_overrides' in config:
            _deep_merge(design, config['design_overrides'])
        if garment_type == 'shirt':
            folder, garment_name = generate_pattern(
                size, design, body_yaml, output_base,
                garment_prefix=garment_prefix,
            )
        else:
            folder, garment_name = generate_pattern(
                size, design, body_yaml, output_base,
                garment_prefix=garment_prefix,
                reposition_panels=config.get('reposition_panels', False),
                ankle_clearance_pct=config.get('ankle_clearance_pct', 0.05),
            )
        # Post-process collar panels: rotate into fold positions.
        # Panels are serialized flat to avoid stitching vertex collapse;
        # rotation must be applied to the spec JSON before mesh generation.
        _apply_collar_fold_rotations(folder)

        generated.append((folder, garment_name, size))

    # Step 2: Verify measurements (pants only)
    if garment_type != 'shirt':
        print("\n--- Step 2: Verifying pattern measurements ---")
        for folder, name, size in generated:
            spec_files = list(folder.glob('*_specification.json'))
            if spec_files:
                verify_measurements(spec_files[0], size, production_data[size],
                                    body_yaml=body_yaml,
                                    elastic_waistband=config.get('elastic_waistband', False))
            else:
                print(f'  No spec file found for size {size}')
    else:
        print("\n--- Step 2: Skipping measurement verification (not implemented for shirts) ---")

    # Step 3: Simulate
    print("\n--- Step 3: Running simulations ---")
    sim_results = []
    for folder, name, size in generated:
        print(f'\nSimulating size {size}...')
        try:
            sim_folder = simulate_pattern(
                folder, name, output_base,
                body_name=body_name, sim_props=sim_props_dict,
            )
            if sim_folder:
                sim_results.append((sim_folder, size))
        except Exception as e:
            print(f'  Simulation failed for size {size}: {e}')
            import traceback
            traceback.print_exc()

    # Step 4: Combined meshes
    print("\n--- Step 4: Creating combined body+garment meshes ---")
    for sim_folder, size in sim_results:
        print(f'\nCombining size {size}...')
        try:
            save_combined_mesh(sim_folder, body_name=body_name)
        except Exception as e:
            print(f'  Combined mesh failed for size {size}: {e}')
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print("\nGenerated patterns:")
    for folder, name, size in generated:
        print(f"  Size {size}: {folder}")
    print("\nSimulation results:")
    for sim_folder, size in sim_results:
        print(f"  Size {size}: {sim_folder}")
    print(f"\nAll output in: {output_base}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run garment generation & simulation pipeline')
    parser.add_argument('config', help='Path to garment config YAML')
    parser.add_argument('--sizes', help='Override sizes to run (comma-separated)', default=None)
    parser.add_argument('--body-yaml', help='Override body YAML path', default=None)
    parser.add_argument('--body-name', help='Override body name (matches .obj path under assets/bodies/)', default=None)
    parser.add_argument('--gpu', help='CUDA device ID', default='1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = load_garment_config(args.config)
    if args.sizes:
        config['sizes']['run'] = [int(s) for s in args.sizes.split(',')]
    if args.body_yaml or args.body_name:
        config['_body_overridden'] = True
    if args.body_yaml:
        config['body']['yaml'] = args.body_yaml
    if args.body_name:
        config['body']['name'] = args.body_name

    run_pipeline(config)
