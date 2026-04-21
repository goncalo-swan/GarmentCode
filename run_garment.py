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
from assets.garment_programs import collars


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
                ankle_clearance_pct=config.get('ankle_clearance_pct', 0.03),
            )
        # Post-process collar panels: rotate into fold positions.
        # Panels are serialized flat to avoid stitching vertex collapse;
        # rotation must be applied to the spec JSON before mesh generation.
        collars.apply_collar_fold_rotations(folder)

        # Optionally flip hood panels to hang behind the back.
        if config.get('hood_down', False):
            collars.apply_hood_down(folder)

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
