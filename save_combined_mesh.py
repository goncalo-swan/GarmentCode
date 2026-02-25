"""
Save combined body + garment mesh after simulation.

Loads the simulated garment mesh and the body mesh, combines them into
a single OBJ file with the body and garment as separate groups.

Usage:
    # After running test_garment_sim.py, point to the output folder:
    python save_combined_mesh.py --folder Logs/shirt_mean_260218-11-53-16

    # Or specify paths directly:
    python save_combined_mesh.py \
        --garment Logs/shirt_mean_.../shirt_mean_sim.obj \
        --body assets/bodies/mean_all.obj \
        --output Logs/shirt_mean_.../combined.obj
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
import yaml


def load_body_mesh(body_obj_path):
    """Load body mesh and return trimesh object."""
    body = trimesh.load(str(body_obj_path), process=False)
    return body


def load_garment_mesh(garment_obj_path):
    """Load simulated garment mesh and return trimesh object."""
    garment = trimesh.load(str(garment_obj_path), process=False)
    return garment


def save_combined_obj(garment_mesh, body_mesh, output_path):
    """Save body + garment as a single OBJ with named groups.

    The body vertices come first, then garment vertices.
    Face indices for the garment are offset accordingly.
    """
    body_v = np.array(body_mesh.vertices)
    body_f = np.array(body_mesh.faces)
    garm_v = np.array(garment_mesh.vertices)
    garm_f = np.array(garment_mesh.faces)

    # Offset garment faces by number of body vertices
    garm_f_offset = garm_f + len(body_v)

    with open(output_path, 'w') as f:
        f.write(f"# Combined body + garment mesh\n")
        f.write(f"# Body vertices: {len(body_v)}, Garment vertices: {len(garm_v)}\n\n")

        # Write all vertices (body first, then garment)
        for v in body_v:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for v in garm_v:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Body faces (group: body)
        f.write("\ng body\n")
        for face in body_f:
            # OBJ uses 1-indexed faces
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        # Garment faces (group: garment)
        f.write("\ng garment\n")
        for face in garm_f_offset:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"Combined mesh saved to {output_path}")
    print(f"  Body:    {len(body_v)} vertices, {len(body_f)} faces")
    print(f"  Garment: {len(garm_v)} vertices, {len(garm_f)} faces")
    print(f"  Total:   {len(body_v)+len(garm_v)} vertices, {len(body_f)+len(garm_f)} faces")


def save_combined_glb(garment_mesh, body_mesh, output_path):
    """Save as GLB (binary glTF) with separate meshes for body and garment."""
    # Color the body dark grey
    body_mesh.visual = trimesh.visual.ColorVisuals(
        mesh=body_mesh,
        face_colors=np.tile([80, 80, 80, 255], (len(body_mesh.faces), 1))
    )
    # Color the garment white
    garment_mesh.visual = trimesh.visual.ColorVisuals(
        mesh=garment_mesh,
        face_colors=np.tile([220, 220, 220, 255], (len(garment_mesh.faces), 1))
    )

    scene = trimesh.Scene()
    scene.add_geometry(body_mesh, node_name='body')
    scene.add_geometry(garment_mesh, node_name='garment')
    scene.export(str(output_path))
    print(f"Combined GLB saved to {output_path}")


def find_body_obj_from_folder(sim_folder):
    """Determine the body OBJ path from simulation output folder."""
    sim_folder = Path(sim_folder)

    # Look for body_measurements.yaml in the folder
    body_yaml_files = list(sim_folder.glob('*body_measurements.yaml'))
    if body_yaml_files:
        with open(body_yaml_files[0], 'r') as f:
            body_data = yaml.safe_load(f)
        # Check if there's a body_sample field
        if 'body' in body_data and 'body_sample' in body_data['body']:
            body_name = body_data['body']['body_sample']
        else:
            # Default body name
            body_name = 'mean_all'
    else:
        body_name = 'mean_all'

    # Standard body location
    body_obj = Path('./assets/bodies') / f'{body_name}.obj'
    if body_obj.exists():
        return body_obj

    # Fallback: try all OBJs in bodies folder
    bodies_dir = Path('./assets/bodies')
    obj_files = list(bodies_dir.glob('*.obj'))
    if obj_files:
        print(f"Warning: Expected {body_obj}, using {obj_files[0]}")
        return obj_files[0]

    raise FileNotFoundError(f"No body OBJ found. Tried {body_obj}")


def main():
    parser = argparse.ArgumentParser(
        description='Save combined body + garment mesh after simulation'
    )
    parser.add_argument(
        '--folder', '-f',
        help='Simulation output folder (auto-discovers garment and body)',
        type=str, default=None
    )
    parser.add_argument(
        '--garment', '-g',
        help='Path to simulated garment OBJ (*_sim.obj)',
        type=str, default=None
    )
    parser.add_argument(
        '--body', '-b',
        help='Path to body OBJ',
        type=str, default=None
    )
    parser.add_argument(
        '--output', '-o',
        help='Output path for combined mesh (default: <folder>/combined.obj)',
        type=str, default=None
    )
    parser.add_argument(
        '--glb', action='store_true',
        help='Also save as GLB format (viewable in most 3D viewers)'
    )
    args = parser.parse_args()

    # Resolve paths
    if args.folder:
        folder = Path(args.folder)
        sim_files = list(folder.glob('*_sim.obj'))
        if not sim_files:
            raise FileNotFoundError(f"No *_sim.obj found in {folder}")
        garment_path = sim_files[0]
        body_path = args.body or find_body_obj_from_folder(folder)
        output_path = args.output or str(folder / 'combined.obj')
    elif args.garment:
        garment_path = Path(args.garment)
        body_path = Path(args.body) if args.body else find_body_obj_from_folder(garment_path.parent)
        output_path = args.output or str(garment_path.parent / 'combined.obj')
    else:
        parser.error("Provide either --folder or --garment")

    # Load meshes
    print(f"Loading garment: {garment_path}")
    garment = load_garment_mesh(garment_path)

    print(f"Loading body: {body_path}")
    body = load_body_mesh(body_path)

    # The body OBJ is in meters, garment is in cm.
    # GarmentCode's simulation uses cm internally (b_scale=100 converts m→cm).
    # The sim.obj is already in cm.
    # Body OBJ from assets/bodies/ is in meters.
    # Need to convert body to cm to match.
    body_in_meters = body.vertices.max() < 3.0  # heuristic: body < 3m
    if body_in_meters:
        print("  Body appears to be in meters, converting to cm to match garment")
        body.vertices = body.vertices * 100.0

    # The simulation also shifts the body up if min_y < 0
    # Replicate that shift for the body
    min_y = body.vertices[:, 1].min()
    if min_y < 0:
        shift = abs(min_y)
        print(f"  Applying y-shift of {shift:.2f} cm to match simulation")
        body.vertices[:, 1] += shift

    # Save combined
    save_combined_obj(garment, body, output_path)

    if args.glb:
        glb_path = Path(output_path).with_suffix('.glb')
        save_combined_glb(garment, body, glb_path)


if __name__ == '__main__':
    main()
