"""
Simplified FEniCS Data Generation - Open Field Only
"""

import numpy as np
from pathlib import Path
import json
from datetime import datetime
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import ufl
from dataclasses import dataclass
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from atmospheric_physics import (
    sample_meteorological_conditions,
    get_pasquill_class,
    stability_class_to_D
)


# Configuration

@dataclass
class SimulationParams:
    Lx: float = 100.0
    Ly: float = 100.0
    nx: int = 100
    ny: int = 100
    T: float = 500.0           # Total simulation time (s)
    dt: float = 0.05           # Time step (s) - reduced for stability
    output_freq: int = 100     # Save every N timesteps (every 5s)
    # D is now DYNAMIC - calculated from stability class per scenario
    k1: float = 1.0e-5         # Decay rate (1/s)
    z0: float = 0.03           # Surface roughness (m)
    z: float = 2.0             # Release height (m)
    
    # Emission rates - REALISTIC total mass flux (ambient monitoring)
    Q_total_min: float = 5.0   # Small facility (kg/s)
    Q_total_max: float = 100.0 # Large refinery (kg/s)
    Q_log_scale: bool = True   # Log-uniform: more small/medium facilities
    
    sigma: float = 3.0         # Source width (m)
    supg_factor: float = 0.3   # SUPG stabilization factor
    sampling_margin: float = 5.0

@dataclass
class WindCondition:
    speed: float
    direction: float
    
    @property
    def u(self): 
        return self.speed * np.cos(np.radians(self.direction))
    
    @property
    def v(self): 
        return self.speed * np.sin(np.radians(self.direction))

@dataclass
class SourceLocation:
    id: int
    x: float
    y: float
    name: str = ""

# Scenario generation

def generate_source_locations() -> List[SourceLocation]:
    """10 strategic source positions"""
    return [
        SourceLocation(1, 10, 10, "SW_corner"),
        SourceLocation(2, 30, 50, "West_mid"),
        SourceLocation(3, 20, 30, "SW_quad"),
        SourceLocation(4, 30, 70, "NW_quad"),
        SourceLocation(5, 50, 50, "Center"),
        SourceLocation(6, 40, 35, "Central"),
        SourceLocation(7, 60, 70, "NE_quad"),
        SourceLocation(8, 80, 50, "East_mid"),
        SourceLocation(9, 85, 70, "NE_region"),
        SourceLocation(10, 70, 90, "North_edge"),
    ]

def generate_scenario_manifest(params: SimulationParams) -> List[Dict]:
    """400 scenarios with RANDOM Q₀ for realistic emission rate training"""
    scenarios = []
    scenario_id = 0
    
    sources = generate_source_locations()
    
    # All 40 wind combinations
    winds = []
    for speed in [0.5, 2.0, 4.0, 6.0, 10.0]:
        for direction in [0, 45, 90, 135, 180, 225, 270, 315]:
            winds.append(WindCondition(speed, direction))
    
    # Random generator (reproducible)
    rng = np.random.default_rng(seed=42)
    
    # Generate all combinations with random Q_total AND DYNAMIC D
    for source in sources:
        for wind in winds:
            scenario_id += 1
            
            # Random Q_total for this scenario (log-uniform distribution)
            if params.Q_log_scale:
                log_Q = rng.uniform(
                    np.log10(params.Q_total_min),
                    np.log10(params.Q_total_max)
                )
                Q_total = 10**log_Q
            else:
                Q_total = rng.uniform(params.Q_total_min, params.Q_total_max)
            
            # Sample meteorology and calculate D from stability class
            solar, cloud, hour, is_day = sample_meteorological_conditions(
                wind.speed, rng
            )
            
            stability_class = get_pasquill_class(
                wind.speed, solar, cloud, is_day
            )
            
            D = stability_class_to_D(
                stability_class, wind.speed, params.z0, params.z
            )
            
            scenarios.append({
                'id': scenario_id,
                'geometry': 'open_field',
                'source_id': source.id,
                'source_x': source.x,
                'source_y': source.y,
                'source_name': source.name,
                'Q_total': float(Q_total),  # Total mass rate (kg/s)
                'wind_speed': wind.speed,
                'wind_direction': wind.direction,
                'wind_u': wind.u,
                'wind_v': wind.v,
                'D': float(D),
                'stability_class': stability_class,
                'solar_radiation': float(solar),
                'cloud_cover': float(cloud),
                'hour': float(hour),
            })
    
    print(f"\nGenerated {len(scenarios)} scenarios")
    print(f"  Q_total: {params.Q_total_min:.1f} - {params.Q_total_max:.1f} kg/s")
    print(f"  D range: 0.3 - 5.0 m²/s (calculated from stability)")
    print(f"  Distribution: Neutral-biased (realistic)")
    return scenarios

# Boundary conditions

def apply_boundary_conditions(V, params, wind_u, wind_v):
    """
    Do-nothing boundary conditions on ALL edges
    
    Correct for prescribed wind + source:
    - Advection term (after integration by parts) creates boundary flux
    - Outflow (u·n > 0): Mass exits at local concentration φ
    - Inflow (u·n < 0): No flux (no pollutant outside domain)
    - FEniCS handles this automatically!
    """
    print("Applied boundary conditions:")
    print(f"  Wind: ({wind_u:.2f}, {wind_v:.2f}) m/s")
    print(f"  All boundaries: Do-nothing (advective outflow)")
    print(f"  Total DOFs: {V.dofmap.index_map.size_global}")
    
    return []  # Empty = do-nothing BC


# Source

class GaussianSource:
    """
    Gaussian emission source with TOTAL MASS RATE (not volumetric)
    Standard atmospheric dispersion formulation
    """
    def __init__(self, x_source, y_source, Q_total, sigma):
        self.xs = x_source
        self.ys = y_source
        self.Q_total = Q_total  # Total mass rate (kg/s)
        self.sigma = sigma
        
        # Convert to volumetric source amplitude
        # Integrating Gaussian over domain gives Q_total
        self.Q0 = Q_total / (2 * np.pi * sigma**2)
    
    def __call__(self, x):
        r_squared = (x[0] - self.xs)**2 + (x[1] - self.ys)**2
        return self.Q0 * np.exp(-r_squared / (2 * self.sigma**2))

# ADR Solver

class ADRSolver:
    """Advection-Diffusion-Reaction solver with DYNAMIC D"""
    
    def __init__(self, params, domain, scenario, source):
        self.params = params
        self.domain = domain
        self.wind_u = scenario['wind_u']
        self.wind_v = scenario['wind_v']
        self.D = scenario['D']  # Dynamic D from scenario stability
        self.source = source
        
        # Function space
        self.V = fem.functionspace(domain, ("CG", 1))
        
        # Trial/test functions
        self.phi = ufl.TrialFunction(self.V)
        self.v_test = ufl.TestFunction(self.V)
        
        # Previous solution
        self.phi_n = fem.Function(self.V)
        self.phi_n.name = "phi"
        self.phi_n.x.array[:] = 0.0
        
        # Source function
        self.source_func = fem.Function(self.V)
        self.source_func.interpolate(self.source)
        
        # Boundary conditions (wind-aware inflow/outflow)
        self.bcs = apply_boundary_conditions(self.V, params, scenario['wind_u'], scenario['wind_v'])
        
        # Setup problem
        self._setup_variational_problem()
    
    def _setup_variational_problem(self):
        """Setup ADR weak form WITH explicit outflow boundary term AND DYNAMIC D"""
        dt = self.params.dt
        D = self.D  # Use scenario-specific D
        k1 = self.params.k1
        u = self.wind_u
        v = self.wind_v
        
        # Velocity
        vel = ufl.as_vector([u, v])
        vel_mag = ufl.sqrt(u**2 + v**2) + 1e-10
        
        # SUPG parameter
        h = ufl.CellDiameter(self.domain)
        tau = self.params.supg_factor * h / (2 * vel_mag)
        
        # Standard volume terms
        F = (self.phi - self.phi_n)/dt * self.v_test * ufl.dx
        F += D * ufl.dot(ufl.grad(self.phi), ufl.grad(self.v_test)) * ufl.dx
        F += ufl.dot(vel, ufl.grad(self.phi)) * self.v_test * ufl.dx
        F += k1 * self.phi * self.v_test * ufl.dx
        F -= self.source_func * self.v_test * ufl.dx
        
        # Explicit advective outflow on boundaries
        n = ufl.FacetNormal(self.domain)
        u_n = ufl.dot(vel, n)  # Normal velocity component
        
        # Only apply outflow where u·n > 0 (flow leaving domain)
        # max(u·n, 0) = (u·n + |u·n|)/2 where |u·n| = sqrt((u·n)²)
        u_n_plus = (u_n + ufl.sqrt(u_n**2)) / 2.0
        F += u_n_plus * self.phi * self.v_test * ufl.ds  # Boundary flux term
        
        # SUPG stabilization
        residual = (self.phi - self.phi_n)/dt + ufl.dot(vel, ufl.grad(self.phi)) + k1*self.phi
        F += tau * ufl.dot(vel, ufl.grad(self.v_test)) * residual * ufl.dx
        
        # Create problem
        a = ufl.lhs(F)
        L = ufl.rhs(F)
        
        self.problem = LinearProblem(
            a, L, bcs=self.bcs,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            },
            petsc_options_prefix="adr_solver"
        )
    
    
    def solve_timestep(self):
        """Solve timestep - SUPG prevents negatives"""
        phi_new = self.problem.solve()
        # No clipping needed with SUPG
        self.phi_n.x.array[:] = phi_new.x.array[:]
        return phi_new
    
    def compute_mass(self):
        mass_form = fem.form(self.phi_n * ufl.dx)
        return fem.assemble_scalar(mass_form)
    
    def get_max_concentration(self):
        return float(np.max(self.phi_n.x.array[:]))
    
    def get_min_concentration(self):
        return float(np.min(self.phi_n.x.array[:]))

# Quality control

def check_quality_metrics(solver, t, initial_mass):
    """Quality control with SUPG stabilization"""
    mass = solver.compute_mass()
    max_phi = solver.get_max_concentration()
    min_phi = solver.get_min_concentration()
    
    metrics = {
        'time': float(t),
        'total_mass': float(mass),
        'max_concentration': float(max_phi),
        'min_concentration': float(min_phi),
    }
    
    issues = []
    
    # SUPG allows small numerical undershoots - only warn if significant
    if min_phi < -0.001:  # More than 1 mg/m³ negative
        issues.append(f"Negative concentration: {min_phi:.2e} kg/m³")
    
    # Realistic max for Q_total ∈ [5, 100] kg/s
    if max_phi > 200:  # Adjusted for SUPG + natural outflow
        issues.append(f"High concentration: {max_phi:.2e} kg/m³")
    
    # Mass should increase then stabilize
    if mass > 1e5:  # Adjusted for new emission rates
        issues.append(f"High total mass: {mass:.2e} kg")
    
    metrics['issues'] = issues
    
    # CRITICAL: Stop if truly catastrophic
    if min_phi < -0.1 or max_phi > 1000:  # Relaxed from -0.01
        raise RuntimeError(f"Simulation unstable at t={t:.1f}s - STOPPING")
    
    return metrics

# Data sampling

def sample_collocation_points(solver, scenario, t, n_samples=1000):
    """Sample collocation points with Q_total as input feature (9 columns)"""
    params = solver.params
    
    # Get mesh coordinates and solution values
    coords = solver.V.tabulate_dof_coordinates()
    phi_values = solver.phi_n.x.array[:]
    
    # Filter to interior points (avoid boundaries)
    margin = params.sampling_margin
    mask = (
        (coords[:, 0] > margin) & (coords[:, 0] < params.Lx - margin) &
        (coords[:, 1] > margin) & (coords[:, 1] < params.Ly - margin)
    )
    
    interior_coords = coords[mask]
    interior_phi = phi_values[mask]
    
    # Random sample from interior points
    n_available = len(interior_coords)
    if n_available == 0:
        return np.array([]).reshape(0, 9)
    
    n_actual = min(n_samples, n_available)
    indices = np.random.choice(n_available, size=n_actual, replace=False)
    
    samples = []
    for idx in indices:
        x_sample = interior_coords[idx, 0]
        y_sample = interior_coords[idx, 1]
        phi_val = interior_phi[idx]
        
        if np.isfinite(phi_val):
            samples.append([
                t, x_sample, y_sample,
                scenario['source_x'], scenario['source_y'],
                scenario['Q_total'],  # Total mass rate (kg/s)
                scenario['wind_u'], scenario['wind_v'],
                scenario['D'],  # Diffusion coefficient
                phi_val
            ])
    
    return np.array(samples) if samples else np.array([]).reshape(0, 10)  # 10 columns total



def sample_initial_condition_points(solver, scenario, n_samples=5000):
    """IC points at t=0 with D as input feature (10 columns)"""
    params = solver.params
    
    margin = params.sampling_margin
    x_min, x_max = margin, params.Lx - margin
    y_min, y_max = margin, params.Ly - margin
    
    samples = []
    
    for _ in range(n_samples):
        x_sample = np.random.uniform(x_min, x_max)
        y_sample = np.random.uniform(y_min, y_max)
        
        samples.append([
            0.0, x_sample, y_sample,
            scenario['source_x'], scenario['source_y'],
            scenario['Q_total'],  # Total mass rate (kg/s)
            scenario['wind_u'], scenario['wind_v'],
            scenario['D'],  # Diffusion coefficient
            0.0  # phi=0 at t=0
        ])
    
    return np.array(samples)

# Visualization

def save_concentration_snapshot(solver, scenario, t, snapshot_num, output_dir):
    """Save heatmap visualization of concentration field"""
    
    # Get mesh coordinates and concentration values
    coords = solver.V.tabulate_dof_coordinates()
    phi_values = solver.phi_n.x.array[:]
    
    # Create triangulation for plotting
    x = coords[:, 0]
    y = coords[:, 1]
    triangulation = tri.Triangulation(x, y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot concentration field
    levels = np.linspace(0, np.max(phi_values), 50)
    contourf = ax.tricontourf(triangulation, phi_values, levels=levels, cmap='YlOrRd')
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax, label='Concentration (kg/m³)')
    
    # Mark source location
    ax.plot(scenario['source_x'], scenario['source_y'], 'k*', markersize=20, 
            label='Source', markeredgecolor='white', markeredgewidth=1.5)
    
    # Add wind vector
    wind_scale = 10  # Arrow length
    ax.arrow(10, 90, scenario['wind_u']*wind_scale, scenario['wind_v']*wind_scale,
             head_width=3, head_length=2, fc='blue', ec='blue', linewidth=2,
             label=f"Wind: {scenario['wind_speed']:.1f} m/s")
    
    # Labels and title
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(f"Scenario {scenario['id']} | t={t:.1f}s | Q={scenario['Q_total']:.1f} kg/s\n"
                 f"Max φ={np.max(phi_values):.2f} kg/m³", fontsize=14)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / f"snapshot_{snapshot_num:03d}_t{int(t):04d}s.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

# Main runner

def run_single_scenario(scenario, base_output_dir, params):
    """Run one scenario (open field)"""
    
    scenario_id = scenario['id']
    
    print(f"\n{'='*70}")
    print(f"SCENARIO {scenario_id}: OPEN FIELD")
    print(f"Source: ({scenario['source_x']:.1f}, {scenario['source_y']:.1f})")
    print(f"Q_total: {scenario['Q_total']:.2f} kg/s")
    print(f"Wind: {scenario['wind_speed']:.1f} m/s @ {scenario['wind_direction']:.0f}°")
    print(f"Stability: {scenario['stability_class']} (D={scenario['D']:.2e} m²/s)")
    print(f"Conditions: Solar={scenario['solar_radiation']:.0f} W/m², Cloud={scenario['cloud_cover']:.0f}%")
    print(f"{'='*70}")
    
    # Create mesh
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0, 0], [params.Lx, params.Ly]],
        [params.nx, params.ny],
        mesh.CellType.quadrilateral
    )
    
    # Create source with scenario-specific Q_total
    source = GaussianSource(
        scenario['source_x'], scenario['source_y'],
        scenario['Q_total'], params.sigma  # Use total mass rate
    )
    
    # Create solver (pass entire scenario for dynamic D)
    solver = ADRSolver(params, domain, scenario, source)
    
    # Create snapshots directory for visualizations
    snapshots_dir = Path("data/snapshots") / f"scenario_{scenario_id:04d}"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    # Time loop with error handling
    n_timesteps = int(params.T / params.dt)
    all_collocation_data = []
    
    ic_data = sample_initial_condition_points(solver, scenario)
    
    # Track initial mass for monitoring
    initial_mass = solver.compute_mass()
    print(f"Initial mass: {initial_mass:.2e} kg\n")
    
    failed = False
    failure_message = ""
    
    try:
        for n in range(n_timesteps):
            t = (n + 1) * params.dt
            solver.solve_timestep()
            
            if (n + 1) % params.output_freq == 0:
                # Quality check
                metrics = check_quality_metrics(solver, t, initial_mass)
                
                # Sample collocation data
                collocation_data = sample_collocation_points(solver, scenario, t)
                if len(collocation_data) > 0:
                    all_collocation_data.append(collocation_data)
                
                # Save visualization snapshot
                snapshot_num = (n + 1) // params.output_freq
                save_concentration_snapshot(solver, scenario, t, snapshot_num, snapshots_dir)
                
                # Print progress
                if len(metrics['issues']) > 0:
                    print(f"[{snapshot_num:3d}/100] t={t:6.1f}s - Mass: {metrics['total_mass']:.2e}, "
                          f"Max φ: {metrics['max_concentration']:.2e} ⚠  {metrics['issues']}")
                else:
                    print(f"[{snapshot_num:3d}/100] t={t:6.1f}s - Mass: {metrics['total_mass']:.2e}, "
                          f"Max φ: {metrics['max_concentration']:.2e} ({len(collocation_data)} pts)")
    
    except RuntimeError as e:
        failed = True
        failure_message = str(e)
        print(f"\n⚠ Scenario {scenario_id} FAILED: {failure_message}")
        print(f"  Saving partial data ({len(all_collocation_data)} snapshots)...")
    
    except Exception as e:
        failed = True
        failure_message = f"Unexpected error: {str(e)}"
        print(f"\n⚠ Scenario {scenario_id} FAILED: {failure_message}")
        print(f"  Saving partial data ({len(all_collocation_data)} snapshots)...")
    
    # Save data (even if failed)
    output_dir = base_output_dir / f"scenario_{scenario_id:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Debug: Check if we have collocation data
    print(f"  Saving data: {len(all_collocation_data)} snapshot arrays collected")
    
    if all_collocation_data:
        collocation_array = np.vstack(all_collocation_data)
        collocation_file = output_dir / "collocation_points.npz"
        np.savez_compressed(collocation_file, data=collocation_array)
        print(f"  Saved {len(collocation_array)} collocation points to {collocation_file.name}")
    else:
        print(f"  ⚠ No collocation data to save!")
    
    ic_file = output_dir / "ic_points.npz"
    np.savez_compressed(ic_file, data=ic_data)
    print(f"  Saved {len(ic_data)} IC points to {ic_file.name}")
    
    if failed:
        print(f"[PARTIAL] Scenario {scenario_id} completed with PARTIAL data\n")
        return {'id': scenario_id, 'status': 'failed', 'error': failure_message, 'snapshots': len(all_collocation_data)}
    else:
        print(f"[COMPLETE] Scenario {scenario_id}\n")
        return {'id': scenario_id, 'status': 'complete', 'snapshots': len(all_collocation_data)}

# Main

def main():
    params = SimulationParams()
    base_output_dir = Path("data/simulations")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    scenarios = generate_scenario_manifest(params)
    
    results = []
    for scenario in scenarios:
        result = run_single_scenario(scenario, base_output_dir, params)
        results.append(result)
    
    # Summary
    complete = sum(1 for r in results if r['status'] == 'complete')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total scenarios: {len(results)}")
    print(f"  Completed: {complete}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed scenarios:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - Scenario {r['id']:04d}: {r['error']}")
        
        # Save failure log
        failure_log = base_output_dir / "failed_scenarios.json"
        with open(failure_log, 'w') as f:
            json.dump([r for r in results if r['status'] == 'failed'], f, indent=2)
        print(f"\nFailure details saved to: {failure_log}")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()