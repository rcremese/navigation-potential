import navigation_potential.environments as envs
from navigation_potential.visualizer import Visualizer
from navigation_potential.fields import APFScalarField, LaplacianScalarField, EikonalScalarField, VectorField
from navigation_potential.planners import PathPlanner

import matplotlib.pyplot as plt


# Extended main function to test different environments
def test_environments():
    """Test various environment types."""
    # Create environments
    environments = [
        ("Maze", envs.MazeEnvironment(size=100, corridor_width=5)),
        ("Narrow Passage", envs.NarrowPassageEnvironment(size=100, num_passages=3, passage_width=3)),
        ("Cluttered", envs.ClutteredEnvironment(size=100, num_obstacles=100, min_size=2, max_size=8)),
        ("Rooms", envs.RoomsEnvironment(size=100, door_width=5)),
        ("Multi-Path", envs.MultiPathEnvironment(size=100)),
        ("Random", envs.RandomGridEnvironment(size=100, obstacle_density=0.4))
    ]
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (name, env) in enumerate(environments):
        if i < len(axes):
            ax = axes[i]
            ax.imshow(env.grid, cmap='binary', origin='lower')
            ax.plot(env.start[1], env.start[0], 'go', markersize=10, label='Start')
            ax.plot(env.target[1], env.target[0], 'ro', markersize=10, label='Target')
            ax.set_title(f"{name} Environment")
            ax.legend()
            ax.set_aspect('equal')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return environments


    
def main():
    """Main execution function."""
    # Test different environments
    test_environments()
    
    # Create a specific environment for detailed testing
    env = envs.ClutteredEnvironment(size=100, seed=42)
    visualizer = Visualizer(env)
    
    # Create scalar fields
    print("Computing fields for Rooms environment...")
    apf_scalar = APFScalarField(env, k_att=0.1, k_rep=50.0, rho0=15.0)
    lap_scalar = LaplacianScalarField(env, alpha=1.0)
    eik_scalar = EikonalScalarField(env, sigma=10.0)
    
    # Compute scalar fields
    apf_scalar.compute()
    lap_scalar.compute()
    eik_scalar.compute()
    
    # Create vector fields
    apf_vector = VectorField(apf_scalar)
    lap_vector = VectorField(lap_scalar)
    eik_vector = VectorField(eik_scalar)
    
    # Compute vector fields
    apf_vector.compute(normalize=True)
    lap_vector.compute(normalize=True)
    eik_vector.compute(normalize=True)
    
    # Visualize scalar and vector fields
    visualizer.plot_scalar_and_vector_fields(apf_scalar, apf_vector, "APF Scalar and Vector Fields")
    visualizer.plot_scalar_and_vector_fields(lap_scalar, lap_vector, "Laplacian Scalar and Vector Fields")
    visualizer.plot_scalar_and_vector_fields(eik_scalar, eik_vector, "Eikonal Scalar and Vector Fields")
    
    # Plan paths
    print("Planning paths...")
    apf_planner = PathPlanner(env, apf_vector)
    lap_planner = PathPlanner(env, lap_vector)
    eik_planner = PathPlanner(env, eik_vector)
    
    apf_trajectory, apf_success = apf_planner.plan_path()
    lap_trajectory, lap_success = lap_planner.plan_path()
    eik_trajectory, eik_success = eik_planner.plan_path()
    
    # Compare trajectories
    visualizer.compare_trajectories(
        [apf_trajectory, lap_trajectory, eik_trajectory],
        labels=["APF", "Laplacian", "Eikonal"],
        success_flags=[apf_success, lap_success, eik_success],
        colors=['b', 'g', 'r']
    )
    
    plt.show()
    

if __name__ == "__main__":
    main()
