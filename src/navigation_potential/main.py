# import navigation_potential.environments as envs
# from navigation_potential.visualizer import Visualizer
# from navigation_potential.fields import APFScalarField, LaplacianScalarField, EikonalScalarField, VectorField
# from navigation_potential.planners import PathPlanner

# import matplotlib.pyplot as plt


# # Extended main function to test different environments
# def test_environments():
#     """Test various environment types."""
#     # Create environments
#     environments = [
#         ("Maze", envs.MazeEnvironment(size=100, corridor_width=5)),
#         ("Narrow Passage", envs.NarrowPassageEnvironment(size=100, num_passages=3, passage_width=3)),
#         ("Cluttered", envs.ClutteredEnvironment(size=100, num_obstacles=100, min_size=2, max_size=8)),
#         ("Rooms", envs.RoomsEnvironment(size=100, door_width=5)),
#         ("Multi-Path", envs.MultiPathEnvironment(size=100)),
#         ("Random", envs.RandomGridEnvironment(size=100, obstacle_density=0.4))
#     ]

#     # Create visualizations
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#     axes = axes.flatten()

#     for i, (name, env) in enumerate(environments):
#         if i < len(axes):
#             ax = axes[i]
#             ax.imshow(env.grid, cmap='binary', origin='lower')
#             ax.plot(env.start[1], env.start[0], 'go', markersize=10, label='Start')
#             ax.plot(env.target[1], env.target[0], 'ro', markersize=10, label='Target')
#             ax.set_title(f"{name} Environment")
#             ax.legend()
#             ax.set_aspect('equal')
#             ax.axis('off')

#     plt.tight_layout()
#     plt.show()

#     return environments



# def main():
#     """Main execution function."""
#     # Test different environments
#     test_environments()

#     # Create a specific environment for detailed testing
#     env = envs.ClutteredEnvironment(size=100, seed=42)
#     visualizer = Visualizer(env)

#     # Create scalar fields
#     print("Computing fields for Rooms environment...")
#     apf_scalar = APFScalarField(env, k_att=0.1, k_rep=50.0, rho0=15.0)
#     lap_scalar = LaplacianScalarField(env, alpha=1.0)
#     eik_scalar = EikonalScalarField(env, sigma=10.0)

#     # Compute scalar fields
#     apf_scalar.compute()
#     lap_scalar.compute()
#     eik_scalar.compute()

#     # Create vector fields
#     apf_vector = VectorField(apf_scalar)
#     lap_vector = VectorField(lap_scalar)
#     eik_vector = VectorField(eik_scalar)

#     # Compute vector fields
#     apf_vector.compute(normalize=True)
#     lap_vector.compute(normalize=True)
#     eik_vector.compute(normalize=True)

#     # Visualize scalar and vector fields
#     visualizer.plot_scalar_and_vector_fields(apf_scalar, apf_vector, "APF Scalar and Vector Fields")
#     visualizer.plot_scalar_and_vector_fields(lap_scalar, lap_vector, "Laplacian Scalar and Vector Fields")
#     visualizer.plot_scalar_and_vector_fields(eik_scalar, eik_vector, "Eikonal Scalar and Vector Fields")

#     # Plan paths
#     print("Planning paths...")
#     apf_planner = PathPlanner(env, apf_vector)
#     lap_planner = PathPlanner(env, lap_vector)
#     eik_planner = PathPlanner(env, eik_vector)

#     apf_trajectory, apf_success = apf_planner.plan_path()
#     lap_trajectory, lap_success = lap_planner.plan_path()
#     eik_trajectory, eik_success = eik_planner.plan_path()

#     # Compare trajectories
#     visualizer.compare_trajectories(
#         [apf_trajectory, lap_trajectory, eik_trajectory],
#         labels=["APF", "Laplacian", "Eikonal"],
#         success_flags=[apf_success, lap_success, eik_success],
#         colors=['b', 'g', 'r']
#     )

#     plt.show()


# if __name__ == "__main__":
#     main()
# main.py
import matplotlib.pyplot as plt
import navigation_potential.environments as envs
from navigation_potential.visualizer import Visualizer
from navigation_potential.fields import ScalarField, VectorField, create_vector_field_from_scalar
from navigation_potential.solvers import APFSolver, LaplaceSolver, EikonalSolver
from navigation_potential.planners import PathPlanner

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
    env = envs.RoomsEnvironment(size=100, door_width=5, seed=10)
    visualizer = Visualizer(env)

    # Create scalar field solvers
    print("Initializing solvers...")
    apf_solver = APFSolver(env)  # Lower resolution for APF
    lap_solver = LaplaceSolver(env)  # Native resolution
    eik_solver = EikonalSolver(env)  # Lower resolution for Eikonal

    # Compute scalar fields
    print("Computing fields for Rooms environment...")
    apf_field = apf_solver.solve(k_att=0.1, k_rep=50.0, rho0=15.0)
    lap_field = lap_solver.solve(alpha=1.0)
    eik_field = eik_solver.solve(sigma=10.0)

    # Print field resolutions
    print(f"APF field resolution: {apf_field.shape}")
    print(f"Laplacian field resolution: {lap_field.shape}")
    print(f"Eikonal field resolution: {eik_field.shape}")

    # Compute vector fields
    print("Computing vector fields...")
    apf_vector = create_vector_field_from_scalar(apf_field, normalize=True)
    lap_vector = create_vector_field_from_scalar(lap_field, normalize=True)
    eik_vector = create_vector_field_from_scalar(eik_field, normalize=True)

    # Visualize scalar and vector fields
    print("Visualizing fields...")
    visualizer.plot_scalar_and_vector_field(apf_field, apf_vector, "APF Scalar and Vector Fields")
    visualizer.plot_scalar_and_vector_field(lap_field, lap_vector, "Laplacian Scalar and Vector Fields")
    visualizer.plot_scalar_and_vector_field(eik_field, eik_vector, "Eikonal Scalar and Vector Fields")

    # Test field resampling
    print("Testing field resampling...")
    eik_field_upsampled = eik_field.resample(2)  # Double resolution
    eik_field_downsampled = eik_field.resample(-2)  # Half resolution

    eik_vector_upsampled = eik_vector.resample(2)  # Double resolution
    eik_vector_downsampled = eik_vector.resample(-2)  # Half resolution

    # Visualize resampled fields
    visualizer.compare_scalar_fields(
        [eik_field, eik_field_upsampled, eik_field_downsampled],
        ["Original", "Upsampled (2x)", "Downsampled Hi-Res (0.5x)"],
        "Eikonal Field Resampling Comparison"
    )

    # Visualize resampled vector fields
    visualizer.compare_vector_fields(
        [eik_vector, eik_vector_upsampled, eik_vector_downsampled],
        ["Original", "Upsampled (2x)", "Downsampled Hi-Res (0.5x)"],
        "Eikonal Vector Field Resampling Comparison"
    )

    # Plan paths using each field
    print("Planning paths would go here...")
    # Future implementation for path planning with these fields

    plt.show()


if __name__ == "__main__":
    main()
