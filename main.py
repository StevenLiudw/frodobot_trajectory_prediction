

from wp.data_utils.dataloader import visualize_trajectory_samples


if __name__ == "__main__":
    visualize_trajectory_samples(
        data_dir="data/filtered_2k",
        output_dir="out/visualization_output",
        num_samples=100,
        n_waypoints=10
    )
    