
# Fordobots Trajectory Prediction and Segmentation

This repository contains tools for processing trajectory data and segmenting images using Facebook's SAM2 model.

---

## Setup and Usage

Follow the steps below to create your environment, install required dependencies, download the dataset, and run the interactive segmentation script.

1. **Create the Environment and Install Dependencies**

   Create a new Conda environment using the provided YAML file, then activate it:

   ```bash
   conda env create -f frod_traj_pred_env.yml
   ```

2. **Install SAM**

   Clone the SAM repository and install it in editable mode:

   ```bash
   git clone https://github.com/facebookresearch/sam2.git && cd sam2
   pip install -e .
   cd ..
   ```

3. **Download the Dataset**

   Download the dataset from Hugging Face by visiting the following link:

   [Fordobots Sampled Frames](https://huggingface.co/datasets/Steven-liudw/Fordobots_sampled_frames/tree/main/)

   Download only the folder you need and place it in your project directory. For example, create a folder named `sampled_frames` at the same level as the `sam2` directory.

4. **Run Interactive Segmentation**

   Launch the interactive segmentation process using the command below:

   ```bash
   python wp/data_utils/label_seg.py --frames_dir ./sampled_frames --interactive --compute_background
   ```

   **Interactive Mode Controls**

   - **Left-click:** Add positive points (green)
   - **Ctrl+Left-click:** Add negative points (red)
   - **Press 'r':** Reset user-added points (keeps waypoints)
   - **Press 'x':** Remove all points (including waypoints)
   - **Press 'backspace':** Undo the last user-added point
   - **Press 'q':** Quit without saving
   - **Press 'd':** Delete the image from the dataset
   - **Press SPACE:** Save and continue to the next image

---

Happy labeling!
```
