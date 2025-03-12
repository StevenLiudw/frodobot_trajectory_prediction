
## Install SAM
```
cd third_party && git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e . && cd ..
```


## Process data

### Pre-processing
- Compass calibration
- EKF filtering for odometry and GPS, calibrated compass
- Filter stationary trajectory

```bash
python wp/data_utils/filter_trajectory.py 
```

### Filtering
- Filter night trajectory
- Uniform sample from GPS locations

```bash
python wp/data_utils/gps_analysis.py --process --report --filter_night \
--sample 1000 --collect --divide 4
```

### Auto labeling
- Waypoint

```bash
python -m wp.data_utils.auto_label_async --max_labels 10000 --visualization_prob 0 \
--shuffle --num_threads 10

# check labels
python wp/data_utils/check_labels.py --data_dir data/filtered_2k --visualize
```

### Segmentation labeling

```bash
python wp/data_utils/label_seg.py --interactive --compute_background --overwrite
```

### Upload / download from hf

```bash
# make sure the repo is set correctly, and set your hf token by --token
python wp/data_utils/hf_utils.py --upload
```
Alternatively, you can directly download a subset https://huggingface.co/datasets/jamiewjm/fd/tree/main
