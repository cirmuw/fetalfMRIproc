# Fetal rs-fMRI
custom workflow for in-utero resting-state fMRI analysis, including preprocessing, quality control, and simulation the hemodynamic response function
### Usage
Run the preprocessing workflow from the command line with the following syntax:
```bash
python run_preprocessing_workflow.py \
    --input <input_bold_file> \
    --input-mask <input_mask_file> \
    --output <output_directory> \
    --T2-to-bold <transformation_matrix> \
    --segmentation <segmentation_file> 
```
A more elaborate example could be:
```bash
python run_preprocessing_workflow.py \
    --input <input_bold_file> \
    --input-mask <input_mask_file> \
    --output <output_directory> \
    --T2-to-bold <transformation_file> \
    --segmentation <segmentation_file> \
    --parcellation <parcellation_file> \
    --registration-method <method> \
    --registration-type <type> \
    --n4-method <method> \
    --despiking-method <method> \
    --vout-method <method> \
    --regression-model <model> \
    --num-components <num_components> \
    --temporal-filter <filter_type> \
    --dummy-frames <num_frames> \
    --dilation-radius <radius> \
    --interleave-factor <factor>
```
### Installation
1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/cirmuw/fetalfMRIproc.git
    cd fetalfMRIproc
    ```
2. (Optional) Set up a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
### How to cite
If you use this code in your work for preprocessing funtional MRI, performing automated QC, or simulating BOLD signal, please cite the following paper:

Athena Taymourtash, Ernst Schwartz, Karl-Heinz Nenning, Roxane Licandro, Patric Kienast, Veronika Hielle, Daniela Prayer, Gregor Kasprian, and Georg Langs (2024). "Measuring the Effects of Motion Corruption in Fetal fMRI". Human Brain Mapping