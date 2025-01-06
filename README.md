# Fetal rs-fMRI
A Python-based, open-source toolkit for the analysis of in-utero fetal resting-state fMRI (rs-fMRI) data. This toolkit provides a flexible, modular workflow for preprocessing, quality control, simulation, and connectivity analysis, with optional integration of external neuroimaging tools like FSL, AFNI, and ANTs. 

### Key Features

* **Modular Preprocessing Pipeline**: We implemented the entire preprocessing workflow including bias field correction, motion correction, despiking, volume outlier rejection, nuisance regression, and temporal filtering. Each step is modular, allowing users to choose and configure methods based on their specific needs.
* **Quality Control**: Automated quality-control report for a single subject including visual and quantitative metrics to assess preprocessing effectiveness and data quality.
* **Connectivity Analysis**: Extraction of ROI time series and performing functional connectivity analysis.
* **Hemodynamic Response Simulation**: 
* **Optional Integration of neuroimaging tools**: Compatible with FSL, AFNI, and ANTs if installed (see below for details).

If you have any questions or comments, please drop an email to `georg.langs@meduniwien.ac.at`.

---

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
--- 

### Installation

This toolkit is **python-based**, requiring no external neuroimaging software for its core functionality. However, tools like **NiftyReg**, **FSL**, **AFNI**, and **ANTs** can be integrated for advanced features if installed.

**Basic Installation:**

Clone the repository to your local machine, set up a virtual environment (using conda or python) and and install dependencies:

<strong style="font-size: smaller;">Using Conda:</strong>
```bash
git clone https://github.com/cirmuw/fetalfMRIproc.git
cd fetalfMRIproc
conda env create -f environment.yaml
conda activate ffmri-env
export PYTHONPATH=$PYTHONPATH:/path/to/fetalfMRIproc
```
<strong style="font-size: smaller;">Using Python Virtualenv:</strong>
```bash
python -m venv ffmri-env
source ffmri-env/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:/path/to/fetalfMRIproc
````

**(Optional) Make ```PYTHONPATH``` Persistent:**
To avoid setting ```PYTHONPATH``` each time, add it to your shell configuration file (```~/.bashrc``` or ```~/.zshrc```):
```bash
echo "export PYTHONPATH=$PYTHONPATH:/path/to/fetalfMRIproc" >> ~/.bashrc
source ~/.bashrc
```
**(Optional) Integrating Neuroimaging Tools**

If you wish to use **NiftyReg**, **FSL**, **AFNI**, or **ANTs**, install them separately and update paths in `src/utilities/definitions.py`.

--- 
### How to cite
If you use this code in your work for preprocessing funtional MRI, performing automated QC, or simulating BOLD signal, please cite:


* [Taymourtash et al. 2024] Athena Taymourtash, Ernst Schwartz, Karl-Heinz Nenning, Roxane Licandro, Patric Kienast, Veronika Hielle, Daniela Prayer, Gregor Kasprian, and Georg Langs (2024). "Measuring the Effects of Motion Corruption in Fetal fMRI". Human Brain Mapping

* [[Taymourtash et al. 2023](https://academic.oup.com/cercor/article/33/9/5613/6908756)] Taymourtash, A., Schwartz, E., Nenning, K. H., Sobotka, D., Licandro, R., Glatter, S., Diogo, M.C., Golland, P., Grant, E., Prayer, D. Kasprian, G. & Langs, G. (2023). Fetal development of functional thalamocortical and cortico–cortical connectivity. Cerebral Cortex, 33(9), 5613-5624