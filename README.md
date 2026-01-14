![SmartQSM logo](https://github.com/project-lightlin/misc/blob/main/img/smartqsm-logo.png?raw=true)

# SmartQSM (ISPRS Journal of Photogrammetry and Remote Sensing, 2026)

SmartQSM is a quantitative structure model for individual tree 3D reconstruction and parameter extraction based on point cloud data. It is a part of [Project LiGHTLIN](https://project-lightlin.github.io/).

## GUI

| Real-time processing and displaying | Interactive architectural analysis |
|-------------------------------------|-----------------------------------|
| ![Real-time processing and displaying](https://github.com/project-lightlin/misc/blob/main/img/smartqsm-effect1.gif?raw=true) | ![Interactive architectural analysis](https://github.com/project-lightlin/misc/blob/main/img/smartqsm-effect2.gif?raw=true) |

Support the extraction of 81 multiscale parameters including:

![SmartQSM parameters](https://github.com/project-lightlin/misc/blob/main/img/smartqsm-parameters.png?raw=true)

You can find the definitions and calculations of these parameters in the [paper]( https://doi.org/10.1016/j.isprsjprs.2026.01.011). Click [here to obtain the errata for the paper](https://github.com/project-lightlin/misc/blob/main/errata/errata_of_j.isprsjprs.2026.01.011.pdf).

> **Note:**
>
> This project is under long-term maintenance.
>
> **A major version upgrade is expected to take place in the summer of 2026.**
>
> We would greatly appreciate it if you could contribute code fixes or new features, and mark your contribution among the contributors:) You can open an issue or mail to the author and we will review it as soon as possible!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=project-lightlin/SmartQSM&type=date&legend=top-left)](https://www.star-history.com/#project-lightlin/SmartQSM&type=date&legend=top-left)

## Contributors

- (Author) Dr. YANG Jie, Beijing Forestry University ([nj_yang_jie@bjfu.edu.cn](mailto:nj_yang_jie@bjfu.edu.cn))

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.html).

To report any (possible) abuse or infringement behavior, please contact [yangtd@ifrit.ac.cn](mailto:yangtd@ifrit.ac.cn) with a detailed description and any supporting evidence. Thank you for helping us make our community a better place!

## Installation and Startup

### 0. Prerequisites

This repo has been verified and tested on Windows 10/11. Linux has not been tested.

1. Please refer to [how-to-install-python-and-deep-learning-libraries](https://project-lightlingithubio.readthedocs.io/en/latest/how-to-install-python-and-deep-learning-libraries.html) to prepare the virtual environment.

2. Install [Git](https://git-scm.com/) and clone the repo into the specified [working directory](https://project-lightlingithubio.readthedocs.io/en/latest/how-to-install-python-and-deep-learning-libraries.html#working-directory) using:

```bash
git clone https://github.com/project-lightlin/SmartQSM.git
```

in the CONDA environment.

*Or* you can:

A. click the *Code* button on the top right corner of the repo page,

B. click *Download ZIP*,

C. extract the downloaded zip file to the desired location.

> **Note:**
>
> We strongly recommend that you install [Git](https://git-scm.com/) for future upgrades.

3. Change the working directory of the virtual environment to the project directory (including requirements.txt).

4. Install the required dependencies in the virtual environment:

```bash
pip install -r requirements.txt
```

### 1. Run SmartQSM

```bash
python entrypoints/smartqsm.py
```

*Or* use command-line arguments:

```bash
python entrypoints/smartqsm.py [-h] [-v] [-y] [-c CONFIG] [-t] [CLOUD_PATHS ...]
```

You can refer to the specific parameter descriptions by using the `-h` option.
The default configuration file represented by `CONFIG` is located in the **configs/** directory of the project;
you can select one based on your requirements.
Alternatively, you can create a **.yaml** file with the same name as your input point cloud file in the same directory to achieve personalized reconstruction,
though this requires some programming experience.

> *Input*
>
> **Warning**: The ground and understory vegetation of the input individual tree point cloud should be cleaned up and there should be no large nontarget tree points, otherwise the reconstructed model will be distorted. 
>
> **Warning**: The shorter the plants (usually referring to those with a height of only about 1m or less), the more likely the reconstructed model is to swell. 
>
> You can import multiple point clouds from different folders in batch at once, and the program will process them sequentially.
>
> The names of the configuration files usually provide a brief summary of the skeletonization algorithm used, along with some additional information. Specifically,
>    - Use the configuration file marked "cpu" if the PyTorch version in the virtual environment is for CPU; use the configuration file marked "GPU" if the version is "cuXXX".
>    - Use “LEAFOFF” for cases with sparse or absent foliage; use “LEAFON” for cases with abundant foliage.


> *Output*
> 
> Each processed individual tree point cloud will generate five output files (**\*_active_crown.ply**, **\*_branches.ply**, **\*_crown.ply**, **\*_qsm.mat** and **\*_skeleton.dxf**) in the same directory. 
> PLY and DXF files can be easily opened by [CloudCompare](https://www.cloudcompare.org/).
> MAT file can be the input of some tools developed for [TreeQSM](https://github.com/InverseTampere/TreeQSM) such as [LeafGen](https://github.com/InverseTampere/leafgen). However, when using it, you need to transform the coordinates in `QSM.cylinder.start`. The recommended solution is to subtract `[QSM.treedata.X_m, QSM.treedata.Y_m, QSM.treedata.altitude_m]` from each coordinate. Otherwise, especially when a projected coordinate system is involved, the model cannot be displayed correctly.

### 2. Run QSM Viewer: Interactive Architectural Analysis Tool

```bash
python entrypoints/qsm_viewer.py
```

*Or*

```bash
python entrypoints/qsm_viewer.py PATH_OF_YOUR_qsm.mat
```
Import the **\*_qsm.mat** file, which must have a corresponding and unmodified **\*_branches.ply**. You can switch languages in the Display - Modify Language section.

### 3. Run Parameter Exporter

```bash
python entrypoints/parameter_exporter.py
```

Import one or more **\*_qsm.mat** files and convert them into an XLSX Excel workbook.

### 4. Run Stand Structurer: Calculation and tabular output tool for stand's spatial structure parameters

```bash
python entrypoints/stand_structurer.py
```

Use an Excel worksheet containing tree name, X coordinate, Y coordinate, tree height, DBH, mean crown width, and species to calculate stand's spatial structure parameters and output the results in an XLSX Excel file. The output worksheet from the output workbook by Parameter Exporter can be used as input once a species column is added.

## Known Bugs

1. After running, the window is black and continuously reports errors:

```c++
in void __cdecl filament::PlatformWGL::makeCurrent(struct filament::backend::Platform::SwapChain *,struct filament::backend::Platform::SwapChain *) noexcept:241
reason: wglMakeCurrent() failed. hdc = [Any 16-digit hexadecimal value]

Windows error code: 2000. (null)
```

Visit https://github.com/isl-org/Open3D/issues/3317

2. When running on Linux, it reports `Segmentation fault`.

Visit https://github.com/isl-org/Open3D/issues/6840

3. If automatic upgrades always fail, try replacing `entrypoints/_updater.py` with the latest version in the repo.

## Reference

If the code is helpful for your work, please cite:

> \[1\] Yang, J., Zhang, H., Li, J., Yang, H., Gao, T., Yang, T., Wang, J., Zhang, X., Yun, T., Duanmu, Y., Chen, S., & Shi, Y. (2026). SmartQSM: a novel quantitative structure model using sparse-convolution-based point cloud contraction for reconstruction and analysis of individual tree architecture. *ISPRS Journal of Photogrammetry and Remote Sensing*, *232*, 712–739. https://doi.org/10.1016/j.isprsjprs.2026.01.011

Recommend that you indicate the version used (see `version.txt`) and the configuration file (and any changes you made) in your paper.

## For developers

If

- you need to manually adjust parameters for specific data,  
- or you plan to further develop based on this code,

you can refer to the documentation at:  
https://project-lightlingithubio.readthedocs.io/en/latest/smartqsm-reference.html

For secondary development, please make sure you comply with the license.  
You will need to always click “No” when prompted to upgrade, or directly modify the `check_update` function in `entrypoints/_updater.py` to disable automatic updates.

You may also package your modified code for secondary development and send it via email to the author, along with a brief description, your name, and contact information. If your contribution proves useful, it may be included in future releases and acknowledged in the **Contributors** section.

## Changelog