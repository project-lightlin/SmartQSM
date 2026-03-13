# <img src="https://github.com/project-lightlin/misc/blob/main/img/smartqsm-logo.png?raw=true" width="192" height="24" alt="logo"> 2 Ginkgo *<span style="font-weight:normal;">assists in Comprehensive Survey of Ancient and Famous Trees</span>*

----------- 

**SmartQSM** is a quantitative structure model for individual tree 3D reconstruction and parameter extraction based on point cloud data. It is a part of [Project LiGHTLIN](https://project-lightlin.github.io/).

> **Note:**
>
> This project is under long-term maintenance.
>
> We would greatly appreciate it if you could contribute code fixes or new features, and mark your contribution among the contributors:) You can open an issue or mail to the author (see below) and we will review it as soon as possible!

## :smiley: First Impression

| Real-time processing and displaying | Interactive architectural analysis |
|-------------------------------------|-----------------------------------|
| ![Real-time processing and displaying](https://github.com/project-lightlin/misc/blob/main/img/smartqsm-effect1.gif?raw=true) | ![Interactive architectural analysis](https://github.com/project-lightlin/misc/blob/main/img/smartqsm-effect2.gif?raw=true) |

This method integrates multiple (reimplemented and improved) tree skeleton extraction algorithms including
>
> - :thumbsup: `spconv-contraction` *(Sparse-convolution-based point cloud contraction)* proposed by [Yang et al. (2026)](https://doi.org/10.1016/j.isprsjprs.2026.01.011) <sup>*This repo*</sup> and inspired by [Dobbs et al.(2023)](https://doi.org/10.1007/978-3-031-36616-1_28) [:octocat:](https://github.com/uc-vision/smart-tree)
> - :thumbsup: `layerwise-clustering` proposed by [Xu et al. (2007)](https://doi.org/10.1145/1289603.1289610) and inspired by [Yang et al. (2024)](https://doi.org/10.1002/rse2.399) [:octocat:](https://github.com/wanxinyang/treegraph)
>
> **From Ver 2.0:**
> - :thumbsup: `flexible-layerwise-clustering` inspired by [Wang et al. (2025)](https://doi.org/10.48550/arXiv.2506.15577)
> - `space-colonization` proposed by [Runions et al. (2007)](https://doi.org/10.2312/nph/nph07/063-070)
>
> **More to come...**

and supports the extraction of [82 multiscale parameters (Click here to view details)](https://github.com/project-lightlin/misc/blob/main/doc/parameter-list-of-smartqsm.md) including

| Type | Parameters |
| --- | --- |
|Individual-scale: tree parameters| Location, Number of branches, Max branch order, Tree height, DBH, Girth, Ground diameter, Bole height, Diameter at bole height, Bole length, Bole area, Bole volume, Trunk length, Trunk area, Trunk volume, Stem length, Stem area, Stem volume, Within-crown stem length, Within-crown stem area, Within-crown stem volume, Min crown radius, Azimuth of min crown radius, Height at min crown radius, Mean crown radius, Max crown radius, Azimuth of max crown radius, Height at max crown radius, Min crown width, Azimuth of min crown width, Mean crown width, Max crown width, Azimuth of max crown width, East-west crown width, North-south crown width, Crown convex area, Crown convex volume, Active crown convex area, Active crown convex volume, Crown projection convex area, Crown perimeter, Canopy area, Crown center offset, Crown center azimuth, Min crown spread, Azimuth of min crown spread, Max crown spread, Azimuth of max crown spread, etc.|
|Organ-scale: branch attributes| Order, Base height, Base diameter, Mid-length diameter, Tip diameter, Length, Area, Volume, Max spread, Azimuth, Zenith, Chord length, Arc height, Height difference, Branching radius, Branching angle, Tip deflection angle, Vertical deflection angle, Tip-based DINC, Apex-based DINC, Growth length, Growth area, Growth volume, Base offset, Base azimuth, Insertion distance, etc.|
|Plot-scale: stand's spatial structure indices| Uniform angle index, Hegyi's competition index, Mingling, Tree species diversity mingling, Diameter dominance, Crowdedness, Openness, Within-unit species richness, etc.|


:smile: We are truly honored that SmartQSM was featured on [CCTV News Channel's "Morning News" (Click here to watch the report)](https://tv.cctv.com/2026/02/23/VIDEENOqOEryLSemunBIe6Cs260223.shtml)!

![挂牌保护、智能监测 古树名木在数智化时代焕发新生](https://github.com/project-lightlin/misc/blob/main/img/smartqsm-on-cctv13.jpg?raw=true)

## :star2: Star History

[![Star History Chart](https://api.star-history.com/svg?repos=project-lightlin/SmartQSM&type=date&legend=top-left)](https://www.star-history.com/#project-lightlin/SmartQSM&type=date&legend=top-left)

## :medal_sports: Contributors

> ### Author
> - @teduzhu Dr. Jie Yang (杨杰), Beijing Forestry University ([nj_yang_jie@bjfu.edu.cn](mailto:nj_yang_jie@bjfu.edu.cn))
> ### Facilitator
> - @jk160804211 Dr. Kang Jiang (蒋康), Nanjing Forestry University (who provided reimplementation of space colonization)
> ### Tester
> - @Luneighbour Mr. Nanbo Lu (鲁南博), Central South University of Forestry & Technology
> ### Special Thanks
> 

## :balance_scale: License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.html).

To report any (possible) abuse or infringement behavior, please contact [yangtd@ifrit.ac.cn](mailto:yangtd@ifrit.ac.cn) with a detailed description and any supporting evidence. Thank you for helping us make our community a better place!

## :inbox_tray: Installation and Startup

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

4. Finish post-installation by running

```bash
python post_installation.py
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
The default configuration file represented by `CONFIG` is located in the **configs/** directory of the project; you can select any one based on your requirements.
*Advanced Usage*: You can create a **.yaml** file with the same name as your input point cloud file in the same directory to achieve personalized reconstruction at the file level.

> *Input*
>
> **Warning**: The ground and understory vegetation of the input individual tree point cloud should be cleaned up and there should be no large nontarget tree points, otherwise the reconstructed model will be distorted. 
>
> You can import multiple point clouds from different folders in batch at once, and the program will process them sequentially.
>
> The names of the configuration files usually provide a brief summary of the skeletonization algorithm used, along with some additional information. Specifically,
>    - Use the configuration file marked "cpu" if the PyTorch version in the virtual environment is for CPU; use the configuration file marked "GPU" if the version is "cuXXX".
>    - Use “LEAFOFF” for cases with sparse or absent foliage; use “LEAFON” for cases with abundant foliage.


> *Output*
> 
> Each processed individual tree point cloud will generate five output files in the same directory, including:
> - **\*_active_crown.ply**: Triangle mesh of active crown convex hull
> - **\*_branches.ply**: Triangle mesh of branching structure colored by order
> - **\*_crown.ply**: Triangle mesh of crown convex hull
> - **\*_qsm.mat**: This file can be the input of some tools developed for [TreeQSM](https://github.com/InverseTampere/TreeQSM) such as [LeafGen](https://github.com/InverseTampere/leafgen). However, when using it, you need to transform the coordinates in `QSM.cylinder.start`. The recommended solution is to subtract `[QSM.treedata.X_m, QSM.treedata.Y_m, QSM.treedata.altitude_m]` from each coordinate. Otherwise, especially when a projected coordinate system is involved, the model cannot be displayed correctly.
> - **\*_skeleton.dxf**: Line set of tree skeleton
>
> PLY and DXF files can be opened by [CloudCompare](https://www.cloudcompare.org/).

### 2. Run QSM Viewer: Interactive architectural analysis tool

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

## :rocket: About Upgrade

The program automatically checks for updates at each run and calls Git to implement updates when there are updates. Please install [Git](https://git-scm.com/) properly. If Git fails to clone the repository, you may need to set up a proxy. Please check the sample configuration in `entrypoints/git_config_sample.txt`. Copy this file and rename it to `git_config.txt` in the same directory for the configuration. Remove the `#` at the beginning of the line to make the modified argument activate.

If automatic upgrades always fail, **especially when the version is < 2.0.4**, try replacing `entrypoints/_updater.py` with the latest version in the repo.

## :warning: Known Bugs

1. After running, the window is black and continuously reports the following error, Your computer may have a dual-graphics setup (integrated + dedicated). Please set the preferred graphics processor to High-performance processor in your settings. For detail, visit https://github.com/isl-org/Open3D/issues/3317

```c++
in void __cdecl filament::PlatformWGL::makeCurrent(struct filament::backend::Platform::SwapChain *,struct filament::backend::Platform::SwapChain *) noexcept:241
reason: wglMakeCurrent() failed. hdc = [Any 16-digit hexadecimal value]

Windows error code: 2000. (null)
```

2. When running on Linux, GUI may not start. The error `Segmentation fault` does not seem to affect the operation of the program. If there are any issues, perhaps you can refer to https://github.com/isl-org/Open3D/issues/6840

## :memo: Reference

If the code is helpful for your work, please cite:

> Yang, J., Zhang, H., Li, J., Yang, H., Gao, T., Yang, T., Wang, J., Zhang, X., Yun, T., Duanmu, Y., Chen, S., & Shi, Y. (2026). SmartQSM: a novel quantitative structure model using sparse-convolution-based point cloud contraction for reconstruction and analysis of individual tree architecture. *ISPRS Journal of Photogrammetry and Remote Sensing*, *232*, 712–739. https://doi.org/10.1016/j.isprsjprs.2026.01.011
> 

Recommend that you indicate the version used (see `version.txt`) and the configuration file (and any changes you made) in your paper.

## :wrench: For developers

For secondary development, please make sure you comply with the license.  

You will need to always click “No” when prompted to upgrade, or directly modify the `check_update` function in `entrypoints/_updater.py` to disable automatic updates.

We strongly recommend that you develop based on the minimum Ver 2.0, as many fields and methods in Ver 1.X were deprecated or changed after Ver 2.0.

You may also package your modified code for secondary development and send it via email to the author, along with a brief description, your name, and contact information. If your contribution proves useful, it may be included in future releases and acknowledged in the **Contributors** section.
