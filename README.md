# seismic-clustering
This repository provides an unsupervised machine learning workflow to identify fault structures and spatial cluster orientations from earthquake catalogs. By combining density-based spatial clustering (**DBSCAN**) and Principal Component Analysis (**PCA**), the pipeline detects hypocenter clusters, evaluates clustering quality via multiple density and distance metrics, and extracts geometric plane features corresponding to subsurface fault configurations.

---

##  Requirements & Environment
- **Python Version**: Python 3.10+
- **Primary Dependencies**:
  - `numpy==1.26.4`
  - `pandas==2.3.3`
  - `geopandas==1.1.1`
  - `scikit-learn==1.7.2`
  - `kDBCV==1.0.0`
  
### Installation

1. **Clone the repository and set up a virtual environment**:

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```



##  Input Data Format
Please place your raw catalog files into the catalog/ directory (e.g., catalog/filename.csv).  
The pipeline expects earthquake catalogs in **CSV** format. 
* `catalog` (*pd.DataFrame*): Earthquake catalog attributes
  - `otime` (*str / datetime*): Origin time (`yyyy-mm-dd HH:MM:SS.sss`)
  - `lat` (*float*): Latitude in decimal degrees (°)
  - `lon` (*float*): Longitude in decimal degrees (°)
  - `dep` (*float*): Hypocenter depth (km)
  - `mag` (*float*): Earthquake magnitude



##  Execution

To execute the clustering and geometric fitting workflow, run the main script:

```bash
python cal_DBSCAN_PCA.py
```


## Pipeline & Subroutine Specifications

### 1. cal_DBSCAN
Performs density-based clustering on normalized spatial hypocenter coordinates and computes clustering validity indices.

* **Inputs**:
  * `infile` (*str / Path*): Path to the earthquake catalog CSV file.
  * `ep` (*float*): Search radius $\epsilon$ (normalized range: `0` to `1`).
  * `sp` (*int*): Minimum number of samples required to form a core point ($\text{MinPts}$).

* **Outputs**:
  * `n`       (*int*)  : Total number of detected clusters (include noise).
  * `s_ss`    (*float*): Silhouette Score.
  * `s_dbcv`  (*float*): Density-Based Clustering Validation score.
  * `s_dbi`   (*float*): Davies-Bouldin Index.
  * `s_chi`   (*float*): Calinski-Harabasz Index.
  * `n_noise` (*int*) : Total count of noise points.
  * `dfx` (*pd.DataFrame*): Earthquake catalog merged with cluster assignment labels.
      - `otime` (*str / datetime*): Origin time (`yyyy-mm-dd HH:MM:SS.sss`)
      - `lat` (*float*): Latitude in decimal degrees (°)
      - `lon` (*float*): Longitude in decimal degrees (°)
      - `dep` (*float*): Hypocenter depth (km)
      - `mag` (*float*): magnitude
      - `x`   (*float*): Projected X coordinate(m) (TWD97 Easting)
      - `y`   (*float*): Projected Y coordinate(m) (TWD97 Northing)
      - `z`   (*float*): Hypocenter depth(m)
      - `grps`(*int*)  : Cluster label (-1: noise; 0, 1, 2, ... cluster labels)


### 2. run_pca
Fits planar geometry to hypocenter coordinates within each identified cluster to infer strike, dip, and dimensions.

* **Inputs**:
  * `df` (*pd.DataFrame*): Clustered event dataset generated from `cal_DBSCAN`.

* **Outputs**:
  * `df_pca`  (*pd.DataFrame*): Fault plane properties :
    - `grp`   (*int*)  : Cluster label
    - `mw`    (*float*): Moment magnitude
    - `width` (*float*): Fault width (km)
    - `length`(*float*): Fault length (km)
    - `strike`(*float*): Fault strike angle (°)
    - `dip`   (*float*): Fault dip angle (°)

## License
The project is licensed under the [MIT License](LICENSE).


