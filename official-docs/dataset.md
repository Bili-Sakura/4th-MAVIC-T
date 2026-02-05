# Dataset

Source URL: https://www.codabench.org/competitions/12566/

## Dataset

## Dataset Availability

**Training and evaluation**:

- [Train Data and Validation(EO+SAR)](https://codalab.lisn.upsaclay.fr/my/datasets/download/0b7459aa-37af-4d51-a13b-6a6568c1fe1d)
- [Train Data UC Davis (SAR, IR, RGB)](https://codalab.lisn.upsaclay.fr/my/datasets/download/b756ef2b-8b05-4686-9efd-cb6b740175a8)
- [Train Data Manhattan (SAR, IR, RGB)](https://codalab.lisn.upsaclay.fr/my/datasets/download/84f2e85a-b89f-434f-9bf4-0eda888ae997)
- [Train Data Bingham (SAR, IR, RGB)](https://codalab.lisn.upsaclay.fr/my/datasets/download/8cd394ac-be0e-41df-bb71-6a9b5530df50)
- [Train Data Centerfield (SAR, IR, RGB)](https://codalab.lisn.upsaclay.fr/my/datasets/download/9ea2e761-9d2f-4dfe-b398-bc46e3fd0f71)
- [Validation (SAR, IR, RGB)](https://codalab.lisn.upsaclay.fr/my/datasets/download/2c5e1f60-00dd-46c2-8c91-3753970b4905)

**Testing**:

The testing set can be downloaded from the **"Files"** tab in this repository, under the **"MAVIC_T_2025_test_set"** option, or it can be accessed externally from [here](https://drive.google.com/file/d/17rNN7eGqgT8o-viPmODTaua6TJSwH58H/view?usp=sharing).

## Dataset Description

We reuse the MAGIC-STACKS data from previous competitions. This dataset incorporates diverse geospatial data sources for its analysis. The United States Geological Survey’s (USGS) Earth Resources Observation and Science (EROS) program provides crucial land change imagery, including satellite and aerial data. Specifically, MAGIC leverages the EROS High Resolution Orthoimagery (HRO) dataset, which offers a vast collection of orthorectified aerial imagery with submeter resolution and consistent scale.

The dataset is a composite of three sources, processed and aligned to form uniform chipped stacks. The three source datasets used are UNICORN, USGS HRO, and the UMBRA open data program. Note we have both an aerial source for SAR and a satellite source for SAR. Each dataset undergoes preprocessing to ensure images are georectified, stored as GeoTIFFs with spatial referencing, and organized in a unified directory. Subsequently, a standardized chipping process divides the preprocessed data into spatially aligned 200m x 200m image stacks, indexed by their WGS-84 topleft coordinates

### Usage notes

- The training set may be used freely for model development and training.
- The validation set is intended for local evaluation and model tuning and should not be used for final reporting.
- The test set must not be used for training, as its annotations are not released.
- Participants are expected to use only the data provided within the scope of this challenge, in accordance with the terms and conditions.
