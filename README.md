# CropML

# Non-invasive Phenotyping for Water and Nitrogen Uptake by Deep Roots Explored using Machine Learning
Under review (Plant and soil)
# Authors
Satyasaran Changdar, Olga Popovic, Tomke Susanne Wacker, Bo Markussen, Erik Bjørnager Dam, Kristian Thorup-Kristensen
Affiliations
University of Copenhagen, Denmark

# https://doi.org/10.21203/rs.3.rs-2608651/v1
# CropML

CropML is a machine learning project focused on non-invasive phenotyping of crop root systems using machine learning techniques. The goal of this project is to explore the relationship between root distribution and resource uptake in crops using machine learning models.

## Table of Contents

- [Abstract](#abstract)
- [Usage](#usage)
- [Data](#data)
- [Contributing](#contributing)
- [Installation](#installation)


## Abstract

This project investigates the relationship between root distribution and resource uptake in crops using machine learning techniques. The study utilizes the RadiMax semi-field root-screening facility to phenotype winter wheat genotypes for root growth. Square root of planar root length density (Sqrt_pRLD) measurements are collected at different soil depths, and their correlation with deep soil nitrogen uptake and drought resilience potential is explored using machine learning models. The results demonstrate the importance of deep rooting for water and nitrogen uptake in crops.

To run the code and reproduce the analysis, please use the `RadimaxPaper_ML_June25.ipynb` Jupyter Notebook file.

## Usage

Open the RadimaxPaper_ML_June25.ipynb notebook in Jupyter Notebook or any compatible notebook application. Run the notebook cells in sequential order to execute the code and reproduce the non-invasive phenotyping analysis. 
Feel free to modify the code, experiment with different machine learning models or techniques, and explore the results. Other .py files are our custome library files


Please refer to the notebook or relevant sections of the code for further details on the dataset's structure and preprocessing steps.
## Data
The project's data (2018 and 2019, Raw data) is stored in the Data folder, which contains the necessary files for the analysis.


## Contributing
Contributions to this project are welcome! If you find any issues, have suggestions, or would like to add new features, feel free to open an issue or submit a pull request.

## Installation

To set up the project locally, follow these steps:

```shell
git clone https://github.com/satyasaran/CropML.git
cd CropML
python -m venv env
source env/bin/activate (for Linux/Mac)
env\Scripts\activate (for Windows)
pip install -r requirements.txt
Once the dependencies are installed, you can proceed to the usage section above to run the code. ```

