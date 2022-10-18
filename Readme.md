# Machine learning models for predicting severe COVID-19 outcomes in hospitals

This is the Githup Repository for the paper Prediction of Covid-19 associated sentinel events based on laboratory values via machine learning in hospitals by Wendland etal. [1] containing the code to reconstruct our results and additional supplementary files. In our work we construct machine-learning based models
for the prediction of in-hospital mortality, transfer to intensive-care unit (ICU) and mechanical ventilation of hospitalized Covid-19 patients. We use age, biological sex and averaged covariates of the first 48 or 24 hours after admission to a hospital as (possible) covariates. 

**Important**: The patient data used in our analysis is not included in this repository because we are not allowed to publicly share these data due to german law. 

# Content of the repository
* In the folder Code_Paper you can find the code to reproduce our findings
* The folder Tests contain complete csv files with all wilcoxon-rank-sum-tests and t-tests for all endpoints 
* The lab_abbreviations.csv file contains a complete list of the full names of all laboratory values

# Explanations regarding the Code for the paper

* To install all package dependencies we recommend to create a anaconda environment via the environment.yml file using "conda env create -f environment.yml"

# List of Package dependencies

* pandas [2]
* numpy [3]
* matplolib [4] 
* seaborn [5]
* sklearn [6] 
* scipy [7]
* statsmodels [8]
* xgboost [9]
* skopt [10]
* tidyverse [11]
* proc [12]

# References 
[1] Wendland P, Schmitt V, Zimmermann J, Schenkel-Häger C, Kschischo M. "Machine learning models for predicting severe COVID-19 outcomes in hospitals". 2022

[2] McKinney W. Data Structures for Statistical Computing in Python, Austin, Texas: 2010, p. 56–61. https://doi.org/10.25080/Majora-92bf1922-00a.

[3] Harris CR, Millman KJ, van der Walt SJ, Gommers R, Virtanen P, Cournapeau D, et al. Array programming with NumPy. Nature 2020;585:357–62. https://doi.org/10.1038/s41586-020-2649-2.

[4] Hunter JD. Matplotlib: A 2D Graphics Environment. Comput Sci Eng 2007;9:90–5. https://doi.org/10.1109/MCSE.2007.55.

[5] Waskom M. seaborn: statistical data visualization. JOSS 2021;6;3021. https://doi.org/10.21105/joss.03021.

[6] Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: Machine Learning in Python 2018.

[7] Virtanen P, Gommers R, Oliphant TE, Haberland M, Reddy T, Cournapeau D, et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat Methods 2020;17:261–72. https://doi.org/10.1038/s41592-019-0686-2.

[8] Seabold S, Perktold J. Statsmodels: Econometric and Statistical Modeling with Python, Austin, Texas: 2010, p. 92–6. https://doi.org/10.25080/Majora-92bf1922-011.

[9] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco California USA: ACM; 2016, p. 785–94. https://doi.org/10.1145/2939672.2939785.

[10] Head T, MechCoder, Louppe G, Iaroslav Shcherbatyi, Fcharras, Zé Vinícius, et al. Scikit-Optimize/Scikit-Optimize: V0.5.2. Zenodo; 2018. https://doi.org/10.5281/ZENODO.1207017.

[11] Wickham H, Averick M, Bryan J, Chang W, McGowan L, François R, et al. Welcome to the Tidyverse. JOSS 2019;4:1686. https://doi.org/10.21105/joss.01686. 

[12] Robin X, Turck N, Hainard A, Tiberti N, Lisacek F, Sanchez J-C, et al. pROC: an open-source package for R and S+ to analyze and compare ROC curves. BMC Bioinformatics 2011;12:77. https://doi.org/10.1186/1471-2105-12-77.

**Contact**: Philipp Wendland - wendland.philipp@web.de
