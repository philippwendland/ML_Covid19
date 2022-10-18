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
* joblib [7]
* scipy [8]
* statsmodels [9]
* xgboost [10]
* skopt [11]

# References 
[1] Philipp Wendland, Vanessa Schmitt, Jörg Zimmermann, Christof Schenkel-Häger and Maik Kschischo. "Prediction of Covid-19 associated sentinel events based on laboratory values via machine learning in hospitals". 2022

[2] Wes McKinney. "Data Structures for Statistical Computing in Python". Python in Science Conference. Austin, Texas, 2010, p. 56–61. DOI : 10.25080/Majora-92bf1922-00a

[3] Stéfan van der Walt, S Chris Colbert etal. "The NumPy Array: A Structure for Efficient Numerical Computation". Computing in Science & Engineering 13.2, p. 22–30. ISSN : 1521-9615. DOI : 10.1109/MCSE.2011.37

[4] John D. Hunter. "Matplotlib: A 2D Graphics Environment". Computing in Science & Engineering 9.3 (2007), p. 90–95. ISSN : 1521-9615. DOI : 10.1109/MCSE.2007.55

[5] Michael L. Waskom "seaborn: statistical data visualization". 2021 Journal of Open Source Software, p. 3021. DOI: 10.21105/joss.03021

[6] Fabian Pedregosa, Gaël Varoquaux etal. "Scikit-learn: Machine Learning in Python". arXiv:1201.0490 2018

[7] Joblib Development Team "Joblib: running Python functions as pipeline jobs". 2020. Url: https://joblib.readthedocs.io/

[8] Paul Virtanen, Ralf Gommers etal. "SciPy 1.0: fundamental algorithms for scientific computing in Python". Nature Methods 17.3, p. 261–272. ISSN :1548-7091, 1548-7105. DOI : 10.1038/s41592-019-0686-2

[9] Seabold, S., & Perktold, J. "statsmodels: Econometric and statistical modeling with python". 2010. In 9th Python in Science Conference.

[10] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York, NY, USA: ACM; 2016. p. 785–94.. DOI : 10.1145/2939672.2939785

[11] Joblib Development Team "Scikit-Optimize" URL: https://zenodo.org/record/1207017#.Ys6tzoTP0Q8

**Contact**: Philipp Wendland - wendland.philipp@web.de
