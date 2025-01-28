# Data Science Consulting: Milwaukee Assessor's Office

## Overview

The aim of this project was to work with the Milwaukee Assessor's Office to explore and analyze their proprietary dataset of residential properties within Milwaukee. It was created as the Data Science Consulting Capstone Project at Loyola University Chicago. Given a dataset of properties and their features within Milwaukee, our goal was to come up with a list of 3-7 comparable properties for any particular subject property.  The resulting model would be used to assist the Office in defending the appraisal value for that subject property.

## Group Members

- Yamini Adusumilli: K-Means Clustering, Tableu Visualizations, Report Creation
- Hannah Harrach: Data Imputation, Report Creation
- Aldrich Paras: Exploratory Data Analysis, Feature Set Importance, General Boosted Model, K-D Trees, Report Creation

## Features 
- End-to-End Data Science Consulting
- Periodic Client Progress Presentations
- Full Report Documentation
- Machine Learning
- EDA

## Contributions
- Conducted EDA with Dependent Variable Distributions
- Developed a multi-dimensional KD-tree model within Sci-kit Learn to organize the data in k-dimensional space and search for similar properties or "comparable properties"
- Utilized a General Boosted Model to identify and use the most important features within the dataset


## Report Link
[Milwaukee Consulting Report](https://docs.google.com/document/d/18O1WtuvRIMa-UQsNSEyEuJmoAj5FVB0gZ8M7IXfxzIE/edit?usp=sharing)

## Installation and Usage

1. Clone the Repository using git clone https://github.com/ajtparas/milwaukee_consulting.git

2. Navigate to directory and open the project at milwaukee_consulting.Rproj

3. Raw Data is located at housing_data.csv, and final cleaned dataset is imputeddatafinal.csv
   
4. Open and run milwaukee_consulting_analysis_EDA.Rmd for EDA feature distribution Analysis

5. Open and run feature_importane.Rmd for general boosted model and feature importance selection

## Reflection
  
One of the biggest challenges was ensuring smooth communication between the cleaned data and model building. Inputs for creating models were dependent on having cleaned / imputed datasets, and workflow was bottlenecked at each handoff point. By communicating specific project deadlines within the group, we resolved potential workflow inefficiencies while still maintaining progress reports and communication with the client.  

## Future Work
- Create feature weights based on importance for more accurate machine learning accuracy
