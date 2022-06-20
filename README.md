## Zillow Zestimate Log Error Analysis and Improvement Modeling

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Project Summary
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Project Objectives
> - Use clustering to improve EDA and/or create new features that improve the predictive power of log error (a proxy for Zestimate).
> - Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook report.
> - Create modules (acquire.py, wrangle.py) that make your process repeateable.
> - Deliver a 5 minute presentation consisting of a high-level notebook walkthrough using the Jupyter Notebook from above; the presentation should be appropriate for the target audience.
> - Answer panel questions about your code, process, findings and key takeaways, and model.

#### Business Goals
> - Identify Key Drivers of Zillow's current model's log error.  This is based on exisiting modeling work.  This will help us (Zillow) continuously improve on our product offering as it is one of our most important business offerings.
> - Construct a ML classification model that predicts a home's log error based on key features.  This will ensure the driver's identified are fully engaged towards improving the customer experience.
> - Document my process well enough to be presented or read like a report.

#### Audience
> - Zillow Data Science Team
> - CodeUp Students!

#### Project Deliverables
> - A final report notebook (final_report.ipynb)
> - A predictor of log error driven by best ML model
> - All necessary modules to make my project reproducible

#### Data Dictionary
- Note: Includes only pre-encoded features of signifigance:

|Target|Datatype|Definition|
|:-------|:--------|:----------|
| abs_logerror | 45474 non-null: float64 | difference between log of Zestimate and log of actual price |

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| bedrooms       | 45474 non-null: int64 | number of bedrooms |
| bathrooms        | 45474 non-null: in64 | number of bathrooms in home |
| sqft       | 45474 non-null: int64 | home interior size, in square feet |
| yearbuilt        | 45474 non-null: int64 | year the home was built |
| latitude        | 45474 non-null: int64 | location (N-S) of home |
| longitude        | 45474 non-null: int64 | location (E-W) of home |
| lotsize        | 45474 non-null: int64 | size of the property (land) in square feet |
| transactiondate        | 45474 non-null: object | latest date property was sold |
| structurevalue        | 45474 non-null: int64 | tax assessed cost of all buildings on the property |
| salesvalue        | 45474 non-null: int64 | most recent tax assessed value (sale price) for the property |
| landvalue        | 45474 non-null: int64 | tax assessed cost of underlying land of the property |
| taxamount        | 45474 non-null: int64 | tax assessed on the property |
| structure_value_ratio        | 45474 non-null: float64 | structurevalue/salesvalue (allows comparison between strucure and land value) |
| structurevalue        | 45474 non-null: int64 | tax assessed cost of all buildings on the property |
| geo_sales_value        | 45474 non-null: int64 | average value of similarly located homes |
| county *        | 45474 non-null: uint8 | * ENCODED into 3 values for county (Orange, Ventura, Los Angeles) |
| landusecode **       | 45474 non-null: unit8 | ** ENCODED county level code indicating zoning use (14 unique values) |

#### Initial Hypotheses

> - **Hypothesis 1 -**
> - Using a narrow geographic feature will improve logerror

> - **Hypothesis 2 -** 
> - Structure value is correlated with logerror, more so than land value

> - **Hypothesis 3 -** 
> - Land use codes have an impact on logerror

> - **Hypothesis 4 -** 
> - Transaction date has an impact on logerror

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Executive Summary - Conclusions & Next Steps
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

> - Problem:  Zillow needs to better understand the driver's behind its current model's log error measurement - that is the difference between the natural log of a property's 'Zestimate' and the natural log of the actual selling price.
> - Actions: Examined 2017 actual selling price data ('tax assessed value') for 3 Southern California counties, along with features collected by Zillow and the log error from Zillow's Zestimate (the Target variable).  Used clustering technique to dive deeper into feature relationships and create new ones. Then built numerous log error prediction models based on these features, testing the best one on the dataset to see if it can beat baseline.
> - Conclusions: Initial clustering efforts provided no major illiuminations on the drivers of log error, nor created any useful features or groupings of continuous variables that improved the model much compared to existing features.  Overall though, a ~10% improvement over baseline logerror was modeled.
> - Recommendations: Instead of predicting effects on logerror, the analysis should go back to focusing on improving overall model accuracy (reducing RMSE around sales price - aka taxvaluedollaramt).  For next steps I would continue to look for useful clustering.

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Pipeline Stages Breakdown

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

##### Plan
- [x] Create README.md with data dictionary, project and business goals, come up with initial hypotheses.
- [x] Acquire data from the Zillow (Codeup) Database and create a function to automate this process. Save the function in an acquire.py file to import into the Final Report Notebook.
- [x] Clean and prepare data for the first iteration through the pipeline, MVP preparation. Create a function to automate the process, store the function in a wrangle.py module, and prepare data in Final Report Notebook by importing and using the funtion.
- [x] Investigate data, formulate hypotheses, visualize analsyis and run statistical tests when necessary (ensuring signifigance and hypotheses are created and assumptions met).  Document findings and takeaways.
- [x] Use clustering and cluster analysis to better understand the data or reate new features.
- [x] Establish a baseline accuracy.
- [x] Train multiple different regression models, to include hyperparameter tuning.
- [x] Evaluate models on train and validate datasets.
- [x] Choose the model with that performs the best and evaluate that single model on the test dataset.
- [x] Create csv file with predictions on test data.
- [x] Document conclusions, takeaways, and next steps in the Final Report Notebook.

___

##### Plan -> Acquire
> - Store functions that are needed to acquire zillow home data from the Codeup data science database server; make sure the acquire.py module contains the necessary imports for anyone with database access to run the code.
> - The final function will return a pandas DataFrame.
> - Import the acquire function from the acquire.py module and use it to acquire the data in the Final Report Notebook.
> - Complete some initial data summarization (`.info()`, `.describe()`, `.value_counts()`, ...).
> - Plot distributions of individual variables.
___

##### Plan -> Acquire -> Prepare/Wrange
> - Store functions needed to wrangle the zillow home data; make sure the module contains the necessary imports to run the code. The final functions (wrangle.py) should do the following:
    - Split the data into train/validate/test.
    - Handle any missing values.
    - Handle erroneous data and/or outliers that need addressing.
    - Encode variables as needed.
    - Create any new features, if made for this project.
> - Import the prepare functions from the wrangle.py module and use it to prepare the data in the Final Report Notebook.
___

##### Plan -> Acquire -> Prepare -> Explore
> - Answer key questions, my hypotheses, and figure out the features that can be used in a regression model to best predict the target variable, selling_price. 
> - Run at least 2 statistical tests in data exploration. Document my hypotheses, set an alpha before running the tests, and document the findings well.
> - Create visualizations and run statistical tests that work toward discovering variable relationships (independent with independent and independent with dependent). The goal is to identify features that are related to churn (the target), identify any data integrity issues, and understand 'how the data works'. If there appears to be some sort of interaction or correlation, assume there is no causal relationship and brainstorm (and document) ideas on reasons there could be correlation.
> - Summarize my conclusions, provide clear answers to my specific questions, and summarize any takeaways/action plan from the work above.
___

##### Plan -> Acquire -> Prepare -> Explore -> Model
> - Feature Selection and Encoding: Are there any variables that seem to provide limited to no additional information? If so, remove them.  Also encode any non-numerical features of signifigance.
> - Establish a baseline accuracy to determine if having a model is better than no model and train and compare at least 4 different models.
> - Train (fit, transform, evaluate) multiple models, varying the algorithm and/or hyperparameters you use.
> - Compare evaluation metrics across all the models you train and select the ones you want to evaluate using your validate dataframe.  In this case we used Precision (Positive Predictive Value).
> - Based on the evaluation of the models using the train and validate datasets, choose the best model to try with the test data, once.
> - Test the final model on the out-of-sample data (the testing dataset), summarize the performance, interpret and document the results.
___

##### Plan -> Acquire -> Prepare -> Explore -> Model -> Deliver
> - Introduce myself and my project goals at the very beginning of my notebook walkthrough.
> - Summarize my findings at the beginning like I would for an Executive Summary.
> - Walk the management team through the analysis I did to answer my questions and that lead to my findings. (Visualize relationships and Document takeaways.) 
> - Clearly call out the questions and answers I am analyzing as well as offer insights and recommendations based on my findings.

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Reproduce My Project

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md
- [ ] Download the aquire.py, wrangle.py, splitter.py, evaluate.py, explore.py, model_comparator.py and final_report.ipynb files into your working directory.
- [ ] For more details on the analysis, in particular clustering charts, download the working_notebook.ipynb file as well as the EDA_focus.ipynb.
- [ ] Add your own env file to your directory. (user, password, host)
- [ ] Run the final_report.ipynb notebook

##### Credit to Faith Kane (https://github.com/faithkane3) for the format of this README.md file.