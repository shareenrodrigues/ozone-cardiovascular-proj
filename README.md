# Association between Ozone and Cardiovascular Disease in California

**MATH 261A: Regression Theory and Methods**  
San José State University | Fall 2026  
Instructor: Professor Peter Gao

## Overview

This project investigates whether ambient ozone concentration predicts 
cardiovascular disease emergency department visit rates across California 
census tracts using simple linear regression.

**Research question:** Does ambient ozone concentration predict 
cardiovascular disease ED visit rates across California census tracts?

## Data

CalEnviroScreen 4.0 (OEHHA, 2021) — census tract-level environmental 
and health data for California.

- **Predictor:** Daily maximum 8-hour average ozone (ppm), May–October 2017–2019
- **Outcome:** Age-adjusted ED visit rate for heart attacks per 10,000 residents, 2015–2017
- **Sample size:** 7,924 census tracts (after removing missing values)

Data file not included in this repository due to size. Download from:  
https://oehha.ca.gov/calenviroscreen/report/calenviroscreen-40

## Reproducing the Analysis

1. Download the CalEnviroScreen 4.0 Excel file and place it in `data/`
2. Open `report.qmd` in RStudio
3. Click **Render** (or run `quarto render report.qmd`)

**Required R packages:** tidyverse, readxl, janitor, here, broom

## AI Use Disclosure

Claude was used as a writing assistant during the drafting 
of this report. Claude provided feedback on section drafts, identified 
formatting and code errors, and explained Quarto/R syntax. 