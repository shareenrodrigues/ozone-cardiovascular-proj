# Ozone and Cardiovascular Disease in California

**Author:** Shareen Rodrigues  
**Course:** MATH 261A: Regression Theory and Methods  
**Institution:** San José State University  
**Instructor:** Professor Peter Gao  
**Date of Submission:** September 25, 2026

## Description

This project investigates whether ambient ozone concentration predicts 
cardiovascular disease emergency department visit rates across California 
census tracts using simple linear regression.

**Research question:** Does ambient ozone concentration predict 
cardiovascular disease ED visit rates across California census tracts?

## Project Structure
ozone-cardiovascular-project/
├── report.qmd # Main Quarto report (source)
├── report.pdf # Rendered PDF report
├── linear_regression_fit.R #R script for the analysis
├── README.md # This file
└── data/ # Not included — see Data section below


## Data

Data were obtained from CalEnviroScreen 4.0 (OEHHA, 2021), a publicly 
available dataset produced by the California Office of Environmental 
Health Hazard Assessment (OEHHA).

**License:** CalEnviroScreen 4.0 is a public domain dataset produced by 
a California state agency and is freely available for public use. It can 
be downloaded from:  
https://oehha.ca.gov/calenviroscreen/report/calenviroscreen-40

The data file is not included in this repository due to file size. 
Download the Excel file from the link above and place it in a `data/` 
folder to reproduce the analysis.

## Reproducing the Analysis

1. Download the CalEnviroScreen 4.0 Excel file and place it in `data/`
2. Open `report.qmd` in RStudio
3. Click **Render** (or run `quarto render report.qmd`)

**Required R packages:** tidyverse, readxl, janitor, here, broom

## AI Use Disclosure

Claude (Anthropic) was used as a writing assistant during the drafting 
of this report. Claude provided feedback on section drafts, identified 
formatting and code errors, and explained Quarto and R syntax. 