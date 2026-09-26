install.packages("tidyverse")
install.packages("readxl")
install.packages( "janitor")
install.packages("here")
install.packages( "broom")

library(tidyverse)
library(readxl)
library(janitor)
library(here)
library(broom)

#load data
calEnvData <- read_excel(
  here(
    "data",
    "calenviroscreen40resultsdatadictionary_F_2021.xlsx"
  ),
  sheet = "CES4.0FINAL_results"
) |>
  clean_names()

#take only required variables
analysisData <- calEnvData |>
  select(
    census_tract,
    california_county,
    total_population,
    ozone,
    cardiovascular_disease
  )

#clean data with missing values
analysisData |>
  summarise(
    total_rows = n(),
    missing_ozone = sum(is.na(ozone)),
    missing_cardiovascular = sum(is.na(cardiovascular_disease))
  )

analysisData <- analysisData |>
  drop_na(ozone, cardiovascular_disease)

#data summary
descriptiveTable <- tibble(
  Variable = c(
    "Ozone concentration (ppm), 2017–2019",
    "Heart attack ED visits per 10,000, 2015–2017"
  ),
  `Sample size` = c(
    nrow(analysisData),
    nrow(analysisData)
  ),
  Mean = c(
    mean(analysisData$ozone),
    mean(analysisData$cardiovascular_disease)
  ),
  `Standard deviation` = c(
    sd(analysisData$ozone),
    sd(analysisData$cardiovascular_disease)
  ),
  Minimum = c(
    min(analysisData$ozone),
    min(analysisData$cardiovascular_disease)
  ),
  Median = c(
    median(analysisData$ozone),
    median(analysisData$cardiovascular_disease)
  ),
  Maximum = c(
    max(analysisData$ozone),
    max(analysisData$cardiovascular_disease)
  )
)

knitr::kable(
  descriptiveTable,
  digits = 4,
  align = "lrrrrrr"
)

cor(analysisData$ozone, analysisData$cardiovascular_disease)

#lm fit plot
ggplot(
  analysisComplete,
  aes(
    x = ozone,
    y = cardiovascular_disease
  )
) +
  geom_point(
    alpha = 0.25,
    color = "steelblue"
  ) +
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = TRUE,
    color = "darkred"
  ) +
  labs(
    title = "Ozone and Cardiovascular Emergency Department Visits",
    subtitle = "California census tracts in CalEnviroScreen 4.0",
    x = "Average daily maximum 8-hour ozone concentration",
    y = "Cardiovascular ED visits per 10,000 people"
  ) +
  theme_minimal()

#rescale ozone
analysisData <- analysisData |>
  mutate(ozone_01 = ozone / 0.01)

#linear model fit
ozoneModel <- lm(
  cardiovascular_disease ~ ozone_01,
  data = analysisComplete
)

#Estimated simple linear regression coefficients
regressionTable <- tidy(
  ozoneModel,
  conf.int = TRUE,
  conf.level = 0.95
) |>
  mutate(
    term = recode(
      term,
      `(Intercept)` = "Intercept",
      ozone_01 = "Ozone increase of 0.01 ppm"
    ),
    estimate = round(estimate, 4),
    std.error = round(std.error, 4),
    statistic = round(statistic, 3),
    p.value = format.pval(
      p.value,
      digits = 3,
      eps = 0.001
    ),
    conf.low = round(conf.low, 4),
    conf.high = round(conf.high, 4)
  ) |>
  select(
    term,
    estimate,
    std.error,
    statistic,
    p.value,
    conf.low,
    conf.high
  ) |>
  rename(
    Term = term,
    Estimate = estimate,
    `Standard error` = std.error,
    `t statistic` = statistic,
    `p-value` = p.value,
    `95% CI lower` = conf.low,
    `95% CI upper` = conf.high
  )

knitr::kable(
  regressionTable,
  align = "lrrrrrr"
)

#Simple linear regression model fit statistics
modelFitTable <- glance(ozoneModel) |>
  transmute(
    `Sample size` = nobs,
    `R-squared` = round(r.squared, 4),
    `Adjusted R-squared` = round(adj.r.squared, 4),
    `Residual standard error` = round(sigma, 4),
    `F statistic` = round(statistic, 3),
    `Model p-value` = format.pval(
      p.value,
      digits = 3,
      eps = 0.001
    )
  )

knitr::kable(
  modelFitTable,
  align = "rrrrrr"
)

#Diagnostic plots
diagnosticData <- augment(ozoneModel)

#Residual vs Fitted plot
ggplot(
  diagnosticData,
  aes(x = .fitted, y = .resid)
) +
  geom_point(
    alpha = 0.20,
    color = "steelblue"
  ) +
  geom_hline(
    yintercept = 0,
    color = "gray40",
    linetype = "dashed"
  ) +
  geom_smooth(
    method = "loess",
    formula = y ~ x,
    se = FALSE,
    color = "darkred",
    linewidth = 1
  ) +
  labs(
    x = "Fitted cardiovascular ED visit rate",
    y = "Residual"
  ) +
  theme_minimal()

#Normal Q-Q plot
ggplot(
  diagnosticData,
  aes(sample = .std.resid)
) +
  stat_qq(
    alpha = 0.25,
    color = "steelblue"
  ) +
  stat_qq_line(
    color = "darkred",
    linewidth = 1
  ) +
  labs(
    x = "Theoretical normal quantiles",
    y = "Standardized residuals"
  ) +
  theme_minimal()
