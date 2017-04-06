---
title: "Red Wine Quality Prediction"
output:
  html_document: default
  html_notebook: default
---

by Jie Hu, Email: [jie.hu.ds@gmail.com](mailto:jie.hu.ds@gmail.com)


```{r packages}
library(GGally)  # scatter matrix
library(scales)
library(memisc)  # summarize regression
library(lattice)
library(MASS)
library(car)     # recode variable
library(reshape2) # wrangle data
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

# Here're labels will be used in plots

FIXED_ACIDITY_LABEL = "Fixed Acidity (g/" ~dm^3~")"
VOLATILE_ACIDITY_LABEL = "Volatile Acidity (g/" ~dm^3~")"
CITRIC_ACID_LABEL = "Citric Acid (g/" ~dm^3~")"
RESIDUAL_SUGAR_LABEL = "Residual Sugar (g/" ~dm^3~")"
CHLORIDES_LABEL = "Chlorides (g/" ~dm^3~")"
FREE_SULFUR_DIOXIDE_LABEL = "Free surful dioxide (mg/" ~dm^3~")"
TOTAL_SULFUR_DIOXIDE_LABEL = "Total surful dioxide (mg/" ~dm^3~")"
DENSITY_LABEL = "Density (g/" ~cm^3~")"
PH_LABEL = "pH"
SULPHATES_LABEL = "Sulphates (g/" ~dm^3~")"
QUALITY_LABEL = "Quality Score"
ALCOHOL_LABEL = "Alcohol (%)"
```

This markdown will use explorsive data analysis to figure out which attributes affect quality of red wine significantly. To do this, I use the [dataset](https://s3.amazonaws.com/udacity-hosted-downloads/ud651/wineQualityReds.csv) including the quality rate by at least 3 experts and the chemical properties of the wine. This dataset might indicate how current experts, representing the test nowadays, think what a good red wine is.

```{r echo=FALSE, Load_the_Data}
# Load the Data
wine_data <- read.csv("wineQualityReds.csv")
wine_data <- subset(wine_data, select = -c(X))
```

# Univariate Plots Section

## Explore part

To begin with, let's summarise the data:

```{r}
str(wine_data)
```

The dataset includes 1599 observations with  12 variables.

Then let's explot the variables one by one.
Because all the variables are numeric, I will mainly use histogram to explore and figure out if there're something interesting worth further steps.

```{r}
hist_plot <- function(variable, 
                      data, 
                      label,
                      title,
                      binwidth = 0.1, 
                      scale_xlab = 1) {
  MIN = min(variable)
  MAX = max(variable)
  
  plot <- ggplot(aes(variable), data = data) +
          geom_histogram(binwidth = binwidth, fill = "#01cab4") +
          labs(x = label) +
          ggtitle(title) +
          coord_cartesian(xlim = c(MIN,MAX)) +
          scale_x_continuous(breaks = seq(MIN, MAX, scale_xlab))

  return(plot)
}
```

**- quality -**

Quality is what this report concerns with, the rate here represent average rate from at least 3 experts. First let's see how the quality of 1599 wine distributed.

```{r}
hist_plot(variable = wine_data$quality, 
          data = wine_data, 
          label = QUALITY_LABEL, 
          title = "Histogram of Quality",
          binwidth = 0.5, 
          scale_xlab = 1)
```

```{r}
summary(wine_data$quality)
```
 
```{r}
table(wine_data$quality)
```

Quality, ranging from 3-8, is integer type data. About 82.5% observations get 5-6 ratings, while only 14.2% (227 counts) got 3,7 or 8 scores on quality rating. Because the score were average made by 3 or more experts and I assume it's trustworthy. 

**-fixed.acidity-**
  
```{r, echo=FALSE}
hist_plot(variable = wine_data$fixed.acidity, 
           data = wine_data, 
           title = "Histogram of Fixed Acidity",
           FIXED_ACIDITY_LABEL, 
           binwidth = 0.1, 
           scale_xlab = 0.5)
```

```{r, echo=FALSE}
summary(wine_data$fixed.acidity)
```


"fixed.acidity" is a measure of inside liquid concentration. 
The histogram a little right-skewed distributed with some outliers located at right side. The most frequent values are between 7-8. IQR is 2.1.

**-volatile.acidity-**

  
```{r, echo=FALSE}
hist_plot(variable = wine_data$volatile.acidity,
           data = wine_data, 
           label = VOLATILE_ACIDITY_LABEL, 
           title = "Histogram of Volatile Acidity",
           binwidth = 0.1, 
           scale_xlab = 0.1)
```

```{r, echo=FALSE}
summary(wine_data$volatile.acidity)
```


"volatile.acidity" is measure of acidity above-surface of liquid.
The histogram is right-skewed distributed with some outliers located at right side. The most frequent values are between 0.4-0.6. IQR is 0.25.

**-citric.acid-**

  
```{r, echo=FALSE}
hist_plot(variable = wine_data$citric.acid,
           data = wine_data, 
           label = CITRIC_ACID_LABEL, 
           title = "Histogram of Citric Acid",
           binwidth = 0.01, 
           scale_xlab = 0.1)
```

```{r, echo=FALSE}
summary(wine_data$citric.acid)
```


"citric.acid" is right-skewed distributed with some outliers located at right side. 
The most frequent values 0. It's also interesting a lot of wine have citric.acid = 0, IQR is 0.33.

**-residual.sugar-**

  
```{r, echo=FALSE}
hist_plot(variable = wine_data$residual.sugar,
           data = wine_data, 
           label = RESIDUAL_SUGAR_LABEL,
           title = "Histogram of Residual Sugar",
           binwidth = 0.1, 
           scale_xlab = 1)
```

```{r, echo=FALSE}
summary(wine_data$residual.sugar)
```


"residual.sugar" is right-skewed distributed with a lot of outliers located at right side. 
The most frequent values are between 1.9-2.4. IQR is 1.7.

**-chlorides-**

  
```{r, echo=FALSE}
hist_plot(variable = wine_data$chlorides,
           data = wine_data, 
           label = CHLORIDES_LABEL,
           title = "Histogram of Chlorides",
           binwidth = 0.01, 
           scale_xlab = 0.05)
```

```{r, echo=FALSE}
summary(wine_data$chlorides)
```


"chlorides" is right-skewed distributed with a lot of outliers located at right side. 
The most frequent values are between 0.062-0.112. IQR is 0.02.

**-free.sulfur.dioxide-**

  
```{r, echo=FALSE}
hist_plot(variable = wine_data$free.sulfur.dioxide,
           data = wine_data, 
           label = FREE_SULFUR_DIOXIDE_LABEL, 
           title = "Histogram of Free Sulfur Dioxide",
           binwidth = 1, 
           scale_xlab = 5)
```

```{r, echo=FALSE}
summary(wine_data$free.sulfur.dioxide)
```


"free.sulfur.dioxide" is right-skewed distributed with a lot of outliers located at right side. 
The most frequent values are between 5-8. IQR is 14.
Notice the number of free sulfur is larger than other ingredients like acidity, it's because of different unit is applied. Sulfur is using $g/dm^3$, while acidity variables are using $mg/dm^3$.
Actually, wine contains much less sulfur (free or total) than other ingredients.

**-total.sulfur.dioxide-**

  
```{r, echo=FALSE}
hist_plot(variable = wine_data$total.sulfur.dioxide,
           data = wine_data, 
           label = TOTAL_SULFUR_DIOXIDE_LABEL, 
           title = "Histogram of Total Sulfur Dioxide",
           binwidth = 5, 
           scale_xlab = 15)
```

```{r, echo=FALSE}
summary(wine_data$total.sulfur.dioxide)
```


"total.sulfur.dioxide" is right-skewed distributed with some outliers located at right side. 
The most frequent values are between 15-25. IQR is 40.
Notice wine contains a log of total sulfur dioxide being compared with other ingredients, even more than free sulfur.
It's reasonable total sulfur should be more than free sulfur because conceptually, free sulfur is part of total surful.

**-density-**

  
```{r, echo=FALSE}
hist_plot(variable = wine_data$density,
           data = wine_data, 
           label = DENSITY_LABEL,
           title = "Histogram of Density",
           binwidth = 0.0005, 
           scale_xlab = 0.002)
```

```{r, echo=FALSE}
summary(wine_data$density)
```


"density" is approximately symmetric, and it's surprising the difference among different wines, though they might test significantly different, are not that big.

**-pH-**

  
```{r, echo=FALSE}
hist_plot(variable = wine_data$pH, 
           data = wine_data, 
           label = PH_LABEL,
           title = "Histogram of pH",
           binwidth = 0.1, 
           scale_xlab = 0.1)
```

```{r, echo=FALSE}
summary(wine_data$pH)
```

"pH" is almost symmetric. The most frequent value is between 3.24 and 3.44, IQR is 0.19. It's not a big difference.
But one thing interests is that what's the quality of the wines with lowest and most value of pH? Will be discussed later.


**-sulphates-**

  
```{r, echo=FALSE}
hist_plot(variable = wine_data$sulphates, 
           data = wine_data, 
           label = SULPHATES_LABEL, 
           title = "Histogram of sulphates",
           binwidth = 0.1, 
           scale_xlab = 0.1)
```

```{r, echo=FALSE}
summary(wine_data$sulphates)
```

"sulphates" is right-skewed distributed with some outliers located at right side. The most frequent values are between 0.5-0.7. IQR is 0.18.



**-alcohol-**
  
```{r, echo=FALSE}
hist_plot(variable = wine_data$alcohol, 
           data = wine_data, 
           label = ALCOHOL_LABEL,
           title = "Histogram of Alcohol",
           binwidth = 0.1, 
           scale_xlab = 0.5)
```

```{r, echo=FALSE}
summary(wine_data$alcohol)
```

"alcohol" is right-skewed distributed with some outliers located at right side. The most frequent values are between 9.4-9.6. IQR is 1.6.


## Question answer part
### What is the structure of your dataset?

The red wine quality dataset include 1599 observations and 12 variables. All attributes are numeric, 11 of them are continuous test result and 1, the quality, is rating of integers ranging from 3 to 8. It's pretty tidy and none of attributes have NA values. 

### What is/are the main feature(s) of interest in your dataset?
After take a look at the attributes of the dataset, I found the variables like pH, alcohol, sulphates etc. most interesting to me, because I leared some red wine quality determinant before and do hope to explore how these variables distributed and how they related to wine quality.


### What other features in the dataset do you think will help support your investigation into your feature(s) of interest?
By the above correlation matrix, volatile.acidity, sulphates and alcohol are the attributes most coorelated with quality of wine. Thus, these 3 attributes are most attractive to me.


### Did you create any new variables from existing variables in the dataset?
I haven't create any new features so far. But I will create in below analysis.

### Of the features you investigated, were there any unusual distributions? Did you perform any operations on the data to tidy, adjust, or change the form of the data? If so, why did you do this?
All attributes have outliers with extreme value. So far I haven't remove any data because I want to keep all data in first stage. The next step, when I look into relationship between 2 attributes, I will remove outliers if necessary.


# Bivariate Plots Section

Then I create correlation matrix to figure out which attributes are worth further exploring.

```{r, echo=FALSE, message=FALSE}
library(corrplot)
corrplot( cor(wine_data), 
          method ="number", 
          type ="lower",
          tl.col ='black', 
          tl.cex=1, 
          col = colorRampPalette( c("#01cab4","white","#ff3f49") )(200),
          tl.srt = 40)
```

From this plot, we can see some pair of attributes associate with each other. For example, citric.acidity associates positively with fixed.acidity, while pH negatively associates with fixed.acidity. However, there seems no attributes have strong correlation with quality. 

Here, alcohol, volatile.acidity, sulphates are top3 attributes that associated with quality. So let's explore further on quality and these 3 variables.

Now let's explore how these 3 attributes interact with quality. 
```{r}
boxplot_fun <- function(data,
                        y_var,
                        title,
                        ylab){
  
  plot <- ggplot(data = data, aes(y = y_var, 
                      x = quality)) +
  geom_boxplot(aes(group = quality, fill = quality)) +  
  ggtitle(title) + 
  xlab(QUALITY_LABEL) +
  ylab(ylab) +
  scale_x_continuous(breaks = seq(3,8,1))
  
  return(plot)
} 
```


**-quality vs. alcohol-**

```{r, echo=FALSE}
boxplot_fun(data = wine_data,
            y_var = wine_data$alcohol,
            title = "Alcohol vs. Quality",
            ylab = ALCOHOL_LABEL)
```

It seems there's positive relationship between alcohol and quality. High quality wines are more likely to have high percentage of alcohol.
The corelation coeffecient $R^2 = 0.2263$, which means alcohol can explain only 22.63% the variation of quality.

```{r, echo=FALSE}
summary(lm(data = wine_data, 
           alcohol~quality))
```


**-quality vs. volatile.acidity-**

To avoid overplotting, I add transparency in this plot

```{r, echo=FALSE}

boxplot_fun(data = wine_data,
            y_var = wine_data$volatile.acidity,
            title = "Volatile Acidity vs. Quality",
            ylab = VOLATILE_ACIDITY_LABEL)

```

Now we can see a negative association between these two attributes. While alcohol is increasing with quality, volatile acidity is negatively associated with quality. The corelation coeffecient $R^2 = 0.152$, which means volatile.acidity can explain only 15.2% the variation of quality.

```{r, echo=FALSE}
summary(lm(data = wine_data, quality~volatile.acidity))
```

**-sulphates vs. quality-**

```{r, echo=FALSE}
boxplot_fun(data = wine_data,
            y_var = wine_data$sulphates,
            title = "Sulphates vs. Quality",
            ylab = SULPHATES_LABEL)

```

We can see a positive association between these two attributes. The corelation coeffecient $R^2 = 0.06261$, which means sulphates can explain only 6.26% the variation of quality.

```{r, echo=FALSE}
summary( lm(data = wine_data, 
         quality ~ sulphates))
```

From the matrix, we can also see there're corelated attributes:

```{r, echo=FALSE}

gplot_function <- function(var1, var2, data, xlabel, ylabel, title){
  
  plot <- ggplot(data = data, aes(var1, var2)) +
        geom_point(color = "#01cab4") +
        xlab(xlabel) +
        ylab(ylabel) +
        geom_smooth(method = "lm") +
        ggtitle(title)
  
  return(plot)
}
```


```{r}
gplot_function(var1 = wine_data$citric.acid, 
               var2 = wine_data$fixed.acidity, 
               data = wine_data,
               xlabel = CITRIC_ACID_LABEL,
               ylabel = FIXED_ACIDITY_LABEL,
               title = "Relationship between Citric Acid and Fixed Acidity")
```

This positive association can be due to the fact that citric acid provides solid support to fixed acidity.


```{r}
gplot_function(var1 = wine_data$citric.acid, 
               var2 = wine_data$volatile.acidity, 
               data = wine_data,
               xlabel = CITRIC_ACID_LABEL,
               ylabel = VOLATILE_ACIDITY_LABEL,
               title = "Relationship between Citric Acidity and Volatile Acidity")

```

Volatile acid, however, is negatively associated with citric acid because citric acid rarely volatilize. You can test it but can hardly smell.


```{r}

gplot_function(var1 = wine_data$fixed.acidity, 
               var2 = wine_data$pH, 
               data = wine_data,
               xlabel = FIXED_ACIDITY_LABEL,
               ylabel = PH_LABEL,
               title = "Relationship between Fixed Acidity and pH")

```

Acidity inside liquid, the wine, will certaily decrease pH. The smaller the value of pH is, the liquid becomes more acid.

```{r}
gplot_function(var1 = wine_data$density, 
               var2 = wine_data$fixed.acidity, 
               data = wine_data,
               xlabel = DENSITY_LABEL,
               ylabel = FIXED_ACIDITY_LABEL,
               title = "Relationship between Density and Fixed Acidity")

```

Acidity is provided by acid mocules with heavier weight than water and alcohol because these acid moleculars are resolved as ions wondering among the space of water moleculars.

Next, there're some questions I'm interested in.

1. How's quality of wine with no citric.acid
Recall that we find there's over 150 observations with citric.acid value equal to 0, we have adequate data to explore:

```{r}
noCitric_wines <- subset(wine_data, citric.acid==0)
hist_plot(data = noCitric_wines, 
          variable = noCitric_wines$quality, 
          label = QUALITY_LABEL, 
          title = "Histogram of No-citric wines Quality", 
          binwidth = 0.5, 
          scale_xlab = 1)

```

The distribution is quite similar to overall wine data. Now let's make boxplots to compare the quality of two groups: with/without citric acid.

```{r}
wine_data$has_citric <- (wine_data$citric.acid != 0)

ggplot(data = wine_data, aes(x = has_citric, y = quality)) +
  geom_boxplot(aes(group = has_citric, fill = has_citric)) +
  xlab("Has Citric Acid") +
  ylab("Quality") +
  ggtitle("Quality compare between wines with / without Citric Acid") +
  guides(fill= FALSE)

```

Though the median is different, the plots looks quite similar. Whether having citric acid seems do not affect quality significantly.

2. How pH without or without citric acid? And how's quality of wines with the most extremely acid pH?

```{r}
ggplot(data = wine_data, aes(x = has_citric, y = pH)) +
  geom_boxplot(aes(group = has_citric, fill = has_citric)) +
  xlab("Has Citric Acid") +
  ylab(PH_LABEL) +
  ggtitle("pH compare between wines with / without Citric Acid") +
  guides(fill= FALSE)
```

We can see the wines with citric acid have lower pH. But we don't know if extreme pH will affect quality. If yes, citric acid can be a indirect factor to improve quality of wine, because from the above correlation matrix, we can see, beside fixed acid, citric is the biggest factor associates with pH.

```{r}
# create column to see if pH is less than 3
wine_data$pH_lt3 <- wine_data$pH < 3
table(wine_data$pH_lt3)
```

There're 29 wines with pH less than 3 (A strong acid level!).

```{r}
ggplot(data = wine_data, aes(x = pH_lt3, y = quality)) +
  geom_boxplot(aes(group = pH_lt3, fill = pH_lt3)) +
  xlab("pH < 3") +
  ylab(QUALITY_LABEL) +
  ggtitle("Quality compare between wines with / without pH < 3") +
  guides(fill= FALSE)
```

This is really frustrating. :(
However, we learn extreme pH doesn't affect quality that much.

3. Are wines with higher residual sugar indicating lower quality?

"Sweetness is happiness!"

According to our experience, food with more sugar will be more attractive. Like cake, cola, and even some of the sweet wines.
Does this happen in red wine? 

Let's compare quality by cut data into different level of residual sugar, and then plot quality boxcharts in gourp of sugar level:

```{r}
wine_data$sugar_level <- cut(wine_data$residual.sugar,  
                             breaks = c(0,6,12,18),
                             labels = c("Low","Medium","High"),
                             include.lowest = TRUE, 
                             ordered = TRUE)

ggplot(wine_data, aes(x = sugar_level, y = quality)) +
  geom_boxplot(aes(fill = sugar_level))
```

The plot tells us the relationship between sugarlevel and quality is not very strong. 
From the correlation, we can draw same conclusion.

```{r}
cor(wine_data$residual.sugar, wine_data$quality)
```

Let's refresh these interesting discoveries in this section:
- Increasing fixed.acidity will lead to decreasing pH because more hydrogen ion appears
- Citric.acidity is a kind of acid without volatile, so it's reasonable when citric.acidity increase, fixed acidity will increase. As total acidity is divided into two groups, namely the volatile acids and the nonvolatile or fixed acids. So it's reasonable when Citric.acidity increases, volatile will decrease
- when the total amount of acid increases, density will increase because water molecule is much lighter than these acid ion
- Whether having citric acid is not a significant indicator of quality
- Extreme pH is not an indicator of wine quality
- Residual sugar is not a significant indicator, too 

# Bivariate Analysis

Before further analysis, it's necessary to remove outliers because these outliers might bias our model. Because data are distributed right-skewed in the dimentions of these 3 attrbutes, I will remove top 1% data. 


```{r, echo=FALSE}
# remove top 1% outliers
wine_data.improved <- subset(wine_data, 
                             volatile.acidity < quantile(
                                           wine_data$volatile.acidity, 0.99) & 
                             sulphates < quantile(
                                           wine_data$sulphates, 0.99) &
                             alcohol < quantile(
                                          wine_data$alcohol, 0.99))

summary(wine_data.improved[,c(2,10,11)])
```

After this, I removed 52 observations. Now we have the data which has all maximum of 3 main attributes close to its 3rd quantile. The correlation between quality and one of the main attributors are:

```{r, echo=FALSE}
cor(wine_data.improved[,c(2, 10,11)], 
    y = wine_data.improved$quality)^2
```

Now, I apply the model:

```{r}
m1 <- lm(data = wine_data.improved, quality ~ alcohol)
m2 <- update(m1, ~.+volatile.acidity)
m3 <- update(m2, ~.+ sulphates)

mtable(m1, m2, m3, sdigits = 3)
```

Even the linear model with all 3 most significant predictors, it can count for roughtly 33.7% the variation on quality. 


### Talk about some of the relationships you observed in this part of the investigation. How did the feature(s) of interest vary with other features in the dataset?

Features like alcohol, volatile.acidity and sulphates have relatively stronger (though below moderate level), while other attributes have quite small or even nearly no relationship with quality.

The pH analysis and citric analysis parts are both frustrating to me, I find both are not significant indicators to wine quality even with extreme values. This is to my great surprise.

### Did you observe any interesting relationships between the other features (not the main feature(s) of interest)?

Yes, I listed 4 pairs of corelated attributes in above analysis:
1. citric.acid ~ fixed.acidity, positively associated
2. citric.acid ~ volatile.acidity, negatively associated
3. fixed.acidity ~ pH, negatively associated
4. density ~ fixed.acidity, positively associated


### What was the strongest relationship you found?

pH and fixed acidity is the pair with strongest relationship ($R = -0.68$) I found.

Besides, 2 pairs have strong relationship with $R = 0.67$:
citric.acid ~ fixed.acidity and density ~ fixed.acidity

All these three pairs reach moderate level of association.

# Multivariate Plots Section

Before doing plot, I firstly make quality as factor with levels "Low"(1-3), "Moderate"(4-6), "High"(7-8).
And because the moderate quality has much more variation, here I only consider low/high quality in plots of this section.

```{r, echo=FALSE}
wine_data.improved$quality_level <- cut(wine_data.improved$quality,  
                                        breaks = c(3,4,6,8),
                                        labels = c("Low","Medium","High"),
                                        include.lowest = TRUE)


wine_data.noModerate <- subset(wine_data.improved, quality_level %in% c("Low", "High"))

ggplot(aes(alcohol, 
           volatile.acidity), 
       data = wine_data.noModerate) +
  geom_point(aes(color = quality_level), size = 3) +
  ggtitle("Distribution of quality by volatile.acidity + alcohol") +
  ylab(VOLATILE_ACIDITY_LABEL) +
  xlab(ALCOHOL_LABEL) 

```

There's apparent pattern that high quality wine will more likely to fall into bottom right side of this graph, which means higher alcohol level plus lower volitile acidity will likely to indicate a better red wine.

Now I will exam another 4 groups of attributes of great interests (Want to check if anecdotes I heard are right): 

- Quality, pH and Density


```{r, echo=FALSE}

wine_data.improved.subset <- subset(wine_data.improved, 
                                    quality_level %in% c("Low","High"))

multi_plot <- function(data = wine_data.improved.subset,
                       var1,
                       var2,
                       title,
                       xlab,
                       ylab) {
  plot <- ggplot(aes(var1, 
                 var2), 
                 data = data) +
            geom_point(aes(color = quality_level),
                       size = 3) +
            ggtitle(title) +
            xlab(xlab) +
            ylab(ylab)
  return(plot)
}

multi_plot(var1 = wine_data.improved.subset$pH, 
           var2 = wine_data.improved.subset$density,
           title = "Quality, pH and Density",
           xlab = PH_LABEL,
           ylab = DENSITY_LABEL)
```

This plot indicate that low quality red wines are more likely to have higher pH and high density together. Though this difference might not be obvious.


- Quality, Chlorides and Sulphates

```{r}
multi_plot(var1 = wine_data.improved.subset$chlorides, 
           var2 = wine_data.improved.subset$sulphates,
           title = "Quality, Chlorides and Sulphates",
           xlab = CHLORIDES_LABEL,
           ylab = SULPHATES_LABEL)
```

Under the conclusion that chlorides doesn't affect quality, high quality wines are more likely to have high sulphates. The reason is that sulphates is preservative which can maintain a wine's freshness because of its antioxidant and antibacterial properties.

- Quality, Residual Sugar and Alcohol

```{r}
multi_plot(var1 = wine_data.improved.subset$residual.sugar, 
           var2 = wine_data.improved.subset$alcohol,
           title = "Quality, Residual Sugar and Alcohol",
           xlab = RESIDUAL_SUGAR_LABEL,
           ylab = ALCOHOL_LABEL)

```

As residual sugar doesn't affect quality, alcohol is main indicator of wine quality - higher-quality wines are more likely having higher percentage alcohol.

- Quality, Fixed Acidity and Chlorides

```{r}
multi_plot(var1 = wine_data.improved.subset$fixed.acidity, 
           var2 = wine_data.improved.subset$chlorides,
           title = "Quality, Fixed Acidity and Chlorides",
           xlab = FIXED_ACIDITY_LABEL,
           ylab = CHLORIDES_LABEL)

```

As chlorides doesn't affect quality a lot, higher-quality wines are more likely to have higher fixed acidity.

-Quality, Citric Acid and Density-
```{r}
multi_plot(var1 = wine_data.improved.subset$citric.acid, 
           var2 = wine_data.improved.subset$density,
           title = "Quality, Citric Acid and Density",
           xlab = CITRIC_ACID_LABEL,
           ylab = DENSITY_LABEL)

```

Citric Acid and Density are the 2 most significant indicators for quality. Here I plot them together.
From the chart we can see higher-quality wines are more likely to have higher density and citric acid together, which proof my conclusion above.

# Multivariate Analysis

### Talk about some of the relationships you observed in this part of the investigation. Were there features that strengthened each other in terms of looking at your feature(s) of interest?

From these plots, we can see:
- Higher pH, higher sulphates, higher alcohol and higher fixed acidity are more likely to indicate red wines with higher quality
- it's hard to judge quality by density, chlorides and residual sugar values

### Were there any interesting or surprising interactions between features?
It's surprising these pairs of attributes are almost independent to each other, we can either see horizontal / vertical pattern or the data point massed up.


### OPTIONAL: Did you create any models with your dataset? Discuss the strengths and limitations of your model.

Yes, in above analysis, I created a linear model to see how the attributes like alcohol, sulphates, and volatile.acidity interact with quality. 

The pros of this model include:
- it's straightforward and self-explained what kind of wine will be with high quality
- easy to plot and predict

However, there're cons:
- even with most significant attributes, the model can explain only 33.6% variation of quality, which means over 66.3% variation is out of control, which might make this model unreliable
- the data is distributed in a very complicated pattern, applying linear model might be a naive choice to get rid of too much information

To better improve the result, both advanced model technics and more dimentional data are required, for example, how each score of quality such red wine gets.

------

# Final Plots and Summary

### Plot One

From above analysis, we can see alcohol is the strongest attribute to predict quality. To better show this positive association, here I add mean, median, quatile line to above plot.

```{r echo=FALSE, Plot_One}
ggplot(wine_data.improved, aes(y = alcohol, x = quality)) +
  # add noisy to data
  geom_jitter(alpha = 0.25, color = '#ff3f49', 
              position = position_jitter(h=0, width = 2),
              size = 3) +    
  
  ggtitle("Alcohol vs. Quality (Improved)") + 
  
  # add mean line
  geom_path(stat = "summary", 
            fun.y = mean, 
            linetype=1, 
            aes(color="mean"), 
            size = 2) +
  
  # add 1st quantile
  geom_path(stat = "summary", 
            fun.y = quantile, 
            fun.args = list(probs = 0.25), 
            linetype = 2, 
            aes(color="quartiles"), 
            size = 1.5) +
  
  # add median line
  geom_path(stat = "summary", 
            fun.y = quantile, 
            fun.args=list(probs=.5), 
            linetype=2, 
            aes(color="median"), 
            size = 1) +
  
  # add 3rd quantile
  geom_path(stat = "summary", 
            fun.y = quantile, 
            fun.args = list(probs =.75), 
            linetype = 2, 
            aes(color = "quartiles"), 
            size = 1.5) +
  
  scale_colour_manual("Legend", 
                      values = c("mean"="#45cf13", 
                                 "median"="#006775", 
                                 "quartiles"="#01cab4")) +
  labs(x = QUALITY_LABEL, 
       y = ALCOHOL_LABEL, 
       title ="Alcohol vs. Quality") +
  scale_x_continuous(breaks = seq(0,10,1))

```

### Description One

From the plot, we can see though the data is lying everywhere, there's a pattern can be drawn that the quality of red wine will increase with alcohol percentage.

### Plot Two

From the discussion on correlated attributes, I found, density is highly positively associated with fixed.acidity and residual.sugar, meanwhile highly negatively associates with alcohol, so now let's see if density can be expressed as combination of these 3 attributes:


```{r echo=FALSE, Plot_Two}
attach(wine_data.improved)
wine_data.improved$prod <- wine_data.improved$fixed.acidity * 
                           wine_data.improved$residual.sugar / 
                           wine_data.improved$alcohol

ggplot(data = wine_data.improved, 
       aes(x = prod^(1/3), 
           y = log(density))) +
  geom_point(color = "#ff3f49", 
             alpha = 0.25,
             size = 3) +
  xlab("fixed.acidity * residual.sugar alcohol (" ~g^2/dm^3~ ")") +
  ylab("log(Density)") +
  geom_smooth(method = "lm") +
  ggtitle("Assocation: Density ~ (fixed.acidity * residual.sugar / alcohol)")

```


### Description Two

Here, $R^2$ of above pair is:
```{r, echo=FALSE}
cor(wine_data.improved$prod, wine_data.improved$density)
```

It's pretty high! One of possible explaination can be:
- alcohol has much lower density than water and acid so increase alcohol will decrease density
- fixed acidity and residual sugar will both contribute to increase density because they are heavy than water/alcohol


### Plot Three

The plot that quality is associated with volatile.acidity has data dispersed, it might be hard to figure out pattern, but I can draw some statistics conclusion if based on probability. 

```{r echo=FALSE, Plot_Three}

wine_data.improved$va_levels <- cut(wine_data.improved$volatile.acidity, 
                                    breaks = c(0,0.12,0.39,0.52,0.635,1.01), 
                                    ordered = TRUE)

wine_data.group <- wine_data.improved %>% 
                      group_by(va_levels, quality_level) %>% 
                      summarise(frequency = n()/1547)

ggplot(wine_data.group, 
       aes(x = quality_level, 
           fill = va_levels)) + 
  geom_bar(position = "fill") +
  ylab("Frequency") +
  ggtitle("Frequency distribution of quanlity and volitile.acidity") +
  guides(fill=guide_legend(title="Volitile Acidity level")) +
  xlab(QUALITY_LABEL)

```

### Description Three

From such plot, we can see high quality red wine tends to have more probability of low volitile acidity. So if a red wine has strong acid smell, it possibly a low-quality red wine.

------

# Reflection

In this report, I firstly explore all attributes by their distributions and list the questions I'm interested in. For example, can sweetness, pH and citric acid improve quality?

Then, in bivarate analysis part, I create correlation matrix, select the most 3 significant attributes (alcohol, sulphates, volatile.acidity) associated with quality, and then explore how the data distributes along these 3 attribute dimentions. I find out by histogram and boxplot that data are right skewed distributed along these 3 dimension respectively. To answer my concerns: if sweetness, pH and citric acid can improve quality, I explore the relationship one by one.

Next, I plot how these 3 attributes associate with quality by scatter plot and linear model lines:
- in alcohol vs. quality plot and sulphates vs. quality plot, I find both have positive association
- in volatile.acidity, I find negative association
I use jitter and set transparency to improve the data visualization. Besides, I use linear model to check how these 3 attributes can be fitted by data and then plot 4 scatter plot with different combination to see the strongest associate between attributes. 

Furthermore, I use leveled scatter plot to get idea how quality level distributed in different attributes' dimension scales and reach the assersion: Higher pH, higher sulphates, higher alcohol and higher fixed acidity are more likely to indicate red wines with higher quality.

Finally, I use stacked bar plot to show that higher quality level wine tends to have less probability to involve big volatile acidity value.

# Forecast

This red wine dataset has 12 attributes messed up. Explorasive data analysis does provide with an efficient way to capture idea. But to improve the accuracy of predicting quality of red wine, we can try more improvements, including:
- improve the data, with more data on low / high quality of wines, and more detailed description on how the quality score were given. Better involve more features, like the year of harvest, brew time, location of Vineyard and so on
- use machine learning, like SVM / Decision Tree to mine more details of attributes in more advanced dimensional vision

# Reference

- UC Davis, whats-in-wine: http://waterhouse.ucdavis.edu/whats-in-wine/fixed-acidity
- Predicting Red Wine Quality: Exploratory Data Analysis: https://rpubs.com/jeknov/redwine
- Analysis of Red Wines Dataset by Yohann Lucas: https://rpubs.com/kaltera/UdacityP3
- ggplot docs: http://docs.ggplot2.org/0.9.3.1/position_fill.html
- Sugar in Wine, The Great Misunderstanding: http://winefolly.com/update/sugar-in-wine-misunderstanding/
- The Truth About Sulfites in Wine & the Myths of Red Wine Headaches: http://www.thekitchn.com/the-truth-about-sulfites-in-wine-myths-of-red-wine-headaches-100878
