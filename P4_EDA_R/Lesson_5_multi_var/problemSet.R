# Problem 1
# Create a histogram of diamond prices.
# Facet the histogram by diamond color
# and use cut to color the histogram bars.

# The plot should look something like this.
# http://i.imgur.com/b5xyrOu.jpg

# Note: In the link, a color palette of type
# 'qual' was used to color the histogram using
# scale_fill_brewer(type = 'qual')

library(ggplot2)
data(diamonds)
library(dplyr)
library(gridExtra)
library(tidyr)

str(diamonds)

ggplot(data = diamonds, aes(price)) +
  geom_histogram(aes(fill = cut)) +
  facet_wrap(~color, ncol = 4) +
  scale_fill_brewer(type = 'qual')



# 2.

# Create a scatterplot of diamond price vs.
# table and color the points by the cut of
# the diamond.

# The plot should look something like this.
# http://i.imgur.com/rQF9jQr.jpg


ggplot(diamonds, aes(x = table, y = price)) +
  geom_point(aes(color = cut)) +
  scale_fill_brewer(type = 'qual')

# 3.
# typical table range for majority of diamonds of ideal cut
quantile(subset(diamonds, diamonds$cut == "Ideal")$table, c(0.05,0.95))


quantile(subset(diamonds, diamonds$cut == "Premium")$table, c(0.05,0.95))


# 4.
# Create a scatterplot of diamond price vs.
# volume (x * y * z) and color the points by
# the clarity of diamonds. Use scale on the y-axis
# to take the log10 of price. You should also
# omit the top 1% of diamond volumes from the plot.

# Note: Volume is a very rough approximation of
# a diamond's actual volume.

diamonds <- transform(diamonds, volume = x*y*z)
condition <- quantile(diamonds$volume, 0.99)
ggplot(subset(diamonds, volume <= condition), aes(x = volume, y = log10(price))) +
  geom_point(aes(color = clarity)) +
  scale_color_brewer(type = 'div')



# 5.
# Many interesting variables are derived from two or more others.
# For example, we might wonder how much of a person's network on
# a service like Facebook the user actively initiated. Two users
# with the same degree (or number of friends) might be very
# different if one initiated most of those connections on the
# service, while the other initiated very few. So it could be
# useful to consider this proportion of existing friendships that
# the user initiated. This might be a good predictor of how active
# a user is compared with their peers, or other traits, such as
# personality (i.e., is this person an extrovert?).

# Your task is to create a new variable called 'prop_initiated'
# in the Pseudo-Facebook data set. The variable should contain
# the proportion of friendships that the user initiated.

pf <- read.delim('pseudo_facebook.tsv')

pf$prop_initiated <- ifelse(pf$friend_count == 0, 0, pf$friendships_initiated / pf$friend_count)




# 6.

# Create a line graph of the median proportion of
# friendships initiated ('prop_initiated') vs.
# tenure and color the line segment by
# year_joined.bucket.

# Recall, we created year_joined.bucket in Lesson 5
# by first creating year_joined from the variable tenure.
# Then, we used the cut function on year_joined to create
# four bins or cohorts of users.

# (2004, 2009]
# (2009, 2011]
# (2011, 2012]
# (2012, 2014]


# The plot should look something like this.
# http://i.imgur.com/vNjPtDh.jpg
# OR this
# http://i.imgur.com/IBN1ufQ.jpg

pf$year_joined <- 2014 - ceiling(pf$tenure/365)
pf$year_joined.bucket <- cut(pf$year_joined, breaks = c(2004,2009,2011,2012,2014))

ggplot(data = pf, aes(tenure, prop_initiated)) +
  geom_line(aes(color = year_joined.bucket), stat = "summary", fun.y = median)


# 7.
# Smooth the last plot you created of
# of prop_initiated vs tenure colored by
# year_joined.bucket. You can bin together ranges
# of tenure or add a smoother to the plot.

ggplot(data = subset(pf, tenure > 0), aes(tenure, prop_initiated)) +
  geom_smooth(aes(color = year_joined.bucket))


# 9.

largest_prop_group <- subset(pf, year_joined.bucket == "(2012,2014]")
summary(largest_prop_group$prop_initiated)


# 10.

# Create a scatter plot of the price/carat ratio
# of diamonds. The variable x should be
# assigned to cut. The points should be colored
# by diamond color, and the plot should be
# faceted by clarity.

# The plot should look something like this.
# http://i.imgur.com/YzbWkHT.jpg.

# Note: In the link, a color palette of type
# 'div' was used to color the histogram using
# scale_color_brewer(type = 'div')

ggplot(diamonds, aes(cut,price/carat)) +
  geom_jitter(aes(color = color), position = position_jitter(h = 0))+
  scale_color_brewer(type = 'div') +
  facet_wrap(~clarity, ncol = 3)








