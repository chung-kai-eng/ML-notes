#---------------------------
#install & library packages
#---------------------------

# install.packages("tidyverse")
library(tidyverse)

# install.packages("devtools")
# devtools::install_github("rstudio/EDAWR")
library(EDAWR)
library(tidyr)

#---------------------------
#step 1: tidy data or not?
#---------------------------

View(iris) 
glimpse(iris)
summary(iris)

# ?storms
View(storms)

# ?casesd
View(cases)

# ?pollution
View(pollution)


#------------------------------
#step 2: tidyr: reshaping data
#------------------------------

### gather() columns into rows
cases

gather(
  cases, "year", "n", 2:4)

cases %>% gather("year", "n", 2:4) #good

cases %>% 
        gather("year", "n", 2:4) #good

cases 
%>% gather("year", "n", 2:4) #error

cases %>% 
        gather("year", "n", "2011":"2013")

#exercise
# Q: In "iris" data, 
#    please gather() 1:4 columns into new column, named "part"
#    
View(iris)

iris2 <- iris %>%
        gather(,,)
head(iris2)

#solution:
iris2 <- iris %>%
       gather("part","value",1:4)
head(iris2)




#--------------------------------------------------
### spread() rows into columns
pollution

spread(pollution, size, amount)

pollution %>% 
        spread(size, amount)


pollution2 <- pollution %>% 
        spread(size, amount) %>%
        gather(key="size", value="amount",large:small)

pollution
pollution2
setequal(pollution, pollution2)
?setequal

#exercise
#spread() iris2
iris2

iris2 %>% 
        spread(part,value) #error(只有兩個column)
iris2[300:305,]
#學到的經驗：你必須確保你有observation id，若原本沒有，你就要自己創

#如果你打算spread的data沒有可當id的column，那就不建議無腦做spread
#去觀察toy example

iris.id <- cbind(id=1:nrow(iris),iris)
iris2.id <- iris.id %>%
        gather("part","value",2:5)

iris3.id <-iris2.id %>%
        spread(part,value)

head(iris.id)
head(iris3.id)
setequal(iris.id,iris3.id)


iris.id %>%
        gather("part","value",2:5) %>%
        spread(part,value) %>%
        setequal(iris.id)




#--------------------------------------------------
### unite() several columns into one
storms
storms2 <- storms %>%
        separate(date, c("year", "month", "day"), sep = "-")



#--------------------------------------------------
### separate() one column into several

storms2 %>%
        unite("date", year, month, day, sep="-")



#------------------------------
#step 3: dplyr: transform data
#------------------------------
library(dplyr)
### select()
storms <- read.csv("storms.csv")
storms %>%
        select(storm, pressure)

storms %>%
   select(-storm)

storms %>%
        select(wind:date)
?select

#exercise
#select "Sepal.Length", "Sepal.Width" from iris data

#try
iris.sel <- iris %>% 
        select()

#evaluate the result
head(iris.sel)

#solution:
#iris.sel <- iris %>% 
#        select(Sepal.Length, Sepal.Width)



#--------------------------------------------------
### filter()
storms
storms %>%
        filter(wind >=50,
               storm %in% c("Alberto",	"Alex",	"Allison"))

#exercise
#In iris data, filter() 
#1.Species =  "versicolor" or "virginica"
#2.Sepal.Length > 5

#try
iris.filt <- iris %>%
        filter()

#evaluate the result
iris.filt %>%
        select(Species, Sepal.Length) %>%
        summary()

#solution:
#iris.filt <- iris %>%
#        filter(Species!="setosa", Sepal.Length>5)



#--------------------------------------------------
### arrange()
storms
storms %>% 
        arrange(wind)

storms %>% 
        arrange(desc(wind))

storms %>% 
        arrange(wind,date)



#--------------------------------------------------
### mutate()
storms
storms %>%
        mutate(ratio = pressure / wind)

storms %>%
        mutate(ratio = pressure / wind,
               inverse = ratio^-1)


#exercise
#In iris data, 
#1. create a new column which is named "Sepal.Length.Z"
#2. normalize "Sepal.Length" column and assign to "Sepal.Length.Z"
#hint: use normalize function below
normalized <- function(x) {(x - mean(x))/sd(x)}

#try
iris.mut <-  iris %>%

#evaluate the result
iris.mut %>%
        select(Sepal.Length.Z) %>%
        unlist() %>%
        summary()

#solution
#iris.mut <- iris %>%
#        mutate(Sepal.Length.Z=normalized(Sepal.Length))




#--------------------------------------------------
### summarise()
pollution
pollution %>% 
        summarise(median = median(amount),
                  variance = var(amount))

pollution %>% 
        summarise(mean = mean(amount),
                  sum = sum(amount),
                  n = n())


#exercise
#In iris.mut data, 
#evaluate the mean and variance of "Sepal.Length.Z


#try
iris.mut <- iris %>%
                mutate(Sepal.Length.Z=normalized(Sepal.Length))

iris.mut %>%
        summarise()

#solution
#iris.mut %>%
#        summarise(mean=mean(Sepal.Length.Z),
#                  variance=var(Sepal.Length.Z))




#--------------------------------------------------
### group_by() + summarise

pollution

pollution %>% 
        group_by(city) %>%
        summarise(mean = mean(amount), 
                  sum = sum(amount), 
                  n = n())

head(iris)
iris %>%
        group_by(Species) %>%
        summarise(mean= mean(Sepal.Length),
                  sum= sum(Sepal.Length),
                  n=n())


