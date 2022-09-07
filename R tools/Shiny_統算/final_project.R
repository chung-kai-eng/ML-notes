library(dplyr)
library(GGally)
library(ggplot2)
library(corrplot)
library(reshape2)
library(gridExtra)
library(MASS)
library(mclust)
library(clustMixType)

################## data cleaning and output processed dataset ###############
# dataset <- read.csv("D:\\大四辛酸史\\統計計算\\Assignment\\Final_project\\cardio_train_revised.csv", header = T)
# summary(dataset)
# cat("Original dataset dimension is: ", dim(dataset), "\n")
# dataset <- filter(dataset, dataset$height >= 120 & dataset$height <= 190)
# dataset <- filter(dataset, dataset$weight >= 40 & dataset$weight <= 150)
# dataset <- filter(dataset, dataset$ap_hi >= 90 & dataset$ap_hi <= 180)
# dataset <- filter(dataset, dataset$ap_lo >= 60 & dataset$ap_lo <= 100)
# cat("After removing weird data, dataset dimension is: ", dim(dataset), "\n")
# summary(dataset)
# write.csv(dataset, file = "D:\\大四辛酸史\\統計計算\\Assignment\\Final_project\\cardio_train_revised.csv")
####################################

dataset <- read.csv("D:\\大四辛酸史\\統計計算\\Assignment\\Final_project\\cardio_train_revised.csv", header = T)
dim(dataset)
# pairplot of the dataframe
# dataset is too large which will cause Rstudio crash down
# pairs(dataset)

# change the type of the features (quantitative & qualitative)
dataset <- mutate(dataset, gender = factor(gender), cholesterol = factor(cholesterol),
                  gluc = factor(gluc), smoke = factor(smoke), alco = factor(alco),
                  active = factor(active), cardio = factor(cardio)) %>%
           select(id:age, height:ap_lo, gender, cholesterol:cardio)
summary(dataset)
# add a term bmi into the dataset
dataset$bmi = dataset$weight / (dataset$height / 100) ^ 2

# quantitative data
# correlation matrix
names(dataset)[names(dataset) == "cholesterol"] = "chole"
cor_data.cor = cor(dataset[,2:6], method = c("pearson")) # default: pearson
corrplot.mixed(cor_data.cor, upper="square", lower = "number")

# boxplot of quantitative data
par(mfrow = c(2, 3), cex = 0.6)
col_name <- colnames(dataset)
for (i in 2:6){
  print(i)
  boxplot(dataset[,i], xlab = col_name[i], col = "#FF8888", border = "#FF3333", las = 2)
}
par(mfrow = c(1, 1))


# qualitative data
qualitative_data <- summary(dataset[,7:13])
# barplot of qualitative data
g_list <- list()
for (i in colnames(dataset[7:13])){
  g_list[[i]] <- ggplot(data = dataset, aes_string(i)) + 
                   geom_bar(stat = "count", fill = "#66009D", alpha = 0.4) 
}
do.call(grid.arrange, g_list)
# test whether the amount of sugar will affect blood pressure
g1 <- ggplot(dataset, aes(x = gluc, y = ap_hi)) + 
        geom_boxplot(col ="#FFAA33", fill = "#FFAA33", alpha = 0.4) +
        ggtitle("ap_hi vs. glucose") + 
        ylab("ap_hi (mmHg)") +
        xlab("gluc (type)")

g2 <- ggplot(dataset, aes(x = gluc, y = ap_lo)) + 
        geom_boxplot(col ="#FFAA33", fill = "#FFAA33", alpha = 0.4) +
        ggtitle("ap_lo vs. glucose") + 
        ylab("ap_lo (mmHg)") +
        xlab("gluc (type)")

grid.arrange(g1, g2, ncol = 2)

aov_1 = aov(dataset$ap_hi ~ dataset$gluc)
summary(aov_1)
aov_2 = aov(dataset$ap_lo ~ dataset$gluc)
summary(aov_2)

# test whether the amount of cholesterol will affect blood pressure
g3 <- ggplot(dataset, aes(x = chole, y = ap_hi)) + 
  geom_boxplot(col ="#00AA00", fill = "#00AA00", alpha = 0.4) +
  ggtitle("ap_hi vs. cholesterol") + 
  ylab("ap_hi (mmHg)") +
  xlab("cholesterol (type)")

g4 <- ggplot(dataset, aes(x = chole, y = ap_lo)) + 
  geom_boxplot(col ="#00AA00", fill = "#00AA00", alpha = 0.4) +
  ggtitle("ap_lo vs. cholesterol") + 
  ylab("ap_lo (mmHg)") +
  xlab("cholesterol (type)")

grid.arrange(g3, g4, ncol = 2)

aov_3 = aov(dataset$ap_hi ~ dataset$chole)
summary(aov_3)
aov_4 = aov(dataset$ap_lo ~ dataset$chole)
summary(aov_4)

# test whether smoke will affect blood pressure
g5 <- ggplot(dataset, aes(x = smoke, y = ap_hi)) + 
  geom_boxplot(col ="#5599FF", fill = "#5599FF", alpha = 0.4) +
  ggtitle("ap_hi vs. smoke") + 
  ylab("ap_hi (mmHg)") +
  xlab("smoke (type)")

g6 <- ggplot(dataset, aes(x = smoke, y = ap_lo)) + 
  geom_boxplot(col ="#5599FF", fill = "#5599FF", alpha = 0.4) +
  ggtitle("ap_hi vs. smoke") + 
  ylab("ap_hi (mmHg)") +
  xlab("smoke (type)")

grid.arrange(g5, g6, ncol = 2)
# test one way anova
aov_5 = aov(dataset$ap_hi ~ dataset$smoke)
summary(aov_5)
aov_6 = aov(dataset$ap_lo ~ dataset$smoke)
summary(aov_6)


# patients with cardio has higher ap_hi and ap_lo than those without cardio
g7 <- ggplot(dataset, aes(x = cardio, y = ap_hi)) + 
        geom_boxplot(col ="#FF8888", fill = "#FF8888", alpha = 0.4) +
        ggtitle("ap_hi vs. cardio") + 
        ylab("ap_hi (mmHg)") +
        xlab("cardio")

g8 <- ggplot(dataset, aes(x = cardio, y = ap_lo)) + 
        geom_boxplot(col ="#FF8888", fill = "#FF8888", alpha = 0.4) +
        ggtitle("ap_hi vs. cardio") + 
        ylab("ap_lo (mmHg)") +
        xlab("cardio")
grid.arrange(g7, g8, ncol = 2)

# we can find that patients with cardio has higher bmi than those without cardio
ggplot(dataset, aes(x = cardio, y = bmi)) + 
  geom_boxplot(col ="#FFAA33", fill = "#FFAA33", alpha = 0.4) +
  ggtitle("bmi vs. cardio") + 
  ylab("bmi (kg/m^2)") +
  xlab("cardio")


# histogram of quantitative variable
data_frame(val = X) %>%
  ggplot(., aes(val)) +
  geom_histogram(col = "#9999FF", fill = "#9999FF", alpha = 0.6) +
  ggtitle("historgram by Acceptance_Rejection Region") + 
  ylab("count") +
  xlab("X")


ggplot(dataset, aes(x = height)) + 
  geom_histogram(col = "#9999FF", fill = "#9999FF", alpha = 0.6, bins = 10)

# cardio vs. other qualtitative variable
x = table(dataset$cardio, dataset$gluc)
prop_tab = data.frame(prop.table(x, margin = 1))
print(prop_tab)
names(prop_tab) <- c("cardio", "gluc", "prop")
ggplot(prop_tab, aes_string(x = "cardio", y = "prop", fill = "gluc")) +
  geom_bar(stat = "identity", alpha = 0.8)





# the average of the patients' bmi
mean(dataset[dataset$cardio == 1,]$bmi) # 28.39499
mean(dataset[dataset$cardio == 0,]$bmi) # 26.45576


# 沒有顯著引響 for ^2 & ^3
summary(lm(ap_hi ~ bmi + I(bmi^2) + I(bmi^3), data = dataset))

#############################################
# test linear model and logistic regression
#############################################

model <- lm(ap_hi ~ age + height + weight + ap_lo + gender + chole + active + cardio, data = dataset)
summary(model)

par(mfrow = c(1, 1), cex = 1.5)
plot(model)

# model_2 <- lm(bmi ~ age + ap_hi + ap_lo + gender + chole + active + cardio, data = dataset)
# summary(model_2)
# active is insignificant. However, I think there's something wrong
# because of the variable not included in the model
anova(model)
par(mfrow = c(2, 2), cex = 1.5)
plot(dataset$gender, dataset$ap_hi)

############################################
# Classification
###########################################





###########################################
# Clustering
###########################################
standardized <- function(data, from, to){
  for(i in from:to){
    data[ ,i] <- scale(data[ ,i])
  }
  #print(head(data))
  return(data)
}

#std_data <- standardized(dataset, 2, 6)
clus_data <- dataset %>% select(age, bmi, ap_hi, chole, gluc)
clus_data <- standardized(clus_data, 1, 4)
# lambda: parameter > 0 to trade off between Euclidean distance of numeric variables
# and simple matching coefficient between categorical variables.
sug_lambda <- lambdaest(clus_data, num.method = 2, fac.method = 2, outtype = "numeric")
# determine which factors are important
clus_1 <- kproto(clus_data, 2, lambda = sug_lambda, iter.max = 100)
summary(clus_1)
# visualization of a k-prototypes clustering result for cluster interpretation
# for numerical variables boxplot and for categorical variables barplots
par(mfrow = c(1, 1))
#result <- clprofiles(clus_1, dataset)
#result
clus_1$cluster - 1
# accuarcy table
table <- table(dataset$cardio, (clus_1$cluster - 1))
sum(diag(table)) / sum(table)



# in real world, clusters are often not as clear cut
# by varaition of lambda the emphasize is shifted towards factor/ numeric varaibles


plot(dataset[, 2:12], col = clus_1$cluster + 1)
#, main = "K mean clustering result with k = 2",
#     xlab = "", ylab = "", pch = 20, cex = 2)



# Euclidean distance (it's too large to construct the hierarchical tree)
dist <- dist(dataset[1:2000 , c(2:3)] , diag=TRUE) 
# Hierarchical Clustering with hclust
hc_complete <- hclust(dist, method = "complete")
# Plot the result
plot(hc_complete, main = "Cluster Dendrogram with complete method")

hc_average <- hclust(dist, method = "average")
plot(hc_average, main = "Cluster Dendrogram with average method")

hc_centroid <- hclust(dist, method = "centroid")
plot(hc_centroid, main = "Cluster Dendrogram with centroid method")




############# k-prototype ##################

standardized <- function(data, from, to){
  for(i in from:to){
    data[ ,i] <- scale(data[ ,i])
  }
  #print(head(data))
  return(data)
}

# change the type of the features (quantitative & qualitative)
dataset <- mutate(dataset, gender = factor(gender), cholesterol = factor(cholesterol),
                  gluc = factor(gluc), smoke = factor(smoke), alco = factor(alco),
                  active = factor(active), cardio = factor(cardio)) %>%
  select(id:age, height:ap_lo, gender, cholesterol:cardio)
# add a term bmi into the dataset
dataset$bmi = dataset$weight / (dataset$height / 100) ^ 2
names(dataset)[names(dataset) == "cholesterol"] = "chole"


clus_data <- dataset %>% select(age, bmi, ap_hi, chole, gluc)
clus_data <- standardized(clus_data, 1, 3)
sug_lambda <- lambdaest(clus_data, num.method = 2, fac.method = 2, outtype = "numeric")
clus_1 <- kproto(clus_data, 2, lambda = sug_lambda, iter.max = 200)
table <- table(dataset$cardio, (clus_1$cluster - 1))
sum(diag(table)) / sum(table)

################ simulation data ###################
load(file = "D:\\大四辛酸史\\統計計算\\Assignment\\Final_project\\data.sim.rda")

