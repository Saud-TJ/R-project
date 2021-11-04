# First step: Data Collection

p1<- read.csv("adult.csv",header = TRUE,sep = ",")
View(p1)

################################################################################

# Second step: Data Cleaning

library(dplyr)
p2<- select(p1,age,workclass,education,educational.num,hours.per.week,
            occupation,income)

p2$workclass[p2$workclass=="?"]<-"Unknown"
p2$occupation[p2$occupation=="?"]<-"Unknown"

################################################################################

# Third step Data exploration and analysis

# Visualization of the seleceted variables:

library(ggplot2)

ggplot(p2) + 
  geom_jitter(aes(x=income,y=age))
ggplot(p2) + 
  geom_jitter(aes(x=income,y=workclass))
ggplot(p2) + 
  geom_jitter(aes(x=income,y=education))
ggplot(p2) + 
  geom_jitter(aes(x=income,y=educational.num))
ggplot(p2) + 
  geom_jitter(aes(x=income,y=hours.per.week))
ggplot(p2) + 
  geom_jitter(aes(x=income,y=occupation))

# First tool: Random Forest

library(randomForest)

str(p2)

rf.train.1<- p1[1:48842,c("age","workclass","fnlwgt","education",
                          "educational.num","marital.status","occupation",
                          "relationship","race","gender","capital.gain",
                          "capital.loss","hours.per.week",
                          "native.country")]
rf.label <- as.factor(p1$income)

set.seed(1234)
rf.1<-randomForest(x=rf.train.1 ,y= rf.label , importance = TRUE ,ntree=1000)
rf.1
varImpPlot(rf.1)

# Second tool: Naive Bayes

library(e1071)

traindata <- as.data.frame(p2[1:20000,])
testdata <- as.data.frame(p2[30000,])

traindata
testdata

ageCount <- table(traindata[,c("income","age")])
ageCount

edCount <- table(traindata[,c("income","education")])
edCount

wcCount <- table(traindata[,c("income","workclass")])
wcCount

model <- naiveBayes(income ~.,traindata)
model

results <- predict(model,testdata)
results


# Third Tool: Knn

data.1<- select(p2,age,workclass,education,educational.num,hours.per.week,
                occupation,income)
data.2<- select(p2,age,workclass,education,educational.num,hours.per.week,
                occupation,income)

data.2$workclass[data.2$workclass=="?"]<-"Unknown"
data.2$occupation[data.2$occupation=="?"]<-"Unknown"

summary(data.2)
str(data.2)
data.2$workclass <- as.integer(data.2$workclass,stringsAsFactors = T) 
data.2$education <- as.integer(data.2$education,stringsAsFactors = T)
data.2$occupation <- as.integer(data.2$occupation,stringsAsFactors = T)
data.2$income <-as.factor(data.2$income)
str(data.2)

#Normalize

normalize<-function(nor) {return((nor- min(nor)) /(max(nor) - min(nor))) }

data.2.1<-as.data.frame(lapply(data.2[,1:6],normalize))


set.seed(123)
dat.d <- sample(7:nrow(data.2.1),size = nrow(data.2.1)*0.7,replace = FALSE )
train.dat <- data.1[dat.d,]
test.dat <- data.1[-dat.d,]

train.dat.labels <- data.1[dat.d,7]
test.dat.labels <- data.1[-dat.d,7]
label.dat <- data.1[dat.d,7]
a12 <- data.1[dat.d,7]

library(class)
NROW(train.dat.labels)

#Square root of the numbers of row to get the closer acc  knn number (k)
kn <- knn(train.dat,test.dat,label.dat,cl=a12,k=184)
knn.184 <- knn(train.dat[,7,drop(FALSE)], test.dat[,7,drop(FALSE)], label.dat[,7,drop(FALSE)], cl= train.dat.labels, k=184)
knn.185 <- knn(train.dat, test.dat, cl= train.dat.labels, k=185)

acc184<- 100* sum(test.dat.labels ==knn.184)/NROW(test.dat.labels)
acc185<- 100* sum(test.dat.labels ==knn.185)/NROW(test.dat.labels)

table(knn.184, test.dat.labels)
table(knn.185, test.dat.labels) 


################################################################################

# Fourth step: Data modeling

rf.train.2<- p2[1:48842,c("age","workclass","education","educational.num",
                          "hours.per.week","occupation")]  
rf.label.2 <- as.factor(p2$income)

set.seed(1234)
rf.2<-randomForest(x=rf.train.2 ,y= rf.label.2 , importance = TRUE ,ntree=1000)
rf.2
varImpPlot(rf.2)