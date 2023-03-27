library(caret)
library(doSNOW)#doSNOW will allow us to do trainning in parallel
library(Boruta)
library(mlbench)
library(randomForest)
library(writexl)
library(tidyverse)
library(rpart.plot)


#load data
train<-read.csv("lung_cancer.csv")
View(train)



#setting up factors----
train$Gender<-as.factor(train$Gender)
train$Level<-as.factor(train$Level)
train$Air.Pollution<-as.factor(train$Air.Pollution)
train$Alcohol.use<-as.factor(train$Alcohol.use)
train$Dust.Allergy<-as.factor(train$Dust.Allergy)
train$OccuPational.Hazards<-as.factor(train$OccuPational.Hazards)
train$Genetic.Risk<-as.factor(train$Genetic.Risk)
train$chronic.Lung.Disease<-as.factor(train$chronic.Lung.Disease)
train$Balanced.Diet<-as.factor(train$Balanced.Diet)
train$Obesity<-as.factor(train$Obesity)
train$Smoking<-as.factor(train$Smoking)
train$Passive.Smoker<-as.factor(train$Passive.Smoker)
train$Chest.Pain<-as.factor(train$Chest.Pain)
train$Coughing.of.Blood<-as.factor(train$Coughing.of.Blood)
train$Fatigue<-as.factor(train$Fatigue)
train$Weight.Loss<-as.factor(train$Weight.Loss)
train$Shortness.of.Breath<-as.factor(train$Shortness.of.Breath)
train$Wheezing<-as.factor(train$Wheezing)
train$Swallowing.Difficulty<-as.factor(train$Swallowing.Difficulty)
train$Clubbing.of.Finger.Nails<-as.factor(train$Clubbing.of.Finger.Nails)
train$Frequent.Cold<-as.factor(train$Frequent.Cold)
train$Dry.Cough<-as.factor(train$Dry.Cough)
train$Snoring<-as.factor(train$Snoring)





#Feature selection----
set.seed(111)
boruta<-Boruta(Level~.,data=train,doTrace=2, maxRuns=500)
print(boruta)
plot(boruta, las=2, cex.axis=0.7)





#Subset data to features we wish to keep----
features<-c("Age","Gender","Air.Pollution","Alcohol.use","Dust.Allergy","OccuPational.Hazards","Genetic.Risk","chronic.Lung.Disease","Balanced.Diet","Obesity","Smoking","Passive.Smoker","Chest.Pain","Coughing.of.Blood","Fatigue","Weight.Loss","Shortness.of.Breath","Wheezing","Swallowing.Difficulty","Clubbing.of.Finger.Nails","Frequent.Cold","Dry.Cough","Snoring","Level")
impfeatures<-c("Age","Alcohol.use","Dust.Allergy","Obesity","Passive.Smoker","Coughing.of.Blood","Fatigue","Weight.Loss","Shortness.of.Breath","Wheezing","Swallowing.Difficulty","Clubbing.of.Finger.Nails","Dry.Cough","Snoring","Level")
train<-train[, features]
str(train)
summary(train)
trainimp<-train[,impfeatures]
summary(trainimp)


#Splitting data----
set.seed(54321)
indexes<-createDataPartition(train$Level,
                             times=1,   
                             p=0.7,
                             list=FALSE)
lung_cancer.train<-train[indexes,]
lung_cancer.test<-train[-indexes,]
lung_cancer.trainimp<-trainimp[indexes,]
lung_cancer.testimp<-trainimp[-indexes,]





#We examine the proportions of the level of lung cancer across the datasets----
prop.table(table(train$Level))
prop.table(table(lung_cancer.train$Level))
prop.table(table(lung_cancer.test$Level))



#Here we Train model----
#Here we set up caret to perform 10 fold cross validation repeated 3 times(So we are building 30 models)
#And we use gird search for optimal model hyperparameter values
train.control<-trainControl(method="repeatedcv",
                            number=10,
                            repeats=3
                            )


#Descision Trees with imp features only----
DT1<-train(Level~.,
          data=lung_cancer.trainimp,
          method="rpart",
          trControl=train.control,
          tuneLength=20
          )

#Examining the carets's processing results
print(DT1)


#Make predictions on test set model using DT model
preds1<-predict(DT1,lung_cancer.testimp)
print(preds1)


confusionMatrix(preds1,lung_cancer.testimp$Level)
plot(DT1)
rpart.plot(DT1$finalModel,fallen.leaves = FALSE)



#Descision tree with all features----
DT2<-train(Level~.,
          data=lung_cancer.train,
          method="rpart",
          trControl=train.control,
          tuneLength=20
          )

#Examining the carets's processing results
print(DT2)


#Make predictions on test set model using DT model
preds2<-predict(DT2,lung_cancer.test)
print(preds2)

confusionMatrix(preds2,lung_cancer.test$Level)
plot(DT2)
rpart.plot(DT2$finalModel,fallen.leaves = FALSE)





#Naive Bayes with imp features only----
tune.grid=expand.grid(fL=c(0:4),
                      usekernel=TRUE,
                      adjust=c(1:5))
NaiveBayes1<-train(Level~.,
                data=lung_cancer.trainimp,
                method="nb",
                tuneGrid=tune.grid,
                trControl=train.control
                )

#Examining the carets's processing results
print(NaiveBayes1)


#Make predictions on test set model using svmLinear model
preds3<-predict(NaiveBayes1,lung_cancer.testimp)
print(preds3)

confusionMatrix(preds3,lung_cancer.testimp$Level)
plot(NaiveBayes1)

#Naive Bayes with all features----
tune.grid=expand.grid(fL=c(0:4),
                      usekernel=TRUE,
                      adjust=c(1:5))
NaiveBayes2<-train(Level~.,
                  data=lung_cancer.train,
                  method="nb",
                  tuneGrid=tune.grid,
                  trControl=train.control)

#Examining the carets's processing results
print(NaiveBayes2)


#Make predictions on test set model using svmLinear model
preds4<-predict(NaiveBayes2,lung_cancer.test)
print(preds4)

confusionMatrix(preds4,lung_cancer.test$Level)
plot(NaiveBayes2)




#Random Forest with imp features----
RF1<-train(Level~.,
          data=lung_cancer.trainimp,
          method="rf",
          trControl=train.control,
          tuneLength=5)

#Examining the carets's processing results
print(RF1)


#Make predictions on test set model using Logistic regression model
preds5<-predict(RF1,lung_cancer.testimp)
print(preds5)


confusionMatrix(preds5,lung_cancer.testimp$Level)



#Random Forest with all features----
RF2<-train(Level~.,
           data=lung_cancer.train,
           method="rf",
           trControl=train.control,
           tuneLength=5)

#Examining the carets's processing results
print(RF2)


#Make predictions on test set model using Logistic regression model
preds6<-predict(RF2,lung_cancer.test)
print(preds6)


confusionMatrix(preds6,lung_cancer.test$Level)




#plots----
table1<-data.frame(Algorithm=c("Naive Bayes","Descision Trees","Random Forest"),Accuracy=c(0.8792,0.9866,1.0000),Precison=c(0.90,0.99,1.00),Sensitivity=c(0.88,0.99,1.00))
print(table1)
#write_xlsx(table1,"D://Ajinkya\\allfeatures.xlsx")
table2<-data.frame(Algorithm=c("Naive bayes","Descison Trees","Random Forest"),Accuracy=c(0.9362,0.9866,1.0000),Precison=c(0.94,0.99,1.00),Sensitivity=c(0.93,0.99,1.00))
print(table2)
#write_xlsx(table2,"D://Ajinkya\\Impfeatures.xlsx")
df<-read.csv("allfeatures.csv")
df_f3<-select(df,2:4)
head(df_f3)
df_long<-gather(df,Metric_type,Value,2:4)
head(df_long)
df_sumzd<-group_by(df_long,Algorithm,Metric_type)
head(df_sumzd)
write.csv(df_sumzd,"barplottable.csv")
p<-ggplot(df_sumzd,aes(x=Algorithm,y=Value, fill=Metric_type,label=Value))+geom_bar(stat="identity",position="dodge")+geom_text(position=position_dodge(0.9),vjust=-0.6)+theme_bw()
print(p)

df1<-read.csv("Impfeatures.csv")
df1_f3<-select(df,2:4)
head(df1_f3)
df1_long<-gather(df1,Metric_type,Value,2:4)
head(df1_long)
df1_sumzd<-group_by(df1_long,Algorithm,Metric_type)
head(df1_sumzd)
write.csv(df1_sumzd,"barplottable1.csv")
p1<-ggplot(df1_sumzd,aes(x=Algorithm,y=Value, fill=Metric_type,label=Value))+geom_bar(stat="identity",position="dodge")+geom_text(position=position_dodge(0.9),vjust=-0.6)+theme_bw()
print(p1)













#Comparing----
compare<-data.frame(actual=lung_cancer.test$Level,
                    predicted=preds)
print(compare)
compare1<-data.frame(actual=lung_cancer.testnb$Level,
                     predicted=preds1)   #109
print(compare1)

#Modelinfo----
modelLookup("rpart")
modelLookup("nb")

