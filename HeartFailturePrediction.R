---
  title: "Machine Learning and Survival Analysis for Heart Failture Prediction"
author: "Wanzi Xiao"
date: "01/14/2023"
output:   
  html_document:
  number_sections: true
fig_caption: true
toc: true
fig_width: 7
fig_height: 4.5
theme: cosmo
highlight: tango
code_folding: hide
---
  
  
  ```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
install.packages("survminer")
install.packages("gbm")
library(ggplot2)
install.packages("ggpubr")
install.packages("vctrs")
library(vctrs)
library(ggpubr)
library(survminer)
library(survival)
library(gbm)
install.packages("caret")
install.packages("lattice")
library(lattice)
library(caret)
library(pROC)
install.packages("tree")
library(tree)
install.packages("ISLR")
library(ISLR)
install.packages("vip")
library(vip)
library(e1071)
install.packages("rminer")
library(rminer)
install.packages("tidyverse")
library(tidyverse)
library(reshape2)
install.packages("ggfortify")
library(ggfortify)
library(rpart)
install.packages("skimr")
library(skimr)
library(corrplot)
install.packages("kableExtra")
library(kableExtra)
install.packages("patchwork")
library(patchwork)
install.packages("directlabels")
library(directlabels)
library(randomForest)
install.packages("hrbrthemes")
library(hrbrthemes)
install.packages("viridis")
install.packages("viridisLite")
library(viridisLite)
library(viridis)
library(RColorBrewer)
install.packages("DT")
library(DT)
install.packages("gtsummary")
library(gtsummary)
heart<-read.csv("/Users/wanzi/R/HeartFailturePrediction/data/heart_failure_clinical_records_dataset.csv")
```


# Introduction

#Cardiovascular disease are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide.
#Luckily, most cardiovascular disease can be prevented by addressing behavioral risk factors to population-wide strategies.
#This project is aiming to do an exploratory data analysis,  utilize various machine learning models to detect the most crucial features to predict the heart failure event and apply Cox model, Survival Analysis, and Hazard Ratio to validate the result.

```{r}
# Assign ID
heart$id <- seq.int(nrow(heart))

# Assign Character value to Numeric variables
heart$sexc <-ifelse(heart$sex==1, "Male", "Female")
heart$smoke <-ifelse(heart$smoking==1, "Yes", "No")
heart$hbp <- ifelse(heart$high_blood_pressure==1, "Yes","No")
heart$dia <-ifelse(heart$diabetes==1, "Yes", "No")
heart$anaemiac <- ifelse(heart$anaemia==1 ,"Yes", "No")
# Platelets : Hopkins Medicine
heart$platc <- ifelse(heart$platelets>150000 & heart$platelets <450000, "Platelets Normal", "Platelets Abnormal")
heart$plat <- ifelse(heart$platelets>150000 & heart$platelets <450000, 0,1)

# Serum Sodium: Mayo Clinic
heart$sodiumc <- ifelse(heart$serum_sodium >135 & heart$serum_sodium<145, "Serum Sodium Normal", "Serum Sodium Abnormal")
heart$sodiumn <- ifelse(heart$serum_sodium >135 & heart$serum_sodium<145, 0, 1)

#Creatine Phosphkinase : Mountsinai
heart$cpk <- ifelse(heart$creatinine_phosphokinase >10 & heart$creatinine_phosphokinase<120, "CPK Normal", "CPK Abnormal")
heart$cpkn <- ifelse(heart$creatinine_phosphokinase >10 & heart$creatinine_phosphokinase<120, 0, 1)

#ejection_fraction: Mayo
heart$efraction <-ifelse(heart$ejection_fraction<=75 & heart$ejection_fraction>=41, "Ejection Normal", "Ejection Abnormal")
heart$efractionn <-ifelse(heart$ejection_fraction<=75 & heart$ejection_fraction>=41, 0, 1)

#serum_creatinine :mayo
heart$screat<- ifelse((heart$serum_creatinine<1.35 & heart$serum_creatinine>0.74 & heart$sex==1 ) | (heart$serum_creatinine<1.04 & heart$serum_creatinine>0.59 & heart$sex==0) , "Creatinine Normal", "Creatinine Abnormal"   )
heart$screatn<- ifelse((heart$serum_creatinine<1.35 & heart$serum_creatinine>0.74 & heart$sex==1 ) | (heart$serum_creatinine<1.04 & heart$serum_creatinine>0.59 & heart$sex==0) , 0, 1 )

#age group: Pharma convention  
heart$agegp <- ifelse( heart$age<65, "Age <65", "Age >=65")
heart$agegpn <- ifelse( heart$age<65, 0, 1)

#event vs censor
heart$cnsr <- ifelse(heart$DEATH_EVENT==0, "Censor", "Event")
```

## Original Data table 
```{r}
h1<- subset(heart, select=c(age,anaemia,creatinine_phosphokinase, serum_creatinine,diabetes, ejection_fraction ,high_blood_pressure, platelets , serum_sodium, sex, smoking, DEATH_EVENT))
head(h1, 5)%>% DT::datatable()

h1c<- subset(heart, select=c(agegp,anaemiac,cpk, screat, dia, efraction ,hbp, platc, sodiumc, sexc, smoke, DEATH_EVENT, time))
```


## Modified Categorical Data table
```{r}

head(h1c, 5)%>% DT::datatable()

#Modified Categorical variable selection
m1<- subset(heart, select=c(agegpn,anaemia,cpkn, screatn, diabetes, efractionn ,high_blood_pressure, plat, sodiumn, sex, smoking, DEATH_EVENT))

```


## Training + Testing Data
```{r}
set.seed=8
train.test.split<-sample(2, nrow(h1), replace=TRUE, prob=c(0.8,0.2))
train=h1[train.test.split==1,]
test=h1[train.test.split==2,]

set.seed=18
train.test.split1<-sample(2, nrow(m1), replace=TRUE, prob=c(0.7,0.3))
train1=m1[train.test.split==1,]
test1=m1[train.test.split==2,]

#head(train, 5)%>% DT::datatable()
#head(test, 5)%>% DT::datatable()
```

# Exploratory Data Analysis

## Binary Variable Distribution {.tabset .tabset-fade .tabset-pills}

```{r}
#1. age group
p1<-ggplot(heart, aes(x=agegp))+geom_bar(fill="lightblue")+ labs(x="Age Group")+ theme_minimal(base_size=10)

#2. Sex
p2<-ggplot(heart, aes(x=sexc))+geom_bar(fill="indianred3")+ labs(x="Sex")+ theme_minimal(base_size=10)

#3. Smoking
p3<-ggplot(heart, aes(x=smoke))+geom_bar(fill="seagreen2")+ labs(x="Smoking")+ theme_minimal(base_size=10)

#4. Diabetes
p4<-ggplot(heart, aes(x=dia))+geom_bar(fill="orange2")+
  labs(x="Diabetes Status")+ theme_minimal(base_size=10)

#5. cpk
p5<-ggplot(heart, aes(x=cpk))+geom_bar(fill="lightblue")+
  labs(x="Creatinine Phosphokinase")+ theme_minimal(base_size=8)

#6. Platelets
p6<-ggplot(heart, aes(x=platc))+geom_bar(fill="indianred2")+
  labs(x="Platelets")+ theme_minimal(base_size=8)

#7. serum sodium
p7<-ggplot(heart, aes(x=sodiumc))+geom_bar(fill="seagreen2")+
  labs(x="Serum Sodium") + theme_minimal(base_size=8)

#8. Serum creatinine
p8<-ggplot(heart, aes(x=screat))+geom_bar(fill="orange2")+
  labs(x="Serum Creatinine") + theme_minimal(base_size=8)


#9. anaemia 
p9<-ggplot(heart, aes(x=anaemiac, fill=DEATH_EVENT))+geom_bar(fill="lightblue")+ labs(x="Anaemia")+ theme_minimal(base_size=10)

#10. ejection_fraction
p10<-ggplot(heart, aes(x=efraction))+geom_bar(fill="indianred2")+
  labs(x="Ejection Fraction")+ theme_minimal(base_size=10)

#11. High blood pressure
p11<-ggplot(heart, aes(x=hbp))+geom_bar(fill="seagreen2")+
  labs(x="High Blood Pressure Status")+ theme_minimal(base_size=10)

#12. Event
p12<-ggplot(heart, aes(x=cnsr))+geom_bar(fill="orangered3")+ labs(x="Event Status")+ theme_minimal(base_size=10)

```

### Demographic and Baseline Characters Distribution

```{r}
(p1+p2+p3 +p4)+
  plot_annotation(title="Demographic and Histology Distribution")
```


### Lab Test Result Distribution
```{r}
(p5+p6+p7+p8) + plot_annotation(title="Lab Test Distribution",tag_sep = 5.0)
```


### Disease history Distribution
```{r}
(p9+p10+p11+p12) + plot_annotation(title="Disease History Distribution")
```


## Continuous Variables Disbribution  {.tabset .tabset-fade .tabset-pills}

### Age 
```{r}
#1. Age
c1<- ggplot(heart, aes(x=age))+ geom_histogram(binwidth=5, colour="white", fill="darkseagreen2", alpha=0.8)+
  geom_density(eval(bquote(aes(y=..count..*5))),colour="darkgreen", fill="darkgreen", alpha=0.3)+ scale_x_continuous(breaks=seq(40,100,10))+geom_vline(xintercept = 65, linetype="dashed")+ annotate("text", x=50, y=45, label="Age <65", size=2.5, color="dark green") + annotate("text", x=80, y=45, label="Age >= 65", size=2.5, color="dark red") +labs(title="Age Distribution") + theme_minimal(base_size = 8)
c1

```


### CPK

```{r}

#2. cpk
c2<- ggplot(heart, aes(x=creatinine_phosphokinase))+ geom_histogram(binwidth=100, colour="white", fill="mediumpurple2", alpha=0.8)+
  geom_density(eval(bquote(aes(y=..count..*150))),colour="mediumorchid1", fill="mediumorchid1", alpha=0.3)+ scale_x_continuous(breaks=seq(0,10000,1000))+geom_vline(xintercept = 120, linetype="dashed")+ annotate("text", x=0, y=100, label="CPK Normal", size=2.5, color="dark green") + annotate("text", x=1000, y=80, label="CPK Abnormal", size=2.5, color="dark red")+labs(title="Creatinine Phosphokinase Distribution") + theme_minimal(base_size = 8)
c2
```


### Ejection Fraction
```{r}
c3<- ggplot(heart, aes(x=ejection_fraction))+ geom_histogram(binwidth=5, colour="white", fill="lightpink1", alpha=0.8)+
  geom_density(eval(bquote(aes(y=..count..*5))),colour="mistyrose2", fill="mistyrose2", alpha=0.3)+ scale_x_continuous(breaks=seq(0,80,10))+geom_vline(xintercept = 40, linetype="dashed")+geom_vline(xintercept = 75, linetype="dashed")+ annotate("text", x=20, y=30, label="Abnormal", size=2.5, color="dark red") + annotate("text", x=50, y=30, label="Normal", color="dark green")+  annotate("text", x=80, y=30, label="Abnormal", size=2.5, color="dark red")+labs(title="Ejection Fraction Distribution") + theme_minimal(base_size = 8)
c3
```

### Platelets Count
```{r}
c4<- ggplot(heart, aes(x=platelets))+ geom_histogram(binwidth=20000, colour="white", fill="lightskyblue2", alpha=0.8)+
  geom_density(eval(bquote(aes(y=..count..*25000))),colour="lightsteelblue", fill="lightsteelblue", alpha=0.3)+
  geom_vline(xintercept = 150000, linetype="dashed")+geom_vline(xintercept = 450000, linetype="dashed")+ annotate("text", x=100000, y=30, label="Abnormal", size=2.5, color="dark red") + annotate("text", x=300000, y=30, label="Normal", color="dark green")+  annotate("text", x=500000, y=30, label="Abnormal", size=2.5, color="dark red")+labs(title="Platelets Count") + theme_minimal(base_size = 8)
c4
```



### Serum Sodium
```{r}
c5<- ggplot(heart, aes(x=serum_sodium))+ geom_histogram(binwidth=1, colour="white", fill="lightsalmon", alpha=0.8)+
  geom_density(eval(bquote(aes(y=..count..))),colour="lightcoral", fill="lightcoral", alpha=0.3)+
  geom_vline(xintercept = 135, linetype="dashed")+geom_vline(xintercept = 145, linetype="dashed")+ annotate("text", x=130, y=20, label="Abnormal", size=2.5, color="dark red") + annotate("text", x=142, y=20, label="Normal", color="dark green")+  annotate("text", x=148, y=20, label="Abnormal", size=2.5, color="dark red")+labs(title="Serum Sodium") + theme_minimal(base_size = 8)
c5
```


### Serum Creatinine
```{r}
c6<- ggplot(heart, aes(x=serum_creatinine))+ geom_histogram(binwidth=0.2, colour="white", fill="lightgoldenrod", alpha=0.8)+
  geom_density(eval(bquote(aes(y=..count..*0.2))),colour="moccasin", fill="moccasin", alpha=0.3)+
  geom_vline(xintercept = 0.74, linetype="dashed")+geom_vline(xintercept = 1.35, linetype="dashed")+ annotate("text", x=0.05, y=20, label="Abnormal", size=2.5, color="dark red") + annotate("text", x=1, y=20, label="Normal", color="dark green")+  annotate("text", x=2.5, y=20, label="Abnormal", size=2.5, color="dark red")+labs(title="Serum Creatinine") + theme_minimal(base_size = 8)
c6
```

## Death Event Count with Survival time   {.tabset .tabset-fade .tabset-pills}

### Bubble Chart
As we can see, the triagle shape is Death, circle represents Censored subjects, the size of the shape represents the count of patients live/dead on the same day.
It is clear Censored (circle) patients live longer.

```{r, warning=FALSE}
d1 <- group_by(heart,time,DEATH_EVENT)
d2<- summarise(d1,count=n())
d22 <- arrange(d2, desc(time))


ggplot(d22, aes(x=reorder( time, count), y=time))+ geom_point(aes(size=count, colour=factor(count), shape=factor(DEATH_EVENT)), alpha=1/2)+
  theme_ipsum() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1) ,  legend.position="none") +coord_flip() +ylab("Survival days") +xlab("Survival counts")+ ggtitle("Patient survival time with counts")
```


### Lollipop Chart for Survival Status with Censor

We use the blue line to present censored(alive) patients, and orange line to represent the patients with event(dead). It is very clear the censored patients lived longer in general.

```{r}
heart$idc <- paste("id",as.factor(heart$id))

lol1_100<-ggplot(heart[0:100,], aes(x=idc, y=time)) +
  geom_segment( aes(x=idc, xend=idc, y=0, yend=time),        color=ifelse(heart[0:100,]$DEATH_EVENT==1, "orange", "skyblue"))+
  geom_point( color=ifelse(heart[0:100,]$DEATH_EVENT==1, "red", "darkgreen"), size=0.1, alpha=0.6) +
  theme_light() +
  coord_flip() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.border = element_blank(),
    axis.ticks.y = element_blank()
  )+ ylab("Survival days") +xlab("Patient ID 1-100") + ggtitle("Patient 1-100 Survival Status with Censor")

lol101<-ggplot(heart[101:200,], aes(x=idc, y=time)) +
  geom_segment( aes(x=idc, xend=idc, y=0, yend=time),        color=ifelse(heart[0:100,]$DEATH_EVENT==1, "orange", "skyblue"))+
  geom_point( color=ifelse(heart[0:100,]$DEATH_EVENT==1, "red", "darkgreen"), size=0.1, alpha=0.6) +
  theme_light() +
  coord_flip() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.border = element_blank(),
    axis.ticks.y = element_blank()
  )+ ylab("Survival days") +xlab("Patient ID 101-200") + ggtitle("Patient 101-200 Survival Status with Censor")

lol201<-ggplot(heart[201:299,], aes(x=idc, y=time)) +
  geom_segment( aes(x=idc, xend=idc, y=0, yend=time),      color=ifelse(heart[201:299,]$DEATH_EVENT==1, "orange", "skyblue"))+
  geom_point( color=ifelse(heart[201:299,]$DEATH_EVENT==1, "red", "darkgreen"), size=0.1, alpha=0.6) +
  theme_light() +
  coord_flip() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.border = element_blank(),
    axis.ticks.y = element_blank()
  )+ ylab("Survival days") +xlab("Patient ID 201-299") + ggtitle("Patient 201-299 Survival Status with Censor")

lol1_100
lol101
lol201
```

## Correlations  {.tabset .tabset-fade .tabset-pills}

### Correlation Matrix

From the correlation matrix, we can see Death Event is highly correlated with serum creatinine, age, serum sodium, ejection fraction.

```{r}

r=cor(h1)
corrplot(r, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 90)
```

### Heatmap

We will also show the heatmap as a side evidence.
```{r}
coul <- colorRampPalette(brewer.pal(8, "PiYG"))(25)
heatmap(r, scale="column", col = coul)
```

```{r}
draw_confusion_matrix <- function(cm) {
  
  total <- sum(cm$table)
  res <- as.numeric(cm$table)
  
  # Generate color gradients. Palettes come from RColorBrewer.
  greenPalette <- c("#F7FCF5","#E5F5E0","#C7E9C0","#A1D99B","#74C476","#41AB5D","#238B45","#006D2C","#00441B")
  redPalette <- c("#FFF5F0","#FEE0D2","#FCBBA1","#FC9272","#FB6A4A","#EF3B2C","#CB181D","#A50F15","#67000D")
  getColor <- function (greenOrRed = "green", amount = 0) {
    if (amount == 0)
      return("#FFFFFF")
    palette <- greenPalette
    if (greenOrRed == "red")
      palette <- redPalette
    colorRampPalette(palette)(100)[10 + ceiling(90 * amount / total)]
  }
  
  # set the basic layout
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  classes = colnames(cm$table)
  rect(150, 430, 240, 370, col=getColor("green", res[1]))
  text(195, 435, classes[1], cex=1.2)
  rect(250, 430, 340, 370, col=getColor("red", res[3]))
  text(295, 435, classes[2], cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col=getColor("red", res[2]))
  rect(250, 305, 340, 365, col=getColor("green", res[4]))
  text(140, 400, classes[1], cex=1.2, srt=90)
  text(140, 335, classes[2], cex=1.2, srt=90)
  
  # add in the cm results
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}

```

# Machine Learning Model for Variable Selections

We applied 3 machine learning methods (GLM, GBM, Random Forest) to predict the most crucial factors for Heart Failure Death_Event as dependent variable in the original data structure as well as the categorized data frame. Overall, we found serum_creatinine, age, ejection fraction, and serum sodium are four of the most substantial features, which is consistent with the correlation matrix.

note: Exclude the "time" variable (paradox) 

## GBM {.tabset .tabset-fade .tabset-pills}

**Gradient Boosting ** is machine learning technique used in regression and classification task. It is famous for converting weak learners into strong learners. It starts by assign an equal weight to every tree. then improve upon the prediction of the first tree. We iterate this Tree1 +Tree2 process to a specified number. 

### GBM with Original Data

```{r}
gbm.m<- gbm(train$DEATH_EVENT ~. , data=train, distribution = "bernoulli",
            cv.folds=10, shrinkage=0.01, n.minobsinnode = 10, n.trees=1000)
#gbm.m
gbm.imp=summary(gbm.m)
gbm.imp

gmb.t =predict(object=gbm.m, newdata=test, n.trees=1000, type="response")
presult<- as.factor(ifelse(gmb.t>0.5,1,0))
test$DEATH_EVENT1<-as.factor(test$DEATH_EVENT)
g<-confusionMatrix(presult,test$DEATH_EVENT1)
draw_confusion_matrix(g)

```

### GBM with Categorized Data
```{r}
gbm.m1<- gbm(train1$DEATH_EVENT ~. , data=train1, distribution = "bernoulli",
             cv.folds=10, shrinkage=0.01, n.minobsinnode = 10, n.trees=1000)
#gbm.m1
gbm.imp1=summary(gbm.m1)
gbm.imp1

gmb.t1 =predict(object=gbm.m1, newdata=test1, n.trees=1000, type="response")
presult<- as.factor(ifelse(gmb.t1>0.5,1,0))
test1$DEATH_EVENT1<-as.factor(test1$DEATH_EVENT)
g1<-confusionMatrix(presult,test1$DEATH_EVENT1)
draw_confusion_matrix(g1)

```

## Random Forest {.tabset .tabset-fade .tabset-pills}

### Random Forest with Original Data
```{r}
rforest<- randomForest(factor(DEATH_EVENT) ~. , data=train, ntree=500, importance=TRUE)
#summary(rforest)
imp<-varImp(rforest)
varImpPlot(rforest)

rpredict<- predict(rforest, test, type="class")
cm2<-confusionMatrix(rpredict, test$DEATH_EVENT1)
draw_confusion_matrix(cm2)
```

### Random Forest with Categorized Data

```{r}
rforest1<- randomForest(factor(DEATH_EVENT) ~. , data=train1, ntree=500, importance=TRUE)
#summary(rforest1)
imp1<-varImp(rforest1)
varImpPlot(rforest1)

rpredict1<- predict(rforest1, test1, type="class")
cm21<-confusionMatrix(rpredict1, test1$DEATH_EVENT1)
draw_confusion_matrix(cm21)
```

## General Linear Model{.tabset .tabset-fade .tabset-pills}

### General Linear Model with Original Data

```{r}
lm1 <- glm(DEATH_EVENT ~., data=train, family=binomial(link="logit"))

#summary(lm1)
limp<-varImp(lm1)
backward<-step(lm1,direction="backward", trace=0)
#vi(backward)
p2<- vip(backward,num_features = length(coef(backward)),
         geom="point", horizontal = TRUE, mapping = aes_string(color="Sign"))
p2

glm.t =predict(object=lm1, newdata=test, type="response")
presult<- as.factor(ifelse(glm.t>0.5,1,0))
test$DEATH_EVENT1<-as.factor(test$DEATH_EVENT)
cm3<- confusionMatrix(presult,test$DEATH_EVENT1)
draw_confusion_matrix(cm3)
```

### General Linear Model with Classication

```{r}
lm11 <- glm(DEATH_EVENT ~., data=train1, family=binomial(link="logit"))

#summary(lm1)
limp1<-varImp(lm11)
backward<-step(lm11,direction="backward", trace=0)
#vi(backward)
p21<- vip(backward,num_features = length(coef(backward)),
          geom="point", horizontal = TRUE, mapping = aes_string(color="Sign"))
p21

glm.t1 =predict(object=lm11, newdata=test1, type="response")
presult1<- as.factor(ifelse(glm.t1>0.5,1,0))
test1$DEATH_EVENT1<-as.factor(test1$DEATH_EVENT)
cm31<- confusionMatrix(presult1,test1$DEATH_EVENT1)
draw_confusion_matrix(cm31)

```


# Survival Analysis

## Survival Analysis Basics {.tabset .tabset-fade .tabset-pills}
First, we will test the proportional hazard assumption, and assess the features individually see how it fits with survival curves and the respected Hazard Ratio. Then we will combine three most important features together to create the Risk Low and Risk High Group.


### Definiton and Concept

**Survival Analysis** investigates the time to event outcome  involve censoring is the most common statistical approaches in the medical literature. It is important to know a time of event for each patient. In our case, the time of heart failure occurs.


- Survival Function:  The probability of survival longer than a specific time point

$$ S(t) = P(T > t) $$
  
  - Hazard function: The risk of having the event in the next interval conditional on surviving to the beginning of the interval

$$ h(t)=\lim_{\Delta t \to 0 } \frac{1}{\Delta t} P(T <= t+ \Delta t|T>t)$$
  
  $$ H(t) = -log(S(t)) $$
  **Censor** We use right censoring rule, the event occurred after a specific date



**Cox Model** a well-known model to exploring the relationship between the survival of a patient and several explanatory variables. It estimates the hazard(risk) of event of interest for individuals, given their prognostic variables. 

$$h_i(t)=h_0(t) e^{\beta_1*x1+ \beta_2*x2} $$
  
  
  
  ### Survival function and its Relations
  
  If we know one of the function S(t), f(t), $\lambda (t)$ , $\Lambda(t)$, we can compute the rest three by FTC.

- Survivor function S(t)
$$S(t)=Pr(T \geq t)= \int_t^\infty f(u)du$$
  
  - The density function f(t)
$$ f(t)=\lim_{\Delta t \to 0 } \frac{1}{\Delta t} Pr(t \leq T \leq t+ \Delta t)$$
  By FTC, fundamental calculus part2, we have $F(t)=Pr(T<t)$, which leads to $S(t)=1-F(t)$
  
  - The cumulative hazard function $\Lambda(t)$ or H(t) , with the help of FTC, we have $S(t)=e^{-\Lambda(t)}=e^{-H(t)}$
  $$\Lambda(t) = H(t)= \int_0^t \lambda(u)du = h(u)du$$
  - The hazard function $\lambda(t)$ or h(t)
$$\lambda(t) = h(t) =\frac{f(t)}{S(t)}= -\frac{d}{dt}[logS(t)]$$
  
  S(t) survivor function decrease overtime, while H(t) increase overtime. (survive probability is smaller along the time, while the risk of failure increases over the time)



## Assessing Proportional Hazard

We will use the Cox.zph function to test the Proportional Hazard assumptions, if the HR assumption is violated, we will use Restricted mean survival analysis for further investigation. From the pictures, we can see our data satisfy the Proportional conditions.


1. A significant p-value indicates the proportional hazards assumptions is violated


2. Plot Schoefeld residuals (zero-slop line)

```{r}
mv_fit <- coxph(Surv(time,DEATH_EVENT) ~ efraction+ agegp + screat +sodiumc, data=heart)
ccox<- cox.zph(mv_fit)
print(ccox)

options(repr.plot.width=10, repr.plot.height=40)
ggcoxzph(ccox)
```

## Hazard Ratio {.tabset .tabset-fade .tabset-pills}

The Hazard Ratio presented a strong survival difference between groups.

**Hazard Ratio** : A comparison between the probability of events in a treatment group, compared to the probability of events in a control group. It's used to see if patients receiving a treatment progress faster (slower) than those not receiving treatment.

$$ \lambda  (t_iX)=  \lambda_0(t)exp(\beta X)  $$ 

$$ log\ of \ Hazard = log(a) +b1x1...bkxk$$

HR=1: No difference between treatment or control group


HR<1: Probability of Event happen in treatment group is smaller than control group

HR>1: Probability of Event happen in treatment group is greater than control group


### Hazard Ratio for Important Variable

We can see Ejection fraction, serum creatinine, sodium creatinine normal group has lower probability of event happening, whereas older age group experience high probability of heart failure event.
```{r, warning=FALSE}
ggforest(mv_fit)
```

### Hazard Ratio for other variables
The graph presented High blood pressure, anaemia disease, platelets count and CPK have somewhat significant level in hazard ratio.

Whereas to my surprise, Smoking didn't make a difference.


```{r, warning=FALSE}
mo <- coxph(Surv(time,DEATH_EVENT) ~sexc +dia +hbp +smoke +anaemiac+ platc+cpk, data=heart)
ggforest(mo)
```

## KM Curve

The Kaplan Meier Curve is an estimator used to estimate non-parametric (Log rank test) survival function. The Kaplan Meier Curve is the visual representation of this function that shows the probability of an event at a respective time interval. It requires no assumptions regarding the underlying distribution of the data.

### Ejection Fraction, Serum Creatinine , Age Group , Serum Sodium

The KM plot demonstrated the survival curves separated for the compared groups with respected to the features: Ejection Fraction, Serum Creatinine, Age Group, and Serum Sodium 

```{r}
fit_ef<-survfit(Surv(time,DEATH_EVENT)~heart$efraction, data=heart)
fit_sc<-survfit(Surv(time,DEATH_EVENT)~screat, data=heart)
fit_age<-survfit(Surv(time,DEATH_EVENT)~agegp, data=heart)
fit_sd<-survfit(Surv(time,DEATH_EVENT)~sodiumc, data=heart)

splots<- list()
splots[[1]]<-ggsurvplot(fit_ef,data=heart ,xlab="Days", ggtheme=theme_minimal())
splots[[2]]<-ggsurvplot(fit_sc,data=heart, xlab="Days", ggtheme=theme_minimal())
splots[[3]]<-ggsurvplot(fit_age,data=heart,xlab="Days", ggtheme=theme_minimal())
splots[[4]]<-ggsurvplot(fit_sd,data=heart,xlab="Days", ggtheme=theme_minimal())

arrange_ggsurvplots(splots, print=TRUE, ncol=2, nrow=2)
```

### Final Survival Analysis Model 

Define Risk Group:
  
  **Risk Low**: serum creatinine normal+ age<65 + ejection fraction normal

**Risk High**: rest combination

```{r}
heart$riskgp <- ifelse(heart$agegp=="Age <65" & heart$efraction=="Ejection Normal" & heart$sodiumc=="Serum Sodium Normal", "Risk Low", "Risk High")
fit<-survfit(Surv(time,DEATH_EVENT)~riskgp, data=heart)
km<-ggsurvplot(fit,data=heart, risk.table=TRUE, legend="none", break.time.by=30, size=0.2,tables.height=0.3, xlab="Days")
km
```

```{r}
#log rank p value for groups
survdiff(Surv(time, DEATH_EVENT) ~ riskgp, data=heart)
```

### Cox Proportion Model and Hazard Ratio

Finally, we can see the *p value* from the Cox model is consistent with our result. 

```{r, warning=FALSE}
coxph(Surv(time,DEATH_EVENT) ~ riskgp, data=heart) %>%
  gtsummary::tbl_regression(exp=TRUE)
```

# Summary

From applying various Machine Learning models, we reached a similar result, that Serum Sodium, Serum Creatinine, Ejection Fraction, and Age Group are the most important features in Heart Disease. The survival analysis and Kaplan Meier curves validated our findings. Therefore, we could let the public know if the patient has a chronical heart problem, they can periodically monitor their Serum Creatinine, Serum Sodium, and Ejection Fraction once above 65.

**Prevention is better than Treatment!**
  
  

  