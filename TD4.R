library(data.table)
library(xgboost)
library(randomForest)
library(caret)


digit <- fread("C:/Users/xuan-quang.do/Documents/ESILV/Machine Learning/github_share/digit_recognizer.csv")

names(digit)
sqrt(784) #28*28

set.seed(1)
digit = copy(digit[sample(.N, 0.1 * nrow(digit))])
digit[, .N / nrow(digit) * 100, label][order(label)]

hist(digit$label, breaks = seq(-0.5, 9.5, by = 1))


#Visualization 
plotdigit<- function(datarow){
  # function will produce an image of the data given
  # the function takes away the first value because it is the target
  
  rotate <- function(x) t(apply(x, 2, rev))
  
  title<- datarow[, 1] # get actual value from training data
  datarow <- datarow[, -1] # remove the target column
  datarow<- as.numeric(datarow) # convert to numeric
  
  x<- rep(0:27)/27 
  y<- rep(0:27)/27
  z<- matrix(unlist(datarow), ncol=28, byrow=T)
  z<- rotate(z)
  
  
  main_ti = paste0("Actual Value:", title)
  image(x,y,z, main=main_ti, col=gray.colors(255, start=1, end=0), asp=1,
        xlim=c(0,1), ylim=c(-0.1,1.1), useRaster = T, axes=F, xlab='', ylab='')
}


par(mfrow=c(3, 4))
set.seed(1)
rows = sample(1:nrow(digit), size = 12)
for (i in rows){
  plotdigit(digit[i, ])
}



set.seed(1)
datum = digit[label == 1]
rows = sample(1:nrow(datum), size = 12)
for (i in rows){
  plotdigit(datum[i, ])
}

#### ---- train test 
set.seed(1)
train_index = sample(seq(10), nrow(digit), replace = T)
length(train_index)

train = train_index <= 8
table(train)


#### ---- Linear regression
library(nnet)
str(digit$label)
mod.lm <- multinom(as.factor(label) ~ ., data = digit[train, ], , MaxNWts =1e4)
lm.predict <- predict(mod.lm, digit[!train, ], type = "class")
lm.predict
hist(as.numeric(as.character(lm.predict)), breaks = seq(-0.5, 9.5, by = 1))
print(confusionMatrix(as.factor(lm.predict), as.factor(digit[!train,]$label))) #0.7825

# 1.  Montrer la table de confusion entre valeurs observée et valeur prédite dans 
#    la base de test digit[!train, ]
# 2. Calculer l'accuracy du modèle


#### ---- Elastic Net
library(glmnet)
x_train <- model.matrix(label ~ ., digit[train, ])[, -1]
x_test <-  model.matrix(label ~ ., digit[!train, ])[, -1]
y_train <- as.character(digit[train, ]$label)

?glmnet

alphas = seq(0, 1, length = 10)
lambdas = c(0.1, 0.2, 0.5)
cv_index = sample(seq(5), nrow(x_train), replace = T)

acc = c()
alpha_cv = c()
lambda_cv = c()
for (alpha in alphas){
  for (lambda in lambdas){
    alpha_cv = c(alpha_cv, alpha)
    lambda_cv = c(lambda_cv, lambda)
    acc_cv = c()
    for (k in 1:5){
      x_train_cv = x_train[cv_index != k, ]
      y_train_cv = y_train[cv_index != k]
      
      x_test_cv = x_train[cv_index == k, ]
      y_test_cv = y_train[cv_index == k]
      
      model <- glmnet(x_train_cv, y_train_cv,
                      alpha = alpha, lambda = lambda, family="multinomial")
      
      #prediction
      pred = predict(model, x_test_cv, type="class")
      mcat = table(pred, y_test_cv)
      acc_cv = c(acc, sum(diag(mcat)) / sum(mcat)) #accuracy for 1 fold
    }
    acc = c(acc, mean(acc_cv))
  }
}

result = data.frame(alpha = alpha_cv,
                    lambda = lambda_cv,
                    acc = acc)
setDT(result)[order(result$acc)]

# 1. Fitter un modèle glmnet avec le paramètre alpha = 0.2222222, lambda = 0.2
# 2. Prédire sur la base test en utilisant le modèle déveppé dans l'étape 1
# 3. Créer la matrix de confusion


model <- glmnet(x_train, y_train, alpha = 0.22, lambda = 0.2, family="multinomial")
elas.predict = predict(model, x_test, type="class")
print(confusionMatrix(as.factor(elas.predict), as.factor(digit[!train,]$label))) #0.7426           

#### ---- Decision tree
library(rpart)
tree_mod <- rpart(
  formula = label ~ ., 
  data = digit[train, ],
  method = "class",  #classification
  control = rpart.control(cp = 0)
)

tree_pred = predict(tree_mod, digit[!train, ], type="class")
print(cmat <- confusionMatrix(tree_pred, as.factor(digit[!train,]$label))) #0.7301 


#### ---- randomForest
library(randomForest)
rf.model <- randomForest(as.factor(label) ~.,
                         data = digit[train,], ntree=500) #28 colonnes dans chaque arbre
rf.predict <- predict(rf.model, digit[!train, ])
print(cmat <- confusionMatrix(rf.predict, as.factor(digit[!train,]$label))) # 0.9339  


rf.model.2 <- randomForest(as.factor(label) ~.,
                         data = digit[train,], ntree=1000, mtry = 5) #5 colonnes dans chaque arbres
rf.predict.2 <- predict(rf.model.2, digit[!train, ])
print(cmat <- confusionMatrix(rf.predict.2, as.factor(digit[!train,]$label))) # 0.9237 


## Exercice maison: Grid search pour trouver les paramètres optimaux de randomForest



#### ---- Bagging
# C'est le randomForest avec mtry = nombre de variables

#### ---- XGBOOST
library(xgboost)
digit <- as.data.frame(lapply(digit, as.numeric))

data.train <- xgb.DMatrix(data = data.matrix(digit[train, 2:ncol(digit)]),
                          label = digit[train, ]$label)
data.test <- xgb.DMatrix(data = data.matrix(digit[!train, 2:ncol(digit)]),
                          label = digit[!train, ]$label)

watchlist = list(train = data.train, test = data.test)
parameters <- list(
  # General parameters
  booster              = "gbtree",
  # Booster parameters
  eta                  = 0.1,
  gamma                = 0,
  max_depth            = 4,
  min_child_weight     = 1,
  subsample            = 0.8,
  colsample_by_tree    = 0.8,
  lambda               = 1,
  alpha                = 0,
  # Task parameters
  objective            = "multi:softmax", #classification multiclass
  eval_metric          = "merror",
  num_class            = 10
)

set.seed(1)
xgb.model = xgb.train(parameters, data.train, nrounds = 500, early_stopping_rounds = 50, watchlist)

xgb.predict = predict(xgb.model, data.test)
print(mcat <- confusionMatrix(as.factor(xgb.predict),
                              as.factor(digit[!train,]$label))) #0.926         

# exercice maison: Hyper-parameters tuning

#### ---- Neural Network
library(h2o)
h2o.init()
digit$label = as.factor(digit$label)
h2o.train <- as.h2o(digit[train, ])
h2o.test <- as.h2o(digit[!train, ])

h2o.model <- h2o.deeplearning(x = setdiff(names(digit), "label"),
                              y = "label",
                              training_frame = h2o.train,
                              standardize = TRUE,
                              hidden = c(6),
                              rate = 0.05,
                              epochs = 10,
                              seed = 1234
)


h2o.predictions = as.data.frame(h2o.predict(h2o.model, h2o.test))
print(mcat <- confusionMatrix(as.factor(h2o.predictions$predict),  
                              as.factor(digit[!train,]$label))) #0.738  

# trouver une meilleure architecture du réseaux
# changer hidden, rate, epochs

h2o.model.2 <- h2o.deeplearning(x = setdiff(names(digit), "label"),
                                y = "label",
                                training_frame = h2o.train,
                                standardize = TRUE,
                                hidden = c(400, 200, 100),
                                rate = 0.05,
                                epochs = 10,
                                seed = 1234
)

h2o.predictions.2 = as.data.frame(h2o.predict(h2o.model.2, h2o.test))
print(mcat <- confusionMatrix(as.factor(h2o.predictions.2$predict),  
                              as.factor(digit[!train,]$label))) #0.9271            


# CNN: Convolutional Neural Network: filters lecture
