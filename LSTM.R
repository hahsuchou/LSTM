install.packages('devtools')
library(devtools)
library(magrittr)
install_github('datadotworld/dwapi-r')
library(dwapi)
install.packages('keras')
install_keras()
library(keras)
library(dplyr)
library(ggplot2)
install.packages('ggthemes')
library(ggthemes)
library(lubridate)
install.packages('tensorflow')
install_tensorflow()
library(tensorflow)
data <- read.csv('AAPL.csv') #read Data
summary(data) #summary
ggplot(data, aes(x = Date, y = Close)) + geom_point(color = '#56B4E9') + theme_grey(base_size = 16)

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
#眾數(mode)可以觀察出股價最常出現的價位
getmode(data$Close) # 118.93

#standard deviation，標準差和變異數用來觀察股價資料的離散程度
sd(data$Close) #24.35653

#variance
var(data$Close) #593.2408

Series = data$Close;Series

#LSTM 期望數據處於監督學習模式。
#也就是說，有一個目標變量 Y 和預測變量 X。
#為了實現這一點，我們通過滯後序列來轉換序列，並將時間 (t−k) 的值作為輸入，將時間 t 的值作為輸出，對於 k 步滯後數據集。
#create shift dataset, e.g. input:t-1, output:t
lag_transform <- function(x, k = 1){
  lagged = c(rep(NA, k), x[1:(length(x)- k )])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c(paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
#supervised = lag_transform(Series, 1);supervised

#transform data to stationarity
#x這是通過獲取系列中兩個連續值之間的差異來完成的。
#這種轉換（通常稱為差分）會刪除數據中與時間相關的組件。
#此外，使用差異而不是原始值更容易建模，並且生成的模型具有更高的預測能力。
#split transfer dataset and get slope
#diffed = diff(Series, differences = 1);diffed
#diffed = diff(Series, 1);diffed
diffed = diff(Series);diffed

supervised = lag_transform(diffed, 1);supervised

#Split dataset into training and testing sets
#The following code split the first 70% of the series as training set and the remaining 30% as test set.
N <- nrow(supervised);N #183
n <- round(N * 0.7, digits = 0);n #128
train <- supervised[1:n,];train
test <- supervised[(n+1):N,];test
class(train)
#Normalize the data. rescale function
#將輸入數據 X 重新縮放到激活函數的範圍。
#如前所述，LSTM 的默認激活函數是 sigmoid 函數，其範圍為 [-1, 1]。
#下面的代碼將有助於這種轉換。
#訓練數據集的最小值和最大值是用於縮放訓練和測試數據集以及預測值的縮放係數。
#這確保了測試數據的最小值和最大值不會影響模型。
scale_data <- function(train, test, feature_range = c(0, 1)){
  x <- train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x)) / (max(x) - min(x)))
  std_test = ((test - min(x)) / (max(x) - min(x)))
  
  scaled_train = std_train * (fr_max - fr_min) + fr_min
  scaled_test = std_test * (fr_max - fr_min) + fr_min
  
  return(list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test), scaler = c(min = min(x), max = max(x))))
  }
Scaled <- scale_data(train, test, c(-1, 1));Scaled
summary(Scaled) #重新縮放之後的結果為
y_train <- Scaled$scaled_train$`x`;y_train
x_train <- Scaled$scaled_train$`x-1`

y_test <- as.data.frame(Scaled$scaled_test$x);y_test
x_test <- as.data.frame(Scaled$scaled_test$x-1);x_test

#The following code will be required to revert the predicted values to the original scale.
#需要以下程式才能將預測值恢復到原始尺度。
#inverse-transform
invert_scaling <- function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  t = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(t)
  
  for(i in 1:t){
    X = (scaled[i] - mins) / (maxs - mins)
    rawValues = X * (max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}

#modeling
#將剛處理好的資料投入到模型中，進行訓練。
#Reshape the input to 3-dim
#Since the network is stateful, we have to provide the input batch in 3-dimensional array of the form 
#[samples, timesteps, features] from the current [samples, features],
dim(x_train) <- c(length(x_train), 1, 1);dim(x_train) #取得training的資料長度與維度(128, 1, 1)

#specify required arguments
X_shape1 = dim(x_train)[1];X_shape1 #128
X_shape2 = dim(x_train)[2];X_shape2 #1
X_shape3 = dim(x_train)[3];X_shape3 #1
batch_size = 1 # must be a common factor of both the train and test samples
units = 1 # can just this, in model tuning phase 神經元位數
units = 10

#建立模型
model <- keras_model_sequential();model #錯誤:Valid installation of TensorFlow not found.
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(learning_rate = 0.02, decay = 1e-6),
  metrics = c('accuracy')
)
summary(model)

#fit the model
#set the argument shuffle=FALSE to avoid shuffling the training set and maintain the dependencies between xi and xt+1.
#LSTM also requires resetting of the network state after each epoch.
#run a loop over epochs where within each epoch we fit the model and reset the state via the argument reset_states()
Epochs = 50
for(i in 1:Epochs){
  model %>% fit(x_train, y_train, epochs = 1, batch_size = batch_size, verbose = 1, shuffle = FALSE)
  model %>% reset_states()
}
