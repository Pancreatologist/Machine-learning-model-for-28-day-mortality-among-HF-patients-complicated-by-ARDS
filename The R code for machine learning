######一、特征选择######
####1.boruta#####
rm(list=ls())
library(ggsci)
library(here)
library(AppliedPredictiveModeling)
library(tidymodels)
library(ROSE)
library(smotefamily)
library(themis)
library(Boruta)
library(randomForest)
library(ggplot2)
library(reshape2)
library(dplyr)
library(tidyverse)
library(ggridges) 
library(ggtext)

predictors <- read.table('data_table_for_downstream.tsv',header =T)
#diagnosis是结局变量

#predictors$BMI <- predictors$weight/(predictors$height/100)^2

predictors$diagnosis <- predictors$death_within_hosp_28days



#x选取部分数据
data <- predictors[,c(65,4,12:53)] #分组变量放前面
#替换为自己的数据读取
#因子化
str(data$diagnosis)
library(Boruta)
#pValue 指定置信水平，mcAdj=TRUE 意为将应用 Bonferroni 方法校正 p 值
#此外还可以提供 mtry 和 ntree 参数的值，这些值将传递给随机森林函数 randomForest()
#大部分参数使用默认值


set.seed(202566)
data_total <- data  %>%  
  initial_split(prop = 0.8, strata = diagnosis) 
data_train <- training(data_total) #训练集
data_test <- testing(data_total) # 测试集
# 计算样本权重
positive_count <- sum(data_train$diagnosis == 1)
negative_count <- sum(data_train$diagnosis == 0)
scale_pos_weight <- negative_count / positive_count #3.197479
scale_pos_weight <- positive_count / negative_count #0.3127464
data_train$diagnosis <- as.factor(data_train$diagnosis)
set.seed(2025)
recipe_spec <- recipe(diagnosis ~ ., data = data_train) %>%
  themis::step_smote(diagnosis, over_ratio = 1) %>%
  step_corr(all_numeric_predictors(), 
            threshold = 0.8,  
            method = "pearson") %>%
  step_normalize(all_numeric_predictors()) %>% prep()

recipe_spec %>%
  bake(new_data = NULL) %>%
  count(diagnosis, name = "training")

# 3. 将训练好的转换应用到训练集和测试集
train_transformed <- bake(recipe_spec, new_data =NULL)
test_transformed <- bake(recipe_spec, new_data = data_test)

filter_variable <- colnames(train_transformed) 
filter_variable <- c("diagnosis", filter_variable[filter_variable != "diagnosis"])
data_train <-train_transformed[filter_variable]

#测试集的
filter_variable <- colnames(test_transformed)
filter_variable <- c("diagnosis", filter_variable[filter_variable != "diagnosis"])
data_test <- test_transformed[filter_variable]

set.seed(1234)

fs = Boruta(data_train[,-1],data_train$diagnosis,
            ###通过降低pValue或增加maxRuns，可以更好的区分tentative特征###
            pValue = 0.01,# 筛选阈值，默认0.01
            ### 迭代最大次数,先试运行一下，如果迭代完之后，还留有变量，则下次运行，增大迭代次数。##
            maxRuns = 100,
            mcAdj = TRUE, # Bonferroni方法进行多重比较校正
            doTrace = 2,# 跟踪算法进程
            holdHistory = TRUE, # TRUE,则将所有运行历史保存在结果的ImpHistory部分
            ##getImp设置获取特征重要性值的方法,可以赋值为自己写的功能函数名##
            ##需要设置优化参数,?getImpLegacyRf，查看更多参数设置注意事项##
            getImp =getImpLegacyRfGini,
            ##getImpRfZ()使用ranger进行随机森林分析，获得特征重要性值，默认返回mean decrease accuracy的Z-scores。##
            ##getImpRfRaw()使用ranger进行随机森林分析,默认返回原始置换重要性结果##
            ##getImpRfGini()使用ranger进行随机森林分析,默认返回Gini指数重要性结果##
            ##getImpLegacyRfZ()使用randomForest进行随机森林分析,默认返回均一化的分类准确性重要性结果##
            ##getImpLegacyRfRaw()使用randomForest进行随机森林分析,默认返回原始置换重要性结果##
            ##getImpLegacyRfGini()使用randomForest进行随机森林分析,默认返回Gini指数重要性结果##
            ##getImpFerns()使用rFerns包进行Random Ferns importance计算特征的重要性。它的运行速度比随机森林更快，必须优化depth参数，且可能需要的迭代次数更多##
            ##另外还有getImpXgboost,getImpExtraGini,getImpExtraZ,getImpExtraRaw等设置选项##
            ##参数设置为之前随机森林的调参结果,参数会传递给RandomForest函数##
            #importance = TRUE,
            #ntree=500,maxnodes=7, # 注释掉此句，使用默认参数进行变量筛选
            #mtry = 36 
            ##随着迭代的进行，剩余特征变量数<80后，函数会报警告信息，后续的迭代mtry将恢复默认值##
            ##介意warning信息，可以不用设置mtry##
)

table(fs$finalDecision) 
#绘制鉴定出的变量的重要性
plot(fs)
table(fs$finalDecision) 
boruta_result <- fs
# 绘制 Boruta 重要性历史图
Boruta::plotImpHistory(boruta_result)
# 获取 Boruta 重要性并整理数据
importance_df <- attStats(boruta_result)
importance_df$feature <- rownames(importance_df)
# 过滤出非阴影特征并进行数据转换
importance_df <- importance_df %>% filter(!grepl("^shadow", feature))
importance_long <- as.data.frame(boruta_result$ImpHistory) %>%
  pivot_longer(cols = everything(),
               names_to ="feature",
               values_to ="importance") %>%
  filter(!grepl("^shadow", feature)) %>% # 过滤掉阴影特征
  mutate(feature = gsub("^X","", feature)) %>% # 清理特征名称
  left_join(
    data.frame(
      feature = names(boruta_result$finalDecision),
      decision = factor(boruta_result$finalDecision,
                        levels = c("Confirmed","Tentative","Rejected"))
    ),
    by ="feature"
  )
# 计算每个特征的中位数重要性用于排序
feature_medians <- importance_long %>% group_by(feature) %>% summarise(
  median_imp = median(importance, na.rm = TRUE),
  decision = first(decision)) %>% arrange(desc(median_imp))
# 按中位数重要性重新排序特征因子
importance_long <- importance_long %>% mutate(feature = factor(feature, levels = feature_medians$feature))
confirmed_color <-"#79C377"# 绿色
tentative_color <-"#fdae61"# 橙黄色
rejected_color <-"#d7191c"# 红色

feature_colors <- data.frame(feature = levels(importance_long$feature),
                             stringsAsFactors = FALSE) %>%
  left_join(importance_long %>% select(feature, decision) %>% distinct(), by ="feature") %>%
  mutate(color_code = case_when(
    decision =="Confirmed"~ confirmed_color,
    decision =="Tentative"~ tentative_color,
    decision =="Rejected"~ rejected_color),
    colored_label = sprintf("<span style='color:%s; font-weight:bold'>%s</span>", color_code, feature)
  )

decision_plot <- ggplot(importance_long, aes(x = feature, y = importance, fill = decision)) +
  geom_boxplot(width = 0.7, alpha = 0.85, outlier.size = 1.5, outlier.color ="#404040",
               outlier.alpha = 0.6, notch = FALSE) +
  geom_hline(yintercept = 0, color ="grey50", linetype ="dashed") +
  scale_fill_manual(values = c("Confirmed"= confirmed_color,"Tentative"= tentative_color,
                               "Rejected"= rejected_color), name ="Feature Status") +
  scale_x_discrete(labels = setNames(feature_colors$colored_label, feature_colors$feature)) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.02))) +
  labs(title ="Boruta Feature Importance Analysis", x ="Variable", y ="Importance") +
  theme_minimal(base_size = 12) +
  theme(panel.border = element_rect(color ="black", fill = NA, linewidth = 0.8),
        axis.text.x = element_markdown(angle = 45, hjust = 1, vjust = 1),
        axis.title = element_text(face ="bold"),
        plot.title = element_markdown(size = 16, face ="bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color ="grey30", size = 11),
        axis.text.y = element_text(color ="black"),
        axis.title.x = element_text(margin = unit(c(10, 0, 0, 0),"pt"), color ="grey30"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        legend.position ="top",
        legend.title = element_text(face ="bold"),
        plot.background = element_rect(fill ="white", color = NA),
        plot.margin = unit(c(15, 20, 25, 15),"pt"),
        plot.caption = element_text(color ="grey50", size = 8)) 
# 显示并保存常规重要性结果图
print(decision_plot)
#保存
ggsave("./Boruta重要特征.pdf", decision_plot,width = 10,height = 5)
select.name = getSelectedAttributes(fs) # 获取标记为Confirmed的特征名。
select.name # 随机森林参数设置不同，结果有明显区别。选择分类正确率高的参数。
boruta.name <- select.name
plotdata <- importance_long %>% filter(decision =="Confirmed")
# 检查 Confirmed 变量是否存在
if(nrow(plotdata) == 0) { stop("没有找到 Confirmed 变量，请检查 Boruta 结果。") }
# 计算每个 Confirmed 特征的中位数重要性等统计指标用于排序
feature_medians <- plotdata %>% group_by(feature) %>% summarise(
  median_imp = median(importance, na.rm = TRUE),
  mean_imp = mean(importance, na.rm = TRUE),
  min_imp = min(importance),
  max_imp = max(importance)) %>% arrange(desc(median_imp))
# 再次确认只保留 Confirmed 变量
plotdata <- importance_long %>% filter(decision =="Confirmed")
# 再次检查 Confirmed 变量是否存在
if(nrow(plotdata) == 0) { stop("没有找到 Confirmed 变量，请检查 Boruta 结果。") }

#提取重要的变量
#boruta.finalVars <- data.frame(Item=getSelectedAttributes(fs, withTentative = F), Type="Boruta")
##holdHistory = TRUE,则可使用attStats()获取特征的重要性统计信息。
attStats(fs)
select.name = getSelectedAttributes(fs) # 获取标记为Confirmed的特征名。
select.name # 随机森林参数设置不同，结果有明显区别。选择分类正确率高的参数。

boruta.name <- select.name
#提取筛选出的变量
newdd <- merge(data_train[1],data_train[select.name],by = "row.names",all = F)
#第一列变为列名
rownames(newdd) <- newdd[,1]
newdd <- newdd[,-1]
#导出
write.table(newdd,file="Boruta_variablelist.txt",sep="\t",quote=F,col.names=T)

newddtest <- merge(data_test[1],data_test[select.name],by = "row.names",all = F)
#第一列变为列名
rownames(newddtest) <- newddtest[,1]
newddtest <- newddtest[,-1]
#导出
write.table(newddtest,file="Boruta_variablelist_test.txt",sep="\t",quote=F,col.names=T)

####2.lasso#####
str(data_train$diagnosis)
data_train$diagnosis <- as.numeric(data_train$diagnosis)
#构建模型(第二列开始是自变量)
x=as.matrix(data[,c(2:ncol(data))])
# 第一列是分组
y=data[,1]
#读入包
library(glmnet)
library(pROC)
#构建模型, alpha=1：进行LASSO回归；alpha=0，进行岭回归
fit=glmnet(x, y, family = "binomial", alpha=1)
plot(fit, xvar="lambda", label=T)
#10折交叉验证

cvfit=cv.glmnet(x, y, family="binomial", alpha=1,nfolds = 10)
pdf(file="cvfit.pdf",width=6,height=5.5)
plot(cvfit)
dev.off()
#提取两个λ值：
lambda.min <- cvfit$lambda.min
lambda.1se <- cvfit$lambda.1se
lambda.min
lambda.1se
#指定λ值重新构建模型(通过λ值筛选基因)：
model_lasso_min <- glmnet(x, y, alpha = 1, lambda = lambda.min,family = "binomial")
model_lasso_1se <- glmnet(x, y, alpha = 1, lambda = lambda.1se,family = "binomial")
#拎出模型使用的变量(存放在beta中)：
head(model_lasso_min$beta)#"."表示这个变量没有被使用
#使用as.numeric把.转化为0，然后通过筛选非0的方式将构建模型所使用的变量提取出来。
ID_min <- rownames(model_lasso_min$beta)[as.numeric(model_lasso_min$beta)!=0]#as.numeric后"."会转化为0
length(ID_min)
ID_1se <- rownames(model_lasso_1se$beta)[as.numeric(model_lasso_1se$beta)!=0]
length(ID_1se)
#提取基于最小拉姆达值所筛选的变量
newdd <- data[ID_1se]
#添加分组
#根据ID,合并
newdd = merge(data$diagnosis,newdd,by = "row.names",all = F)
#第一列变为列名
rownames(newdd) <- newdd[,1]
newdd <- newdd[,-1]
library(dplyr)
newdd <- newdd %>% rename(diagnosis = x)

#导出
out=rbind(ID=colnames(newdd),newdd)
write.table(out,file="Lasso_variablelist_1se.txt",sep="\t",quote=F,col.names=F)

### 取boruta跟LASSO的交集
interesect.name <- intersect(boruta.name,ID_1se)
newdd <- data_train[interesect.name]

#添加分组
#根据ID,合并
newdd = merge(data_train$diagnosis,newdd,by = "row.names",all = F)
#第一列变为列名
rownames(newdd) <- newdd[,1]
newdd <- newdd[,-1]
library(dplyr)
newdd <- newdd %>% rename(diagnosis = x)
out=rbind(ID=colnames(newdd),newdd)
#导出
write.table(out,file="boruta跟LASSO的交集_训练集.txt",sep="\t",quote=F,col.names=F)
newddtest <- merge(data_test[1],data_test[interesect.name],by = "row.names",all = F)
#第一列变为列名
rownames(newddtest) <- newddtest[,1]
newddtest <- newddtest[,-1]
#导出
write.table(newddtest,file="boruta跟LASSO的交集_测试集.txt",sep="\t",quote=F,col.names=T)
save.image('boruta跟LASSO的交集.RData')

library(tidymodels)
library(probably)
library(future) # 用于并行计算
library(ggplot2)
library(xgboost)
library(tidymodels)
library(probably)
library(future) # 用于并行计算
library(ggplot2)
library(kernlab)
library(bonsai)
### begin with DNN
set.seed(987)
plan(multisession)
data_train <- readRDS("E:/MIMIC_yw/data_train.rds")
data_test <- readRDS("E:/MIMIC_yw/data_test.rds")
data_train$diagnosis <- factor(data_train$diagnosis)
library(doParallel)
cl <- makePSOCKcluster(parallel::detectCores() - 1)  # 使用除1个核心外的所有核心
registerDoParallel(cl)
rec <- 
  recipe(diagnosis~ ., data = data_train)
cv_folds <- vfold_cv(data_train, v = 10, repeats = 20, strata = diagnosis)
# 指定 XGBoost 模型 
xgb_model <- boost_tree(
  trees = tune(),
  learn_rate = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification") 

xgb_workflow <- workflow() %>% 
  add_recipe(rec) %>% 
  add_model(xgb_model) 

xgb_tuned <- tune_grid(
  xgb_workflow,
  resamples = cv_folds,
  control = control_grid(save_pred = TRUE, verbose = FALSE),
  metrics = metric_set(brier_class, accuracy, roc_auc)
) 

autoplot(xgb_tuned) ##保存6*5 in 的pdf
best_params <- select_best(xgb_tuned, metric = "brier_class") 

final_xgb_workflow <- finalize_workflow(
  xgb_workflow,
  best_params
) 

final_xgb_fit <- fit(final_xgb_workflow, data = data_train) 
xgb_test_preds <- predict(final_xgb_fit, data_test, type = "prob") %>% # augment() 在这里也适用 
  bind_cols(diagnosis = as.factor(data_test$diagnosis))


# 指定SVM模型，使用径向基核函数（RBF）
svm_model <- svm_rbf(
  cost = tune(), # 需要调整的超参数：惩罚因子
  rbf_sigma = tune() # 需要调整的超参数：核函数的带宽
) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification") 

# 构建工作流 
svm_workflow <- workflow() %>% 
  add_recipe(rec) %>% 
  add_model(svm_model) 

svm_tuned <- tune_grid(
  svm_workflow, 
  resamples = cv_folds, 
  control = control_grid(save_pred = TRUE, verbose = FALSE),
  metrics = metric_set(brier_class, accuracy, roc_auc) 
) 

autoplot(svm_tuned) 
best_params <- select_best(svm_tuned, metric = "brier_class") 

final_svm_workflow <- finalize_workflow( 
  svm_workflow, 
  best_params 
) 

final_svm_fit <- fit(final_svm_workflow, data = data_train) 
svm_test_preds <- predict(final_svm_fit, data_test, type = "prob") %>% 
  bind_cols(diagnosis = as.factor(data_test$diagnosis))


#logistic_model
logistic_model <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

logistic_workflow <- workflow() %>%
  add_recipe(rec) %>%
  add_model(logistic_model)
logistic_tuned <- tune_grid(
  logistic_workflow,
  resamples = cv_folds,
  control = control_grid(save_pred = TRUE, verbose = FALSE),
  metrics = metric_set(brier_class, accuracy, roc_auc)
)

autoplot(logistic_tuned)
best_params <- select_best(logistic_tuned, metric = "brier_class")

final_logistic_workflow <- finalize_workflow(
  logistic_workflow,
  best_params
)

final_logistic_fit <- fit(final_logistic_workflow, data = data_train)

logistic_preds <- predict(final_logistic_fit, data_test, type = "prob") %>%
  bind_cols(diagnosis = as.factor(data_test$diagnosis))

# rf_model
rf_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 1000
) %>%
  set_engine("ranger",  importance = "impurity") %>%
  set_mode("classification")

# 
rf_workflow <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_model)

rf_tuned <- tune_grid(
  rf_workflow,
  resamples = cv_folds,
  control = control_grid(save_pred = TRUE, verbose = FALSE),
  metrics = metric_set(brier_class, accuracy, roc_auc)
)

autoplot(rf_tuned)
best_params <- select_best(rf_tuned, metric = "brier_class")
final_rf_workflow <- finalize_workflow(
  rf_workflow,
  best_params
)
final_rf_fit <- fit(final_rf_workflow, data = data_train)
rf_test_preds <- predict(final_rf_fit, data_test,type = 'prob') %>% # augment() 在这里也适用
  bind_cols(diagnosis = as.factor(data_test$diagnosis))

#LightGBM
lgbm_model <- 
  boost_tree(
    mtry = tune(), 
    trees = tune(), 
    tree_depth = tune(),
    learn_rate = tune(), 
    min_n = tune(), 
    loss_reduction = tune()
  ) %>%
  set_engine("lightgbm", verbose = -1) %>%  
  set_mode("classification")
lgbm_workflow <- 
  workflow() %>%
  add_recipe(rec) %>%  
  add_model(lgbm_model)

lgbm_tuned <- tune_grid(
  object = lgbm_workflow,
  resamples = cv_folds,
  grid = 5,  # 添加网格大小（例如20个随机组合）
  control = control_grid(save_pred = TRUE, verbose = FALSE),
  metrics = metric_set(brier_class, accuracy, roc_auc)
)

autoplot(lgbm_tuned)
best_params <- select_best(lgbm_tuned, metric = "brier_class")

final_lgbm_workflow <- finalize_workflow(
  lgbm_workflow,
  best_params
)

final_lgbm_fit <- fit(final_lgbm_workflow, data = data_train)
lgbm_test_preds <- predict(final_lgbm_fit, data_test, type = "prob") %>% # augment() 在这里也适用
  bind_cols(diagnosis = as.factor(data_test$diagnosis))


#
library(C50)  # Adaboost需要这个库
library(dials)  # 用于参数调优
# 定义Adaboost模型
ada_model <- boost_tree(
  trees = tune(),   
  min_n = tune()
) %>% 
  set_engine("C5.0") %>%  # 使用C5.0引擎实现Adaboost
  set_mode("classification")

# 创建工作流
ada_workflow <- workflow() %>% 
  add_recipe(rec) %>% 
  add_model(ada_model)

# 超参数调优
ada_tuned <- tune_grid(
  ada_workflow,
  resamples = cv_folds,
  control = control_grid(save_pred = TRUE, verbose = FALSE),
  metrics = metric_set(brier_class, accuracy, roc_auc)
)

# 可视化调优结果
autoplot(ada_tuned)

# 选择最佳参数
best_ada_params <- select_best(ada_tuned, metric = "brier_class")

# 用最佳参数更新工作流
final_ada_workflow <- finalize_workflow(
  ada_workflow,
  best_ada_params
)

# 训练最终模型
final_ada_fit <- fit(final_ada_workflow, data = data_train)

# 在测试集上进行预测
ada_test_preds <- predict(final_ada_fit, data_test, type = "prob") %>% 
  bind_cols(diagnosis = as.factor(data_test$diagnosis))

###summary data

xgb_pred <- xgb_test_preds %>% select(.pred_1) %>% rename(xgb = .pred_1)
svm_pred <- svm_test_preds %>% select(.pred_1) %>% rename(svm = .pred_1)
logistic_pred <- logistic_preds %>% select(.pred_1) %>% rename(logistic = .pred_1)
rf_pred <- rf_test_preds %>% select(.pred_1) %>% rename(rf = .pred_1)
lgbm_pred <- lgbm_test_preds %>% select(.pred_1) %>% rename(lgbm = .pred_1)
ada_pred <- ada_test_preds %>% select(.pred_1) %>% rename(dnn = .pred_1)
final_preds <- bind_cols(ada_pred, xgb_pred, svm_pred, logistic_pred, rf_pred, lgbm_pred) %>% 
  mutate(diagnosis=ada_test_preds$diagnosis)

writexl::write_xlsx(final_preds,'20250909.xlsx')

library(tidymodels)
library(ggsci)
library(probably)
library(ResourceSelection)
library(dcurves)
library(gtsummary)
library(cowplot)

results <- readxl::read_excel('20250909.xlsx')
results$diagnosis <- factor(results$diagnosis,levels = c(0, 1))

#-----1.混淆矩阵------
###Logistic 预测结果#######
pred_lr <- ifelse(results$logistic < 0.5, 1, 0)   

#  Logistic 创建混淆矩阵  

confusion_matrix_lr <- caret::confusionMatrix(
  factor(pred_lr,levels = c(0, 1),labels = c("0","1")), 
  results$diagnosis,
  positive = "1")   
print(confusion_matrix_lr) 


# 获取指标
metrics_lr <- data.frame(
  Accuracy = confusion_matrix_lr$overall["Accuracy"],
  Precision = confusion_matrix_lr$byClass["Precision"],
  Recall = confusion_matrix_lr$byClass["Recall"],
  F1 = confusion_matrix_lr$byClass["F1"],
  AUC = pROC::auc(response = results$diagnosis, predictor = results$logistic)
)
metrics_lr
# 转置为垂直格式
final_metrics_lr <- metrics_lr %>% 
  t() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Metric") %>%
  rename(Value = "Accuracy")

# 保存结果
write.csv(final_metrics_lr, "lr_metrics.csv", row.names = FALSE)

# 将混淆矩阵转换为数据框
conf_matrix_df_lr <- as.data.frame(confusion_matrix_lr$table)

# 使用ggplot2绘制混淆矩阵热图
conf_plot_lr <- ggplot(conf_matrix_df_lr, aes(x =Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "#1E90FF") +  # 使用更鲜艳的颜色
  labs(
    title = "Confusion Matrix for Logistic Model",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold", color = "darkblue"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 12, hjust = 0.5, face = "italic", color = "darkred"),
    plot.background = element_rect(fill = "grey95"),  # 设置背景颜色
    panel.background = element_rect(fill = "white")  # 设置面板背景颜色
  )
conf_plot_lr

# 保存图片
ggsave("./混淆矩阵_lr.pdf", conf_plot_lr, width = 6, height = 6)

### 随机森林模型 预测结果#####
# 在测试集上评估模型的区分能力
prob_rf <- ifelse(results$rf > 0.5, 1, 0)  
#  随机森林模型 创建混淆矩阵  
confusion_matrix_rf <- caret::confusionMatrix(
  factor(prob_rf,levels = c(0, 1),labels = c("0","1")), 
  results$diagnosis,
  positive = "1")   
print(confusion_matrix_rf) 



# 获取指标
metrics_rf <- data.frame(
  Accuracy = confusion_matrix_rf$overall["Accuracy"],
  Precision = confusion_matrix_rf$byClass["Precision"],
  Recall = confusion_matrix_rf$byClass["Recall"],
  F1 = confusion_matrix_rf$byClass["F1"],
  AUC = pROC::auc(response = results$diagnosis, predictor = results$rf)
)
metrics_rf
# 转置为垂直格式
final_metrics_rf <- metrics_rf %>% 
  t() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Metric") %>%
  rename(Value = "Accuracy")

# 保存结果
write.csv(final_metrics_rf, "rf_metrics.csv", row.names = FALSE)

# 将混淆矩阵转换为数据框
conf_matrix_df_rf <- as.data.frame(confusion_matrix_rf$table)

# 使用ggplot2绘制混淆矩阵热图
conf_plot_rf <- ggplot(conf_matrix_df_rf, aes(x =Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "#1E90FF") +  # 使用更鲜艳的颜色
  labs(
    title = "Confusion Matrix for Random Forest Model",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold", color = "darkblue"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 12, hjust = 0.5, face = "italic", color = "darkred"),
    plot.background = element_rect(fill = "grey95"),  # 设置背景颜色
    panel.background = element_rect(fill = "white")  # 设置面板背景颜色
  )
conf_plot_rf

# 保存图片
ggsave("./混淆矩阵_rf.pdf", conf_plot_lr, width = 6, height = 6)

###支持向量机 预测结果#######
prob_svm <- ifelse(results$svm > 0.5, 1, 0)  

#  支持向量机 创建混淆矩阵  
confusion_matrix_svm <- caret::confusionMatrix(
  factor(prob_svm,levels =c("0", "1"),labels = c("0","1")), 
  results$diagnosis,
  positive = "1")   # 训练集
print(confusion_matrix_svm) 


# 获取指标
metrics_svm <- data.frame(
  Accuracy = confusion_matrix_svm$overall["Accuracy"],
  Precision = confusion_matrix_svm$byClass["Precision"],
  Recall = confusion_matrix_svm$byClass["Recall"],
  F1 = confusion_matrix_svm$byClass["F1"],
  AUC = pROC::auc(response = results$diagnosis, predictor = prob_svm)
)
metrics_svm
# 转置为垂直格式
final_metrics_svm <- metrics_svm %>% 
  t() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Metric") %>%
  rename(Value = "Accuracy")

# 保存结果
write.csv(final_metrics_svm, "svm_metrics.csv", row.names = FALSE)

# 将混淆矩阵转换为数据框
conf_matrix_df_svm <- as.data.frame(confusion_matrix_svm$table)

# 使用ggplot2绘制混淆矩阵热图
conf_plot_svm <- ggplot(conf_matrix_df_svm, aes(x =Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "#1E90FF") +  # 使用更鲜艳的颜色
  labs(
    title = "Confusion Matrix for SVM Model",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold", color = "darkblue"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 12, hjust = 0.5, face = "italic", color = "darkred"),
    plot.background = element_rect(fill = "grey95"),  # 设置背景颜色
    panel.background = element_rect(fill = "white")  # 设置面板背景颜色
  )
conf_plot_svm

# 保存图片
ggsave("./混淆矩阵_svm.pdf", conf_plot_lr, width = 6, height = 6)



###XGboost 预测结果#######
pred_xgb <- ifelse(results$xgb > 0.5, 1, 0)  
# 创建混淆矩阵  
confusion_matrix_xgb <- caret::confusionMatrix(factor(pred_xgb,levels = c(0, 1),labels = c("0","1")), 
                                               as.factor(results$diagnosis), 
                                               positive = "1")  
print(confusion_matrix_xgb) 

# 获取指标
metrics_xgb <- data.frame(
  Accuracy = confusion_matrix_xgb$overall["Accuracy"],
  Precision = confusion_matrix_xgb$byClass["Precision"],
  Recall = confusion_matrix_xgb$byClass["Recall"],
  F1 = confusion_matrix_xgb$byClass["F1"],
  AUC = pROC::auc(response = results$diagnosis, predictor = results$xgb)
)
metrics_xgb 
# 转置为垂直格式
final_metrics_xgb <- metrics_xgb %>% 
  t() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Metric") %>%
  rename(Value = "Accuracy")

# 保存结果
write.csv(final_metrics_xgb, "xgb_metrics_train.csv", row.names = FALSE)

# 将混淆矩阵转换为数据框
conf_matrix_df_xgb <- as.data.frame(confusion_matrix_xgb$table)

# 使用ggplot2绘制混淆矩阵热图
conf_plot_xgb <- ggplot(conf_matrix_df_xgb, aes(x =Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "#1E90FF") +  # 使用更鲜艳的颜色
  labs(
    title = "Confusion Matrix for XGboost Model",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold", color = "darkblue"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 12, hjust = 0.5, face = "italic", color = "darkred"),
    plot.background = element_rect(fill = "grey95"),  # 设置背景颜色
    panel.background = element_rect(fill = "white")  # 设置面板背景颜色
  )
conf_plot_xgb



# 保存图片
ggsave("./混淆矩阵_xgb.pdf", conf_plot_xgb, width = 6, height = 6)

###LightGBM 预测结果#######
pred_lgb <- ifelse(results$lgbm > 0.5, 1, 0)  
# 创建混淆矩阵  
confusion_matrix_lgb <- caret::confusionMatrix(factor(pred_lgb,levels = c(0, 1),labels = c("0","1")), 
                                               as.factor(results$diagnosis), 
                                               positive = "1")  
print(confusion_matrix_lgb) 

# 获取指标
metrics_lgb <- data.frame(
  Accuracy = confusion_matrix_lgb$overall["Accuracy"],
  Precision = confusion_matrix_lgb$byClass["Precision"],
  Recall = confusion_matrix_lgb$byClass["Recall"],
  F1 = confusion_matrix_lgb$byClass["F1"],
  AUC = pROC::auc(response = results$diagnosis, predictor =  results$lgbm)
)
metrics_lgb 
# 转置为垂直格式
final_metrics_lgb <- metrics_lgb %>% 
  t() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Metric") %>%
  rename(Value = "Accuracy")

# 保存结果
write.csv(final_metrics_lgb, "lgb_metrics_train.csv", row.names = FALSE)

# 将混淆矩阵转换为数据框
conf_matrix_df_lgb <- as.data.frame(confusion_matrix_lgb$table)

# 使用ggplot2绘制混淆矩阵热图
conf_plot_lgb <- ggplot(conf_matrix_df_lgb, aes(x =Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "#1E90FF") +  # 使用更鲜艳的颜色
  labs(
    title = "Confusion Matrix for LightGBM Model",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold", color = "darkblue"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 12, hjust = 0.5, face = "italic", color = "darkred"),
    plot.background = element_rect(fill = "grey95"),  # 设置背景颜色
    panel.background = element_rect(fill = "white")  # 设置面板背景颜色
  )
conf_plot_lgb

ggsave("./混淆矩阵_Lgb.pdf", conf_plot_lgb, width = 6, height = 6)


###ada 预测结果#######
pred_ada <- ifelse(results$ada > 0.5, 1, 0) 

# 创建混淆矩阵  
confusion_matrix_ada <- caret::confusionMatrix(factor(pred_ada,levels = c(0, 1),labels = c("0","1")), 
                                               as.factor(results$diagnosis), 
                                               positive = "1")  
print(confusion_matrix_ada) 

# 获取指标
metrics_ada <- data.frame(
  Accuracy = confusion_matrix_ada$overall["Accuracy"],
  Precision = confusion_matrix_ada$byClass["Precision"],
  Recall = confusion_matrix_ada$byClass["Recall"],
  F1 = confusion_matrix_ada$byClass["F1"],
  AUC = pROC::auc(response = results$diagnosis, predictor = results$ada)
)
metrics_ada 
# 转置为垂直格式
final_metrics_ada <- metrics_ada %>% 
  t() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Metric") %>%
  rename(Value = "Accuracy")

# 保存结果
write.csv(final_metrics_ada, "ada_metrics_train.csv", row.names = FALSE)

# 将混淆矩阵转换为数据框
conf_matrix_df_ada <- as.data.frame(confusion_matrix_ada$table)

# 使用ggplot2绘制混淆矩阵热图
conf_plot_ada <- ggplot(conf_matrix_df_ada, aes(x =Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "#1E90FF") +  # 使用更鲜艳的颜色
  labs(
    title = "Confusion Matrix for Adaboost Model",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold", color = "darkblue"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 12, hjust = 0.5, face = "italic", color = "darkred"),
    plot.background = element_rect(fill = "grey95"),  # 设置背景颜色
    panel.background = element_rect(fill = "white")  # 设置面板背景颜色
  )
conf_plot_ada
ggsave("./混淆矩阵_ada.pdf", conf_plot_ada, width = 6, height = 6)
###MELD3 预测结果#######
colnames(results)[8] <- 'MELD'
results$MELD <- as.numeric(results$MELD)
meld_model <- glm(diagnosis~ MELD, data = results, family = binomial(link = "logit"))
results$meld_probability <- predict(meld_model, type = "response")

pred_meld <- ifelse(results$meld_probability > 0.5, 1, 0) 

# 创建混淆矩阵  
confusion_matrix_meld <- caret::confusionMatrix(factor(pred_meld,levels = c(0, 1),labels = c("0","1")), 
                                               as.factor(results$diagnosis), 
                                               positive = "1")  
print(confusion_matrix_meld) 

# 获取指标
metrics_meld <- data.frame(
  Accuracy = confusion_matrix_meld$overall["Accuracy"],
  Precision = confusion_matrix_meld$byClass["Precision"],
  Recall = confusion_matrix_meld$byClass["Recall"],
  F1 = confusion_matrix_meld$byClass["F1"],
  AUC = pROC::auc(response = results$diagnosis, predictor = results$meld_probability)
)
metrics_meld
# 转置为垂直格式
final_metrics_meld <- metrics_meld %>% 
  t() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Metric") %>%
  rename(Value = "Accuracy")

# 保存结果
write.csv(final_metrics_meld, "meld_metrics_train.csv", row.names = FALSE)

# 将混淆矩阵转换为数据框
conf_matrix_df_meld <- as.data.frame(confusion_matrix_meld$table)

# 使用ggplot2绘制混淆矩阵热图
conf_plot_meld <- ggplot(conf_matrix_df_meld, aes(x =Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 6) +
  scale_fill_gradient(low = "white", high = "#1E90FF") +  # 使用更鲜艳的颜色
  labs(
    title = "Confusion Matrix for MELD 3.0",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold", color = "darkblue"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 12, hjust = 0.5, face = "italic", color = "darkred"),
    plot.background = element_rect(fill = "grey95"),  # 设置背景颜色
    panel.background = element_rect(fill = "white")  # 设置面板背景颜色
  )
conf_plot_meld

# 保存图片
ggsave("./混淆矩阵_meld.pdf", conf_plot_meld, width = 6, height = 6)


#-----2.单次ROC曲线绘制------
# 计算ROC对象
library(PRROC)
library(ROCR)
library(pROC)
roc_lr <- roc(results$diagnosis, as.numeric(results$logistic))
roc_rf <- roc(results$diagnosis, as.numeric(results$rf))
roc_svm <- roc(results$diagnosis,as.numeric(results$svm))
roc_xgb <- roc(results$diagnosis, as.numeric(results$xgb))
roc_lgb <- roc(results$diagnosis, as.numeric(results$lgbm))
roc_ada <- roc(results$diagnosis, as.numeric(results$ada))
roc_meld <- roc(results$diagnosis, as.numeric(results$meld_probability))


# 计算AUC值
auc_lr <- roc_lr$auc 
auc_rf <- roc_rf$auc 
auc_svm <- roc_svm$auc 
auc_xgb <- roc_xgb$auc 
auc_lgb <- roc_lgb$auc 
auc_ada <- roc_ada$auc 
auc_meld <- roc_meld$auc
ci_lr <- ci.auc(auc_lr)
ci_rf <- ci.auc(auc_rf)
ci_svm <- ci.auc(auc_svm)
ci_xgb <- ci.auc(auc_xgb)
ci_lgb <- ci.auc(auc_lgb)
ci_ada <- ci.auc(auc_ada)
ci_meld <- ci.auc(auc_meld)


# 创建数据框，包含每个模型的 AUC 值和 ROC 对象
roc_data <- list(
  Model = c("Logistic Regression","Random Forest","SVM", 
            "XGBoost","LightGBM",
            'Adaboost',
            'MELD'),
  AUC = c(auc_lr,auc_rf,auc_svm, auc_xgb,auc_lgb,
          auc_ada, auc_meld),  # 实际模型的 AUC 值
  CI_Lower = c(ci_lr[1], ci_rf[1], ci_svm[1], ci_xgb[1], ci_lgb[1], ci_ada[1],ci_meld[1]),
  CI_Upper = c(ci_lr[3], ci_rf[3], ci_svm[3], ci_xgb[3], ci_lgb[3], ci_ada[3],ci_meld[3]),
  ROC = list(roc_lr,roc_rf,roc_svm, roc_xgb,roc_lgb,
             roc_ada,roc_meld)  # 实际模型的 ROC 对象
)

roc_data$Model <- factor(roc_data$Model,
                         levels = c("Logistic Regression","XGBoost",
                                    "Random Forest","LightGBM",
                                    "SVM",'Adaboost','MELD'))
# 提取 ROC 曲线的数据并转换为数据框
roc_curves <- lapply(seq_along(roc_data$ROC), function(i) {
  roc_obj <- roc_data$ROC[[i]]
  data.frame(
    Sensitivity = roc_obj$sensitivities,
    Specificity = roc_obj$specificities,
    Model = roc_data$Model[i]
  )
})

# 将列表合并为一个数据框
roc_curves <- do.call(rbind, roc_curves)

# 对 ROC 对象进行平滑处理
smoothed_rocs <- lapply(roc_data$ROC, smooth, method = "density")  # 使用密度平滑方法

# 添加 AUC 值到图例标签
roc_data$Label <- paste(roc_data$Model, " (AUC [95%CI]= ", round(roc_data$AUC, 3),", ",
                        round(roc_data$CI_Lower, 3),"-",
                        round(roc_data$CI_Upper, 3), ")", sep = "")

# 绘制平滑 ROC 曲线
roc_plot <- ggroc(smoothed_rocs, linetype = 1, size = 1.2) +
  scale_color_bmj(labels = roc_data$Label) +  # 使用 Lancet 颜色主题并添加 AUC 标签
  labs(title = "ROC Curves for Models",
       x = "1 - Specificity",
       y = "Sensitivity") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.position = "bottom",
    legend.text = element_text(size = 10)
  ) +
  guides(color = guide_legend(ncol = 1, byrow = TRUE))  # 图例分两列显示

# 显示图形
roc_plot

# 保存图片
ggsave("./单次ROC.pdf", roc_plot, width = 5, height = 6)

#-----3.校准曲线绘制------
library(scales)   # 提供刻度格式化工具

# 预测结果及真实标签汇总为一个数据框
calibration_data <- data.frame(
  Model = c(rep("Logistic Regression", length(results$logistic)),
            rep("Random Forest", length(results$rf)),
            rep("SVM", length(results$svm)),
            rep("XGBoost", length(results$xgb)),
            rep("LightGBM", length(results$lgbm)),
            rep("Adaboost", length(results$ada)),
            rep("MELD", length(results$meld_probability))
  ),
  Probability = c(results$logistic, 
                  results$rf,
                  results$svm,
                  results$xgb,
                  results$lgbm,
                  results$ada,
                  results$meld_probability),
  Actual = as.numeric(c(results$diagnosis)) - 1  # 将因子转为数值
)

results$recalibrated_lr_probs <- predict(glm(results$diagnosis ~ results$logistic, family = binomial),type = "response")
results$recalibrated_rf_probs <- predict(glm(results$diagnosis ~ results$rf, family = binomial),type = "response")
results$recalibrated_svm_probs <- predict(glm(results$diagnosis ~ results$svm, family = binomial),type = "response")
results$recalibrated_xgb_probs <- predict(glm(results$diagnosis ~ results$xgb, family = binomial),type = "response")
results$recalibrated_lgb_probs <- predict(glm(results$diagnosis ~ results$lgbm, family = binomial),type = "response")
results$recalibrated_ada_probs <- predict(glm(results$diagnosis ~ results$ada, family = binomial),type = "response")
results$recalibrated_meld_probability_probs <- predict(glm(results$diagnosis ~ results$meld_probability, family = binomial),type = "response")
recalibration_data <- data.frame(
  Model = c(rep("Logistic Regression", length(results$recalibrated_lr_probs)),
            rep("Random Forest", length(results$recalibrated_rf_probs)),
            rep("SVM", length(results$recalibrated_svm_probs)),
            rep("XGBoost", length(results$recalibrated_xgb_probs)),
            rep("LightGBM", length(results$recalibrated_lgb_probs)),
            rep("Adaboost", length(results$recalibrated_ada_probs)),
            rep("MELD", length(results$recalibrated_meld_probability_probs))
  ),
  Probability = c(results$recalibrated_lr_probs, 
                  results$recalibrated_rf_probs,
                  results$recalibrated_svm_probs,
                  results$recalibrated_xgb_probs,
                  results$recalibrated_lgb_probs,
                  results$recalibrated_ada_probs,
                  results$recalibrated_meld_probability_probs),
  Actual = as.numeric(c(results$diagnosis)) - 1  # 将因子转为数值
)

re_cal_plot <- cal_plot_logistic(recalibration_data, truth = Actual, estimate = Probability,
                  .by=Model, 
                  smooth =F
)  +scale_color_bmj()+ facet_wrap(~ Model, ncol =4)
calibration_data$Model <- factor(calibration_data$Model,
                                 levels = c("Logistic Regression","Random Forest","SVM","XGBoost",
                                            "LightGBM",
                                            "Adaboost","MELD"))
recalibration_data$Model <- factor(recalibration_data$Model,
                                 levels = c("Logistic Regression","Random Forest","SVM","XGBoost",
                                            "LightGBM",
                                            "Adaboost","MELD"))
cal_plot <- cal_plot_logistic(calibration_data, truth = Actual, estimate = Probability,
                              .by=Model, 
                              smooth =F
)  +scale_color_bmj()+ facet_wrap(~ Model, ncol =4)
re_cal_plot <- cal_plot_logistic(recalibration_data, truth = Actual, estimate = Probability,
                                 .by=Model, 
                                 smooth =F
)  +scale_color_bmj()+ facet_wrap(~ Model, ncol =4)

# 定义计算各指标的函数
calculate_brier_score <- function(probabilities, actual) {
  mean((probabilities - actual)^2)
}
calculate_log_loss <- function(probabilities, actual) {
  # 将预测概率裁剪到 (0, 1) 范围内
  probabilities <- pmax(pmin(probabilities, 1 - 1e-10), 1e-10)
  -mean(actual * log(probabilities) + (1 - actual) * log(1 - probabilities))
}
calculate_calibration_slope <- function(probabilities, actual) {
  # 将预测概率裁剪到 (0, 1) 范围内
  probabilities <- pmax(pmin(probabilities, 1 - 1e-10), 1e-10)
  fit <- glm(actual ~ log(probabilities / (1 - probabilities)), family = binomial)
  coef(fit)[2]
}
# calculate_calibration_intercept 函数
calculate_calibration_intercept <- function(probabilities, actual) {
  # 将预测概率裁剪到 (0, 1) 范围内
  probabilities <- pmax(pmin(probabilities, 1 - 1e-10), 1e-10)
  # 使用 suppressWarnings 忽略 glm.fit 的警告
  suppressWarnings({
    fit <- glm(actual ~ 1, offset = log(probabilities / (1 - probabilities)), family = binomial)
    coef(fit)[1]
  })
}

calculate_hl_test <- function(probabilities, actual, n_groups = 10) {
  hoslem.test(actual, probabilities, g = n_groups)$p.value
}
calculate_ece <- function(probabilities, actual, n_bins = 10) {
  bin_indices <- cut(probabilities, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE)
  ece <- 0
  for (i in levels(bin_indices)) {
    bin_samples <- bin_indices == i
    if (sum(bin_samples) > 0) {  # 确保分箱中有样本
      bin_prob <- mean(probabilities[bin_samples])
      bin_actual <- mean(actual[bin_samples])
      ece <- ece + abs(bin_prob - bin_actual) * sum(bin_samples) / length(actual)
    }
  }
  ece
}
#  calculate_mce 函数
calculate_mce <- function(probabilities, actual, n_bins = 10) {
  bin_indices <- cut(probabilities, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE)
  mce <- 0
  for (i in levels(bin_indices)) {
    bin_samples <- bin_indices == i
    if (sum(bin_samples) > 0) {  # 确保分箱中有样本
      bin_prob <- mean(probabilities[bin_samples])
      bin_actual <- mean(actual[bin_samples])
      mce <- max(mce, abs(bin_prob - bin_actual))
    }
  }
  mce
}

# 计算每个模型的指标
metrics_summary <- calibration_data %>%
  group_by(Model) %>%
  summarise(
    Brier_Score = calculate_brier_score(Probability, Actual),
    Log_Loss = calculate_log_loss(Probability, Actual),
    Calibration_Slope = calculate_calibration_slope(Probability, Actual),
    Calibration_Intercept = calculate_calibration_intercept(Probability, Actual),
    #HL_Test_P_Value = calculate_hl_test(Probability, Actual),
    ECE = calculate_ece(Probability, Actual),
    MCE = calculate_mce(Probability, Actual)
  )

# 打印指标汇总表
print(metrics_summary)


# 保存结果到CSV文件
write.csv(metrics_summary, file = "calibration_metrics_summary.csv", row.names = FALSE)


#-----4.DCA曲线绘制------
library(rmda)

dca_data <- data.frame(Result = as.factor(results$diagnosis), 
                       results$logistic, 
                       results$rf,
                       results$svm,
                       results$xgb,
                       results$lgbm,
                       results$ada,
                       results$meld_probability)

p <- dca(Result ~ results.rf + results.meld_probability, 
    data = dca_data, 
    thresholds = seq(0, 0.5, by = 0.01), 
    label = list(results.xgb = "Random Forest Model",
                 results.meld_probability= 'MELD score')) %>%
  plot(smooth = TRUE)+scale_colour_jama()+theme_cowplot()

p2 <- dca(Result ~ results.xgb + results.logistic+
            results.rf+results.svm+results.lgbm+results.ada+
            results.meld_probability, 
         data = dca_data, 
         thresholds = seq(0, 0.5, by = 0.01), 
         label = list(results.xgb = "XGboost Model",
                      results.BISAP= 'BISAP score',
                      results.logistic= "Logistic Regression",
                      results.rf="Random Forest",
                      results.svm="SVM",
                      results.lgbm="LightGBM",
                      results.ada="Adaboost",
                      results.meld_probability='MELD')) %>%
  plot(smooth = TRUE)+theme_cowplot(12)+scale_colour_igv()

#### CIC曲线
dca_data$Result <- as.numeric(dca_data$Result)-1
dca.result_rf <- decision_curve(Result ~ results.rf, 
                                 data = dca_data, 
                                 family = "binomial",
                                 thresholds = seq(0, 0.2, by = .01),
                                 bootstraps = 10)

plot_clinical_impact(dca.result_rf, col = "#7D5CC6FF",
                     population.size = 178,    #样本量1000
                     cost.benefit.axis = T,     #显示损失-收益坐标轴
                     n.cost.benefits= 8,
                     xlim=c(0,0.2),
                     confidence.intervals= T)



#-----5.shap可视化-----

#加载模型
library(kernelshap)
library(shapviz)
library(ggplot2)
testdata_features <- data_test %>% select(-diagnosis)
set.seed(123)
bg_X <- data_train %>% 
  select(-diagnosis) %>% 
  slice_sample(n = min(100, nrow(data_train)))

rf_underlying_model <- pull_workflow_fit(final_rf_fit)$fit

explain_kernel_rf <- kernelshap(
  object = rf_underlying_model,
  X = testdata_features,  # 保持数据框格式
  bg_X = bg_X,
  exact = FALSE  # 提高性能
)

shap_value_rf <- shapviz(explain_kernel_rf,
                         testdata_features, which_class = 1,
                          interactions=TRUE) 
sv_interaction(shap_value_rf)
sv_importance(shap_value_rf)


library(corrplot)
sv_dependence(shap_value_rf)

p1 <- sv_waterfall(shap_value_rf,row_id=1L)
p2 <- sv_importance(shap_value_rf,kind="beeswarm")
p3 <- sv_dependence(shap_value_rf,colnames(shap_value_rf),color_var=NULL,
                    ih_scale =T)+scale_color_bmj()

heatmap_shap <- shap_value_rf$X
corr <- round(cor(heatmap_shap), 1)
corrplot(corr, method = "circle")
res <- cor.mtest(heatmap_shap, conf.level = .95)
p <- res$p
# 假设你的原始变量名存储在colnames(corr)中
# 预处理变量名，在适当位置添加换行符

# 使用修改后的标签绘制热图
corrplot(corr, method = "pie", type = "upper", 
         p.mat = p, sig.level = c(.001, .01, .05), diag = FALSE,  # 隐藏对角线 
         insig = "label_sig", pch.cex = 1.2, 
         pch.col = 'black',
         tl.srt = 45,  # 设置标签旋转角度为45度
         tl.cex = 0.8,  # 调整标签字体大小以适应换行
         tl.col = "black")  # 行名也使用相同的修改后标签


