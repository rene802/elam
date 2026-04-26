def train_test_model(self, train_test_data, return_prediction=False, output_margin=False, average='macro', multi_class_opt='ovr', **kwargs):
    

#========================================================================================================================================================================================
# this function performs the training and the testing of the model. the performance of the model is estimated using the ROC AUC metric for classification and the MSE for the regression.
# parameters : train-test_data contains respectively x_train, y_train, x_test, y_test
# return_prediction : bool
# if True, the model predictions on the test set are returned
# output_margin: bool
# whether to output the raw untransformed margin value. If False, model probabilities are returned, not used it when task type is the regression
# average: string, in case of multi-classification we can use average of ROC AUC scores choosing between macros and weighted.
# multi_class_opt : string, is an option to compute ROC AUC score used only in case of multi-classification
# **kwargs: dict, extra kwargs passed on the model fit method
# returns : out can be array or None if return_prediction==True, model prediction on the test set are returned
#========================================================================================================================================================================================

# get number of classes
    n_classes = len(np.unique(y_train))
    self._n_classes = n_classes
    print('==============================')
    print(f"Training {self.model_string} model for {self._task_type}")
    if self._task_type == 'classification':
        print('Number of detected classes:', n_classes)

    # final training with the optimized hyperparams
    print('Training the model: ...')
    self.fit(x_train, y_train, **kwargs)
    print('Training the model: Done!')
    print('Testing the model: ...')
    y_pred_train = self.predict(x_train, output_margin=output_margin)
    y_pred_test = self.predict(x_test, output_margin=output_margin)
    if self._task_type == 'classification':
        roc_score_test = roc_auc_score(
            y_test, y_pred_test, average=average, multi_class=multi_class_opt)
        roc_score_train = roc_auc_score(
            y_train, y_pred_train, average=average, multi_class=multi_class_opt)
        print(f'ROC_AUC_score (test): {roc_score_test:.6f}')
        print(f'ROC_AUC_score (train): {roc_score_train:.6f}')
    else:
        mse_score = mean_squared_error(y_test, y_pred_test)
        print(f'Mean squared error: {mse_score:.6f}')
    print('Testing the model: Done!')
    print('==============================')
    if return_prediction:
        return y_pred_train, y_pred_test
    return None
