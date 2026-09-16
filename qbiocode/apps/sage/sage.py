import os, re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import optuna
import dill as pickle


#####


class QuantumSage():

    '''
    Sage class that will run an ML model over the input data frame which would be some set of defined data characeristics
    and performance metrics associated to the dataset the method use.  Right now it is focused on learning from just the data
    characteristics but it can eventual also include the model parameters as part of the input
    '''

    def __init__(self, data_input):
        '''
        This function initializes the Sage with the input data frame that contains the data characteristics and performance metrics
        '''

        self._columns_data_features = [ '# Features', '# Samples',
                                        'Feature_Samples_ratio', 'Intrinsic_Dimension', 'Condition number',
                                        'Fisher Discriminant Ratio', 'Total Correlations', 'Mutual information',
                                        '# Non-zero entries', '# Low variance features', 'Variation', 'std_var',
                                        'Coefficient of Variation %', 'std_co_of_v', 'Skewness', 'std_skew',
                                        'Kurtosis', 'std_kurt', 'Mean Log Kernel Density',
                                        'Isomap Reconstruction Error', 'Fractal dimension', 'Entropy',
                                        'std_entropy']
        self._columns_metrics = ['accuracy', 'f1_score', 'auc']
        self._columns_metadata = ['Dataset', 'embeddings','datatype', 'model_embed_datatype', 'iteration', 'model', 'BestParams_GridSearch', 'Model_Parameters']

        # Grid-search and serialized model parameters are optional QProfiler
        # outputs.  Keep the QSage input schema stable when either feature was
        # disabled in the producing QProfiler run.
        data_input = data_input.copy()
        for column in ('BestParams_GridSearch', 'Model_Parameters'):
            if column not in data_input.columns:
                data_input[column] = None

        self._input_data_features_only = data_input[self._columns_data_features]
        self._input_data_metrics = data_input[self._columns_metrics]
        self._input_data_metadata = data_input[self._columns_metadata]

        self._available_models = list(set(self._input_data_metadata['model']))
        if 'none' in self._available_models:
            self._available_models.remove('none')
        self._available_models.sort()
        self._available_embeddings = list(set(self._input_data_metadata['embeddings']))
        self._available_embeddings.sort()
        self._available_metrics = self._columns_metrics
        self._available_metrics.sort()

        self._results_subsages = {}

        self.set_seed()

    # TODO: trained sage should predict over every metric so that the user can decide what they want predicted
    def predict(self, input_data, metric = 'f1_score'):
        '''
        This function is used to make the prediction for a given metric on each of the models.
        The input data should be a DataFrame with the same features as the training data.
        The metric should be one of the available metrics (f1_score, auc, accuracy).
        It returns a DataFrame with the predictions for each model and the R2 score of the prediction.
        The predictions are sorted by the product of the metric and the R2 score, so that the best performing model is at the top.
        If the metric is not one of the available metrics, it will return None.
        If the input data does not have the same features as the training data, it will raise an error.

        Args:
            input_data (pd.DataFrame): DataFrame with the same features as the training data.
            metric (str): The metric to predict. Should be one of the available metrics (f1_score, auc, accuracy).

        Returns:
            predictions_df (pd.DataFrame): DataFrame with the predictions for each model and the R2 score of the prediction.
        '''

        predictions = []
        for model in self._available_models:
            pred = self._results_subsages[metric][model]['fit_model'].predict(input_data)[0]
            r2 = self._results_subsages[metric][model]['r2']
            predictions.append([model, pred, r2])
        predictions_df = pd.DataFrame( predictions, columns = ['model',metric,'r2'] )
        predictions_df[metric+'*r2'] = predictions_df[metric] * predictions_df['r2']
        predictions_df = predictions_df.sort_values(metric+'*r2', ascending=False)
        return predictions_df


    def train_sub_sages(self, test_size=0.2, sage_type='random_forest', n_iter=None, cv=5):
        """
        Train sub-sage predictors for each ML model and performance metric.
        
        This function trains regression models (Sage) that learn to predict
        model performance based on data complexity features. A separate sub-sage
        is trained for each combination of ML model and performance metric.
        
        Parameters
        ----------
        test_size : float, optional
            Proportion of data to use for testing (0.0 to 1.0). Default is 0.2.
        sage_type : str, optional
            Type of regressor to use as Sage. Must be one of:
            
            - 'random_forest': Random Forest with hyperparameter tuning (default)
            - 'mlp': Multi-Layer Perceptron with grid search
            - 'xgboost_optuna': XGBoost with Optuna optimization (state-of-the-art)
            
            Only ONE sage type can be selected per training run.
            
        n_iter : int, optional
            For Random Forest: number of hyperparameter search iterations (default: 50).
            For MLP: maximum number of training epochs (default: 1000).
            For XGBoost-Optuna: number of Optuna trials (default: 100).
            If None, uses the default for the selected sage_type.
        cv : int, optional
            Number of cross-validation folds for hyperparameter evaluation.
            Default is 5. Used by all sage types.
        
        Returns
        -------
        None
            Results are stored in the internal ``_results_subsages`` dictionary with structure:
            
            .. code-block:: python
            
                {
                    'metric1': {
                        'model1': {
                            'fit_model': <trained model>,
                            'preds': <predictions on test set>,
                            'y_test': <true values>,
                            'params': <model parameters>,
                            'mae': <mean absolute error>,
                            'mse': <mean squared error>,
                            'rmse': <root mean squared error>,
                            'r2': <R² score>
                        },
                        ...
                    },
                    ...
                }
        
        Raises
        ------
        ValueError
            If sage_type is not one of the valid types.
        ImportError
            If sage_type is 'xgboost_optuna' but XGBoost or Optuna is not installed.
        
        Notes
        -----
        The function iterates over all available metrics and models, training a
        separate predictor for each combination. Progress is printed during training.
        
        Only one sage type can be used per training run. If you want to compare
        different sage types, you need to train them separately and compare results.
        
        **Recommended Sage Type:**
        
        For best performance on continuous value prediction, use 'xgboost_optuna',
        which combines the power of gradient boosting with advanced Bayesian
        hyperparameter optimization.
        
        Examples
        --------
        Train with Random Forest (default):
        
        >>> sage.train_sub_sages(test_size=0.2, sage_type='random_forest')
        
        Train with MLP:
        
        >>> sage.train_sub_sages(test_size=0.2, sage_type='mlp')
        
        Train with XGBoost-Optuna (state-of-the-art):
        
        >>> sage.train_sub_sages(test_size=0.2, sage_type='xgboost_optuna', n_iter=200)
        
        Train with custom hyperparameter search:
        
        >>> sage.train_sub_sages(sage_type='random_forest', n_iter=100, cv=10)
        
        See Also
        --------
        _sage_random_forest : Random Forest Sage implementation
        _sage_mlp : MLP Sage implementation
        _sage_xgboost_optuna : XGBoost with Optuna Sage implementation (state-of-the-art)
        predict : Make predictions using trained Sages
        """
        # Validate sage_type parameter
        valid_sage_types = ['random_forest', 'mlp', 'xgboost_optuna']
        if sage_type not in valid_sage_types:
            raise ValueError(
                f"Invalid sage_type '{sage_type}'. Must be one of {valid_sage_types}. "
                f"Only one sage type can be selected per training run."
            )
        
        for metric in self._available_metrics:
            print(f"Working on {metric}")

            self._results_subsages[metric] = {}
            for model in self._available_models:
                model_indices = self._input_data_metadata[ self._input_data_metadata['model'] == model ].index
                X = self._input_data_features_only.loc[ model_indices ]
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                y = self._input_data_metrics.loc[model_indices][metric].fillna(0).to_numpy()
                
                print(f"Working on {model}")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = self._seed)

                # Calculate SLGH (Scaled Latent Geometric Hardness)
                X_train = calculate_SLGH(X_train)
                X_test = calculate_SLGH(X_test)

                if sage_type == 'random_forest':
                    # Use default n_iter=50 for Random Forest if not specified
                    rf_n_iter = n_iter if n_iter is not None else 50
                    self._results_subsages[metric][model] = self._sage_random_forest(
                        X_train, X_test, y_train, y_test, n_iter=rf_n_iter, cv=cv
                    )
                elif sage_type == 'mlp':
                    # Use default n_iter=1000 for MLP if not specified
                    mlp_n_iter = n_iter if n_iter is not None else 1000
                    self._results_subsages[metric][model] = self._sage_mlp(
                        X_train, X_test, y_train, y_test, n_iter=mlp_n_iter, cv=cv
                    )
                elif sage_type == 'xgboost_optuna':
                    # Use default n_iter=100 for XGBoost-Optuna if not specified
                    xgb_n_iter = n_iter if n_iter is not None else 100
                    self._results_subsages[metric][model] = self._sage_xgboost_optuna(
                        X_train, X_test, y_train, y_test, n_iter=xgb_n_iter, cv=cv
                    )

    def _sage_mlp(self, X_train, X_test, y_train, y_test, n_iter=1000, cv=5):
        """
        Train a Multi-Layer Perceptron (MLP) regressor as a Sage predictor with hyperparameter tuning.
        
        This function performs a grid search over MLP hyperparameters to find the best
        configuration, then makes predictions on the test set. The search uses cross-validation
        to evaluate different parameter combinations. Early stopping is used to prevent
        overfitting during training.
        
        The function is called internally by :meth:`train_sub_sages` and is not meant
        to be called directly by users. It is designed to work with preprocessed data
        that has been split into training and test sets.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (data complexity metrics).
        X_test : pd.DataFrame
            Test features (data complexity metrics).
        y_train : pd.Series
            Training labels (model performance values).
        y_test : pd.Series
            Test labels (model performance values).
        n_iter : int, optional
            Maximum number of training iterations (epochs) for the MLP. Default is 1000.
            Training will run for at most this many iterations, but may stop earlier
            if early stopping criteria are met (no improvement for 10 consecutive epochs).
            Higher values allow more training time but take longer to run.
            Lower values speed up training but may underfit if set too low.
        cv : int, optional
            Number of cross-validation folds for hyperparameter evaluation. Default is 5.
            Each parameter combination is evaluated using k-fold cross-validation to
            ensure robust performance estimates.
        
        Returns
        -------
        dict
            Dictionary containing:
            
            - 'fit_model' : GridSearchCV
                Trained MLP model with best parameters from grid search
            - 'preds' : np.ndarray
                Predictions on test set
            - 'y_test' : pd.Series
                True test labels
            - 'params' : dict
                Best hyperparameters found by grid search
            - 'mae' : float
                Mean Absolute Error on test set
            - 'mse' : float
                Mean Squared Error on test set
            - 'rmse' : float
                Root Mean Squared Error on test set
            - 'r2' : float
                R² score on test set
        
        Notes
        -----
        The hyperparameter search space includes:
        
        - hidden_layer_sizes: [(32, 10), (64, 32), (100,), (50, 25)]
        - activation: ['relu', 'tanh']
        - solver: ['adam', 'lbfgs']
        - alpha: [0.0001, 0.001, 0.01] (L2 regularization)
        - learning_rate: ['constant', 'adaptive']
        
        The MLP uses:
        
        - Early stopping with patience of 10 epochs (``n_iter_no_change=10``)
        - Adaptive learning rate starting at 0.001
        - Maximum iterations controlled by ``n_iter`` parameter
        - Automatic batch size
        - 10% validation split for early stopping
        
        **Interpretation of n_iter for MLP:**
        
        Unlike Random Forest where ``n_iter`` controls the number of hyperparameter
        search iterations, for MLP it controls the **maximum number of training epochs**.
        The actual training may stop earlier due to early stopping (if no improvement
        for 10 consecutive epochs). This allows you to control the training time while
        still benefiting from early stopping to prevent overfitting.
        
        See Also
        --------
        _sage_random_forest : Alternative Random Forest sub-sage
        train_sub_sages : Main training function that calls this method
        """
        from sklearn.model_selection import GridSearchCV

        # Define hyperparameter grid for MLP
        param_grid = {
            'hidden_layer_sizes': [(32, 10), (64, 32), (100,), (50, 25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }

        # Initialize MLP with early stopping
        mlp = MLPRegressor(
            batch_size='auto',
            learning_rate_init=0.001,
            max_iter=n_iter,  # Use n_iter for maximum training iterations
            random_state=self._seed,
            n_iter_no_change=10,  # Fixed early stopping patience
            early_stopping=True,
            validation_fraction=0.1
        )

        # Initialize GridSearchCV with configurable cv parameter
        mlp_grid = GridSearchCV(
            estimator=mlp,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
            scoring='r2'
        )
        
        # Train
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        mlp_grid.fit(X_train, y_train)
        preds = mlp_grid.predict(X_test)
        params = mlp_grid.best_params_

        # Evaluate on held out
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        result = {
            'fit_model': mlp_grid,
            'preds': preds,
            'y_test': y_test,
            'params': params,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

        return result

    def _sage_xgboost_optuna(self, X_train, X_test, y_train, y_test, n_iter=100, cv=5):
        """
        Train an XGBoost regressor as a Sage predictor with advanced hyperparameter optimization using Optuna.
        
        This function uses Optuna, a state-of-the-art hyperparameter optimization framework, to find
        the best XGBoost configuration through Bayesian optimization. XGBoost is a gradient boosting
        algorithm known for its superior performance on tabular data and continuous value prediction.
        
        The function is called internally by :meth:`train_sub_sages` and is not meant to be called
        directly by users. It is designed to work with preprocessed data that has been split into
        training and test sets.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (data complexity metrics).
        X_test : pd.DataFrame
            Test features (data complexity metrics).
        y_train : pd.Series
            Training labels (model performance values).
        y_test : pd.Series
            Test labels (model performance values).
        n_iter : int, optional
            Number of Optuna trials for hyperparameter optimization. Default is 100.
            Higher values explore more parameter combinations but take longer.
            Each trial uses cross-validation to evaluate the parameter set.
        cv : int, optional
            Number of cross-validation folds for hyperparameter evaluation. Default is 5.
            Each parameter combination is evaluated using k-fold cross-validation to
            ensure robust performance estimates.
        
        Returns
        -------
        dict
            Dictionary containing:
            
            - 'fit_model' : xgb.XGBRegressor
                Trained XGBoost model with best parameters from Optuna optimization
            - 'preds' : np.ndarray
                Predictions on test set
            - 'y_test' : pd.Series
                True test labels
            - 'params' : dict
                Best hyperparameters found by Optuna
            - 'mae' : float
                Mean Absolute Error on test set
            - 'mse' : float
                Mean Squared Error on test set
            - 'rmse' : float
                Root Mean Squared Error on test set
            - 'r2' : float
                R² score on test set
            - 'study' : optuna.Study
                Complete Optuna study object with optimization history
        
        **Hyperparameter Search Space:**
        
        The optimization explores:
        
        - n_estimators: [50, 500] - number of boosting rounds
        - max_depth: [3, 10] - maximum tree depth
        - learning_rate: [0.001, 0.3] - step size shrinkage (log scale)
        - subsample: [0.6, 1.0] - fraction of samples per tree
        - colsample_bytree: [0.6, 1.0] - fraction of features per tree
        - min_child_weight: [1, 10] - minimum sum of instance weight in child
        - gamma: [0, 5] - minimum loss reduction for split
        - reg_alpha: [0, 1] - L1 regularization (log scale)
        - reg_lambda: [0, 10] - L2 regularization (log scale)
        
        **Installation Requirements:**
        
        To use this function, install the required packages:
        
        .. code-block:: bash
        
            pip install xgboost optuna
        
        **Performance Tips:**
        
        - Increase n_iter (e.g., 200-500) for better optimization on complex problems
        - Use higher cv (e.g., 10) for more robust evaluation with sufficient data
        - Monitor the Optuna study object to understand optimization progress
        - Consider using GPU acceleration for XGBoost on large datasets
        
        Examples
        --------
        Train with XGBoost-Optuna (via train_sub_sages):
        
        >>> sage.train_sub_sages(sage_type='xgboost_optuna', n_iter=200, cv=10)
        
        See Also
        --------
        _sage_mlp : MLP sub-sage with grid search
        _sage_random_forest : Random Forest sub-sage with randomized search
        train_sub_sages : Main training function that calls this method
        
        References
        ----------
        .. [1] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
               In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
               Discovery and Data Mining (pp. 785-794).
        .. [2] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna:
               A next-generation hyperparameter optimization framework. In Proceedings of
               the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data
               Mining (pp. 2623-2631).
        """

        from sklearn.model_selection import cross_val_score
        
        # Preprocess data
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Define objective function for Optuna
        def objective(trial):
            """Optuna objective function for hyperparameter optimization."""
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self._seed,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            # Create model with suggested parameters
            model = xgb.XGBRegressor(**params)
            
            # Evaluate with cross-validation
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring='r2',
                n_jobs=-1
            )
            
            return scores.mean()
        
        # Create Optuna study with pruning
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self._seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        # Suppress Optuna's verbose output
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Optimize hyperparameters
        study.optimize(objective, n_trials=n_iter, show_progress_bar=False)
        
        # Get best parameters
        best_params = study.best_params
        best_params['random_state'] = self._seed
        best_params['n_jobs'] = -1
        best_params['verbosity'] = 0
        
        # Train final model with best parameters
        best_model = xgb.XGBRegressor(**best_params)
        best_model.fit(X_train, y_train)
        
        # Make predictions
        preds = best_model.predict(X_test)
        
        # Evaluate on test set
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        result = {
            'fit_model': best_model,
            'preds': preds,
            'y_test': y_test,
            'params': best_params,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'study': study  # Include study for analysis
        }
        
        return result


    def _sage_random_forest(self, X_train, X_test, y_train, y_test, n_iter=50, cv=5):
        """
        Train a Random Forest regressor as a sub-sage predictor with hyperparameter tuning.
        
        This function performs a randomized search over the hyperparameter space to find
        the best Random Forest configuration, then makes predictions on the test set.
        The search uses cross-validation to evaluate different parameter combinations.
        
        The function is called internally by :meth:`train_sub_sages` and is not meant
        to be called directly by users. It is designed to work with preprocessed data
        that has been split into training and test sets.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (data complexity metrics).
        X_test : pd.DataFrame
            Test features (data complexity metrics).
        y_train : pd.Series
            Training labels (model performance values).
        y_test : pd.Series
            Test labels (model performance values).
        n_iter : int, optional
            Number of iterations for randomized hyperparameter search. Default is 50.
            Higher values explore more parameter combinations but take longer.
        cv : int, optional
            Number of cross-validation folds for hyperparameter evaluation. Default is 5.
            Each parameter combination is evaluated using k-fold cross-validation.
        
        Returns
        -------
        dict
            Dictionary containing:
            
            - 'fit_model' : RandomizedSearchCV
                Trained Random Forest model with best parameters
            - 'preds' : np.ndarray
                Predictions on test set
            - 'y_test' : pd.Series
                True test labels
            - 'params' : dict
                Best hyperparameters found by randomized search
            - 'mae' : float
                Mean Absolute Error on test set
            - 'mse' : float
                Mean Squared Error on test set
            - 'rmse' : float
                Root Mean Squared Error on test set
            - 'r2' : float
                R² score on test set
        
        Notes
        -----
        The hyperparameter search space includes:
        
        - n_estimators: [100, 200, ..., 900] trees
        - max_depth: [5, 6, ..., 19] maximum tree depth
        - min_samples_split: [2, 3, ..., 9] minimum samples to split
        - min_samples_leaf: [1, 2, 3, 4] minimum samples per leaf
        - bootstrap: [True, False] whether to use bootstrap sampling
        
        The function handles infinite values and NaN by replacing them with 0.
        
        See Also
        --------
        _sage_mlp : Alternative MLP sub-sage
        train_sub_sages : Main training function that calls this method
        """

        param_distributions = {
            'n_estimators': np.arange(100, 1000, 100),
            'max_depth': np.arange(5, 20),
            'min_samples_split': np.arange(2, 10),
            'min_samples_leaf': np.arange(1, 5),
            'bootstrap': [True, False]
        }

        # Initialize the Random Forest Regressor
        rf = RandomForestRegressor(random_state=self._seed)

        # Initialize RandomizedSearchCV with configurable cv parameter
        rf_random = RandomizedSearchCV(estimator=rf,
                                    param_distributions=param_distributions,
                                    n_iter=n_iter,
                                    cv=cv,
                                    random_state=self._seed,
                                    n_jobs=-1)
        
        # Train
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        rf_random.fit(X_train, y_train)
        preds = rf_random.predict(X_test)
        params = rf_random.best_params_

        # Evaluate on held out
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        result = {
            'fit_model' : rf_random,
            'preds' : preds,
            'y_test' : y_test,
            'params' : params,
            'mae' : mae,
            'mse' : mse,
            'rmse' : rmse,
            'r2' : r2
        }

        return result

    def plot_results(self, figsize = (6,4), saveFile='' ):

        ''' This function plots the results of the sub-sages trained on the input data.
        It will create a bar plot for each metric showing the performance of each model, and a scatter plot of the predictions vs. true values.
        The bar plot will show the mean absolute error (mae), mean squared error (mse), root mean squared error (rmse), and R2 score (r2) for each model.
        The scatter plot will show the predictions vs. true values for each model.
        If saveFile is provided, the plots will be saved to that file. Otherwise, the plots will be shown.
        It is designed to be used after the train_sub_sages function has been called, and the sub-sages have been trained.
        Args:
            saveFile (str): The file name to save the plots. If empty, the plots will be shown. Default is ''.
        Returns:
            None: The function does not return anything, it just plots the results of the sub-sages.
        
        '''

        results = []
        preds = pd.DataFrame()
        for metric in self._available_metrics:
            for model in self._available_models:
                scores = pd.Series( self._results_subsages[metric][model].values(), index=self._results_subsages[metric][model].keys())
                results.append( [model, metric]+list(scores[['mae','mse','rmse','r2']]) )
                p = self._results_subsages[metric][model]['preds']
                y = self._results_subsages[metric][model]['y_test']
                preds = pd.concat( [preds,
                                    pd.DataFrame( [[model]*len(p),[metric]*len(p),p,y], index = ['model', 'metric', 'pred', 'y_test'] ).transpose() ] )
        
        # Create results DataFrame after collecting all results
        if not results:
            print("Warning: No results to plot. Train QSages first.")
            return
        
        results_df = pd.DataFrame(results, columns=['model','metric','mae','mse','rmse','r2'])
        results_df = results_df.melt(id_vars=['model', 'metric'])
        
        for metric in self._available_metrics:
            plt.figure(figsize=figsize)
            # Filter data for current metric
            metric_data = results_df[results_df['metric']==metric]
            sns.barplot(data = metric_data, x = 'variable', y = 'value', hue = 'model', hue_order=self._available_models)  # type: ignore[arg-type]
            plt.title( "Predictive performance for each model for " + metric)
            plt.xlabel( "Metric")
            plt.ylabel( "Value" )
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            if saveFile != '':
                plt.savefig( re.sub( '.pdf', '', saveFile) + '_' + metric + '_barplot.pdf', bbox_inches='tight' )
            plt.show()
            plt.close()

            # Filter predictions for current metric
            toPlot = preds[ preds['metric'] == metric ]
            plt.figure(figsize=figsize)
            plt.title( "Predictive performance for each model for " + metric)
            sns.scatterplot( data = toPlot, x = 'y_test', y = 'pred', hue = 'model' )  # type: ignore[arg-type]
            plt.xlabel( "Actual")
            plt.ylabel( "Predicted" )
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            if saveFile != '':
                plt.savefig( re.sub( '.pdf', '', saveFile) + '_' + metric + '_scatterplot.pdf', bbox_inches='tight' )
            plt.show()
            plt.close()


    def set_seed(self, seed=42):
        self._seed = seed


def calculate_SLGH(df, train_pct = 0.7):
    id_col = 'Intrinsic_Dimension'
    fdr_col = 'Fisher Discriminant Ratio'
    num_samples = '# Samples'
    n_train = np.ceil(df[num_samples] * train_pct)
    eps = 1e-8

    df = pd.DataFrame(df)
    df['SLGH'] = (-np.log(df[id_col] + eps) - np.log(1.0 + df[fdr_col] * n_train))
    return(df)


def main():
    """
    Command-line interface for QSage (Quantum Sage).
    
    This CLI allows users to train QSage models from the command line using CSV data files.
    QSage learns relationships between dataset complexity measures and model performance,
    enabling prediction of model performance on new datasets.
    
    Usage:
        qsage --input data.csv --output results/ [options]
    
    The input CSV should contain:
        - Dataset complexity features (# Features, # Samples, Intrinsic_Dimension, etc.)
        - Performance metrics (accuracy, f1_score, auc)
        - Metadata (Dataset, embeddings, model, etc.)
    
    QProfiler Integration:
        QSage is designed to work directly with QProfiler output. Simply use the
        compiled_results.csv file generated by QProfiler as input:
        
        # Step 1: Run QProfiler
        qprofiler --config-name=config.yaml
        
        # Step 2: Train QSage with QProfiler output
        qsage --input compiled_results.csv --output sage_results/
    
    Examples:
        # Basic usage with QProfiler output
        qsage --input compiled_results.csv --output sage_results/
        
        # With custom cross-validation and hyperparameter search
        qsage --input compiled_results.csv --output results/ --cv 10 --n-iter 100
        
        # Train Random Forest sub-sages
        qsage --input data.csv --output results/ --model-type rf --seed 42
        
        # Train MLP sub-sages
        qsage --input data.csv --output results/ --model-type mlp --n-iter 2000
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description='QSage: Quantum-inspired model selection oracle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Train QSage on profiler results
        qsage --input compiled_results.csv --output sage_results/
        
        # Train with specific metric and iterations
        qsage --input data.csv --output results/ --metric accuracy --n-iter 200
        
        # Train with custom seed
        qsage --input data.csv --output results/ --seed 123
        
        For more information, see: https://ibm.github.io/QBioCode/apps/sage.html
                """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input CSV file containing dataset features and model performance metrics'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for results and plots'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--model-type',
        default='random_forest',
        choices=['rf', 'mlp', 'random_forest', 'xgboost', 'xgboost_optuna'],
        help='Type of sub-sage model to train: rf/random_forest (Random Forest), mlp (MLP), '
             'or xgboost/xgboost_optuna (XGBoost with Optuna - state-of-the-art). '
             'Default: random_forest. Only one type can be trained per run.'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing (default: 0.2)'
    )
    
    parser.add_argument(
        '--n-iter',
        type=int,
        default=None,
        help='For Random Forest: number of hyperparameter search iterations (default: 50). '
             'For MLP: maximum training epochs (default: 1000). '
             'For XGBoost-Optuna: number of Optuna trials (default: 100)'
    )
    
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds for hyperparameter evaluation (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("="*80)
    print("QSage: Quantum Model Selection Oracle")
    print("="*80)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Test size: {args.test_size}")
    print(f"Random seed: {args.seed}")
    print(f"Model type: {args.model_type}")
    print(f"Cross-validation folds: {args.cv}")
    if args.n_iter is not None:
        if args.model_type in ['rf', 'random_forest']:
            print(f"Hyperparameter search iterations: {args.n_iter}")
        elif args.model_type == 'mlp':
            print(f"Maximum training epochs: {args.n_iter}")
        elif args.model_type in ['xgboost', 'xgboost_optuna']:
            print(f"Optuna optimization trials: {args.n_iter}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    try:
        data = pd.read_csv(args.input)
        # data['Model_Parameters'] = [ 'None' if str(x) == 'nan' else x for x in data['Model_Parameters'] ]
        data['embeddings'] = [ 'None' if str(x) == 'nan' else x for x in data['embeddings'] ]
        print(f"Loaded {len(data)} rows with {len(data.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize QSage
    print("\nInitializing QSage...")
    try:
        sage = QuantumSage(data)
        sage.set_seed(args.seed)
        print(f"Available models: {sage._available_models}")
        print(f"Available embeddings: {sage._available_embeddings}")
        print(f"Available metrics: {sage._available_metrics}")
    except Exception as e:
        print(f"Error initializing QSage: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Train sub-sages
    print(f"\nTraining sub-sages...")
    try:
        # Map CLI argument to sage_type
        sage_type_map = {
            'rf': 'random_forest',
            'random_forest': 'random_forest',
            'mlp': 'mlp',
            'xgboost': 'xgboost_optuna',
            'xgboost_optuna': 'xgboost_optuna'
        }
        sage_type = sage_type_map[args.model_type]
        
        print(f"  Training {sage_type} sub-sages...")
        sage.train_sub_sages(
            test_size=args.test_size,
            sage_type=sage_type,
            n_iter=args.n_iter,
            cv=args.cv
        )
        
        print("Training complete!")
    except Exception as e:
        print(f"Error training sub-sages: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Save Sage model
    pickle.dump(sage, open(os.path.join(args.output, 'trained_sage.pkl'), 'wb'))

    # Generate and save plots
    print(f"\nGenerating plots...")
    try:
        output_file = os.path.join(args.output, 'sage_results.pdf')
        sage.plot_results(saveFile=output_file)
        print(f"Plots saved to: {output_file}")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}", file=sys.stderr)
    
    # Save results summary
    print("\nSaving results summary...")
    try:
        results_summary = []
        for metric in sage._available_metrics:
            for model in sage._available_models:
                if metric in sage._results_subsages and model in sage._results_subsages[metric]:
                    result = sage._results_subsages[metric][model]
                    results_summary.append({
                        'model': model,
                        'metric': metric,
                        'mae': result['mae'],
                        'mse': result['mse'],
                        'rmse': result['rmse'],
                        'r2': result['r2']
                    })
        
        if results_summary:
            results_df = pd.DataFrame(results_summary)
            results_file = os.path.join(args.output, 'sage_summary.csv')
            results_df.to_csv(results_file, index=False)
            print(f"Results summary saved to: {results_file}")
            print("\nResults Summary:")
            print(results_df.to_string(index=False))
        else:
            print("Warning: No results to save")
    except Exception as e:
        print(f"Warning: Could not save results summary: {e}", file=sys.stderr)
    
    print("\n" + "="*80)
    print("QSage training completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
