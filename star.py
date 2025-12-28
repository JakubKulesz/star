import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import scipy.stats
import statsmodels.api as sm
import time
from datetime import datetime
from numdifftools import Hessian

class STAR:
    """
    Smooth Transition Regression (STAR) model.

    This class implements a nonlinear regression model where coefficients smoothly
    change between regimes according to a logistic or exponential transition function.
    The regimes are governed by at least one threshold variable, which
    can be lagged:
    - dependent variables,
    - independent variables,
    - time trend,
    - external time series.

    Key functionalities include:
    - Nonlinear Least Squares (NLS) estimation of model parameters.
    - Flexible specification of transition variables with optional lags.
    - Prediction of out-of-sample values using estimated parameters.
    - Inspection of regime weights for model diagnostics.
    
    Parameters
    ----------
    threshold_variables : list of dict
        List specifying transition variables controlling regime changes.
        Each dictionary must include:
            - 'type': str, one of {'dependent', 'independent', 'time', 'external'} (always required)
            - 'name': str, name of the X column (required if type=='independent')
            - 'lag': int, lag applied to the transition variable (optional)
            - 'value': float, fixed threshold value that will overrite the estimation process of he parameter (optional)
            - 'data': pd.DataFrame (required if type=='external')
            - 'transition_function': str, 'logistic' or 'exponential' (always required)
    lags : list of int, optional
        List of lags of the dependent variable to be included in the regression.
    """

    def __init__(self, threshold_variables, lags=None):

        self.lags = lags
        self.threshold_variables = threshold_variables
        self.params = None
        self.param_names = None

    
    @staticmethod
    def transition_function(transition_function, transition_var, gamma, threshold_var):
        """
        Transition function claculation.

        Parameters
        ----------
        transition_function : str
            Type of transition function: 'logistic' or 'exponential'.
        transition_var : array-like
            Transition variable.
        gamma : float
            Slope (smoothness) parameter of the transition function.
        threshold_var : float
            Threshold value (center of the transition function).

        Returns
        -------
        ndarray
            Array of weights between 0 and 1.
        """

        if transition_function == 'logistic':
            return 1 / (1 + np.exp(-gamma * (transition_var - threshold_var)))
        if transition_function == 'exponential':
            return 1 - np.exp(-gamma * (transition_var - threshold_var)**2)

    
    def _generate_transition_variables(self, y, X):
        """
        Creates a DataFrame containing all transition variables as specified
        in `threshold_variables`, including appropriate lags.

        Parameters
        ----------
        y : pandas.Series or pandas.DataFrame
            Dependent variable.
        X : pandas.DataFrame
            Matrix of independent variables.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing transition variables.
        """

        transition_df = pd.DataFrame(index=y.index)

        # Numeration of each type of transition variable
        dep, ind, tim, ext = 1, 1, 1, 1

        # Loop for each regime change
        for var in self.threshold_variables:
            # Extract information about lag if it exists
            lag_value = var['lag'] if 'lag' in var else 0
            # Prepare each transition variable according to the input values
            if var['type'] == 'dependent':
                transition_df[f'dependent_{dep}'] = y.shift(lag_value)
                dep += 1
            elif var['type'] == 'independent':          
                independent_series = X[var['name']].shift(lag_value).rename(f'independent_{ind}')    
                independent_df = independent_series.to_frame()
                transition_df = transition_df.merge(independent_df, left_index=True, right_index=True, how='left')
                ind += 1
            elif var['type'] == 'time':
                transition_df[f'time_{tim}'] = y.div(y).cumsum()
                tim += 1
            elif var['type'] == 'external':
                if 'data' in var and isinstance(var['data'], (pd.Series, pd.DataFrame)):
                    external_series = var['data'].shift(lag_value)
                    if isinstance(external_series, pd.Series):
                        external_series = external_series.rename(f'external_{ext}')
                    else:
                        external_series.columns = [f'external_{ext}']
                    transition_df = transition_df.merge(external_series, how='left', right_index=True, left_index=True)
                    ext += 1

        return transition_df


    def _nls_calculation(self, params, y, X, transition_df, transition_function, fixed_params=None):
        """
        Nonlinear Least Squares estimation.

        Parameters
        ----------
        params : array-like
            Vector of model parameters.
        y : pd.Series or pd.DataFrame
            Dependent variable.
        X : pd.DataFrame
            Independent variables.
        transition_df : pd.DataFrame
            Matrix of transition variables.
        fixed_params : dict, optional
            Dictionary mapping index of threshold parameters to fixed values.

        Returns
        -------
        ndarray
            Vector of residuals.
        """

        # Get information needed to calculate a number of the parameters.
        num_regime_changes = transition_df.shape[1]
        num_independent = X.shape[1]

        # Integer
        integer = params[:1]
        # Parameters of independent variables
        beta = params[1:1 + ((num_regime_changes + 1) * num_independent)]
        # Parameters for slopes of all of the transition functions
        gamma_values = params[1 + ((num_regime_changes + 1) * num_independent):1 + ((num_regime_changes + 1) * num_independent) + num_regime_changes]
        # Parameters for moments of al of the regime changes
        threshold_values = params[1 + ((num_regime_changes + 1) * num_independent) + num_regime_changes:1 + ((num_regime_changes + 1) * num_independent) + num_regime_changes * 2]
        # Overwriting predetermined parameters
        if fixed_params:
            for idx, value in fixed_params.items():
                threshold_values[idx] = value

        # No regime part predictions
        linear_sum = integer + X @ beta[:num_independent]
        # Regime part predictions calculated in the loop for each regime
        nonlinear_sum = np.zeros_like(y)
        for r in range(num_regime_changes):
            trans_func = self.transition_function(
                transition_function[r],
                transition_df[:, r],
                gamma_values[r],
                threshold_values[r]
            )
            
            X_r = (X @ beta[((r + 1) * num_independent): ((r + 2) * num_independent)]) # Different betas for each regime
            nonlinear_sum += (trans_func * X_r)

        # Composition of the linear and non linear predictions
        predicted_values = linear_sum + nonlinear_sum

        return (y - predicted_values).flatten()


    def _loglike(self, params):
        res = self._nls_calculation(
            params,
            self.y,
            self.X,
            self.transition_variables,
            self.transition_functions,
            self.fixed_params
        )
        sigma2 = np.mean(self.resid**2)
        T = len(res)
        return (
            -0.5*T*np.log(2*np.pi)
            -0.5*T*np.log(sigma2)
            -0.5*np.sum(res**2)/sigma2)
    

    def fit(self, y, X):
        """
        Calbibrates model parameters to the provided dataset.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Dependent variable.
        X : pd.DataFrame
            Independent variables.

        Returns
        -------
        self : STAR
            Fitted model object with estimated parameters, covariance matrix,
            fitted values, residuals, R-squared, AIC, BIC, and results summary.
        """    
        
        start_time = time.time()

        # Copy input
        y = y.copy()
        X = X.copy()
        self.exog_names = X.columns.tolist()

        # Extend the dataset of the lagged variables
        if self.lags:
            for l in self.lags:
                X[f'dependent_L{l}'] = y.shift(l)

        # Create dataframe with all of the transformed transition variables combined
        transition_variables = self._generate_transition_variables(y, X)

        # Remove all the unnecessary observations (any row wiith NAs)
        _combined_df = y.merge(X, left_index=True, right_index=True, how='left')
        _combined_df = _combined_df.merge(transition_variables, left_index=True, right_index=True, how='left')
        _combined_df = _combined_df.dropna()
        y = _combined_df[y.columns]
        X = _combined_df[X.columns]
        transition_variables = _combined_df[transition_variables.columns]
        del _combined_df

        # Calculate initial parameters. Initial betas stay the same for every regime
        integer_init = np.mean(y)
        beta_init = sm.OLS(y, sm.add_constant(X)).fit().params[1:].to_list() * (transition_variables.shape[1] + 1)

        # Calculate initial parameters for each regime switching function
        time_count = transition_variables.columns.str.startswith('time_').sum()
        time_counter = 1
        gamma_init = []
        threshold_init = []

        for var in transition_variables.columns:
            # For dependent regime function inital gamma is set up as 1/std(x) and initial threshold value as mean(x) (if not provided)
            if str(var).startswith('dependent_'):
                gamma_init.append(1 / transition_variables[var].std())
                if 'value' in self.threshold_variables[int(var.split('_')[1]) - 1]:
                    threshold_init.append(self.threshold_variables[int(var.split('_')[1]) - 1]['value'])
                else:
                    threshold_init.append(transition_variables[var].mean())
            # For independent regime function inital gamma is set up as 1/std(x) and initial threshold value as mean(x) (if not provided)
            elif str(var).startswith('independent_'):
                gamma_init.append(1 / transition_variables[var].std())
                if 'value' in self.threshold_variables[int(var.split('_')[1]) - 1]:
                    threshold_init.append(self.threshold_variables[int(var.split('_')[1]) - 1]['value'])
                else:
                    threshold_init.append(transition_variables[var].mean())
            # For time regime function inital gamma is set up as 0.5 and initial threshold
            # values are evenly distributed for every time regime function to prevent multicollinearity (if not provided)       
            elif str(var).startswith('time_'):
                gamma_init.append(0.5)
                if 'value' in self.threshold_variables[int(var.split('_')[1]) - 1]:
                    threshold_init.append(self.threshold_variables[int(var.split('_')[1]) - 1]['value'])
                else:
                    threshold_init.append(time_counter / (time_count + 1) * transition_variables.shape[0])
                    time_counter += 1
            # For external regime function inital gamma is set up as 1/std(x) and initial threshold value as mean(x) (if not provided
            elif str(var).startswith('external_'):
                gamma_init.append(1 / transition_variables[var].std())
                if 'value' in self.threshold_variables[int(var.split('_')[1]) - 1]:
                    threshold_init.append(self.threshold_variables[int(var.split('_')[1]) - 1]['value'])
                else:
                    threshold_init.append(transition_variables[var].mean())
        self.transition_names = transition_variables.columns.tolist()

        # combined initial parmeters vector
        initial_params = [integer_init] + beta_init + gamma_init + threshold_init
        self.initial_params = initial_params

        self.fixed_params = {idx: var['value'] for idx, var in enumerate(self.threshold_variables) if 'value' in var}

        self.param_names = (
            ['const'] +
            [f'{c}__base' for c in X.columns] +
            [f'{c}__reg_{r + 1}' for r in range(transition_variables.shape[1]) for c in X.columns] +
            [f'gamma__reg_{r + 1}' for r in range(transition_variables.shape[1])] +
            [f'threshold__reg_{r + 1}' for r in range(transition_variables.shape[1])]
        )

        # convert all datasets to NumPy arrays to improve optimization efficiency
        self.y_index = y.index.tolist()
        self.y = y.to_numpy().flatten()
        self.X = X.to_numpy()
        self.transition_variables = transition_variables.to_numpy()
        self.transition_functions = [var.get('transition_function') for var in self.threshold_variables]

        # parameters estimation via the scipy.optimize package
        result = least_squares(
            self._nls_calculation,
            initial_params,
            args=(
                self.y,
                self.X,
                self.transition_variables,
                self.transition_functions,
                self.fixed_params),
            method='lm'
        )

        # saving the estimation results
        self.params = result.x

        # calculating all of the necassary ans udeful metrics
        self.n = self.X.shape[0]
        self.df_model = len(self.params)
        self.df_resid = self.n - self.df_model
        self.resid = result.fun
        self.fittedvalues = self.y - self.resid

        H = Hessian(self._loglike)
        H = H(self.params)
        self.cov_matrix = np.linalg.pinv(-H)    
        
        self.scale = np.var(self.resid, ddof=self.df_model)
        # Alt Hessian
        #J = result.jac
        #self.cov_matrix = self.scale * np.linalg.pinv(J.T @ J)
        
        self.bse = np.sqrt(np.diag(self.cov_matrix))

        eps = 1e-8
        self.tvalue = self.params / np.where(self.bse < eps, np.nan, self.bse)
        self.pvalue = 2 * (1 - scipy.stats.t.cdf(np.abs(self.tvalue), df=self.df_resid))

        self.results_df = pd.DataFrame({
            'Parameter': self.param_names,
            'Estimate': self.params,
            'Std Error': self.bse,
            't-value': self.tvalue,
            'p-value': self.pvalue
        })

        ss_res = max(np.sum(self.resid**2), 1e-8)
        ss_tot = ((self.y - self.y.mean())**2).sum(axis=0).sum()
        self.rsquared = 1 - (ss_res / ss_tot)
        self.aic = float(-2*self._loglike(self.params)/self.n + 2*self.df_model/self.n)
        self.bic = float(-2*self._loglike(self.params)/self.n + 2*self.df_model*np.log(self.n)/self.n)

        self.model_time = time.time()
        self.calc_time = self.model_time - start_time

        return self
    

    def summary(self):
        """
        Displays a summary of the fitted LSTR model including:
        - Model statistics: R-squared, AIC, BIC, DF, N
        - Parameter estimates with standard errors, t-values, and p-values
        - Description of each transition variable (type, function, lag, fixed value, data source)
        """

        print('=' * 80)
        print(' ' * 34 + ' STAR MODEL ' + ' ' * 34)
        print('=' * 80)
        print(f'{"Model Time:":<20}{datetime.fromtimestamp(self.model_time).strftime("%Y-%m-%d %H:%M:%S"):<35}{"R-squared:":<15}{self.rsquared:>10.4f}')
        print(f'{"N:":<20}{self.n:<35}{"AIC:":<15}{self.aic:>10.4f}')
        print(f'{"DF:":<20}{self.df_resid:<35}{"BIC:":<15}{self.bic:>10.4f}')
        print('=' * 80)
        print(f'{"Coefficient":<20}{"Estimate":>15}{"Std. Error":>15}{"t-stat":>15}{"p-value":>15}')
        print('-' * 80)
        for idx, row in self.results_df.iterrows():
            if not np.isnan(row["Std Error"]) and row["Std Error"] < 0.001:
                std_error = ">0.001"
            elif not np.isnan(row["Std Error"]):
                std_error = f'{row["Std Error"]:.4f}'
            else:
                std_error = "nan"
            if not np.isnan(row["t-value"]) and abs(row["t-value"]) > 1000:
                t_stat = "Extreme"
            elif not np.isnan(row["t-value"]):
                t_stat = f'{row["t-value"]:.4f}'
            else:
                t_stat = "nan"
            estimate = f'{row["Estimate"]:.4f}' if not np.isnan(row["Estimate"]) else 'nan'
            p_value = f'{row["p-value"]:.4f}' if not np.isnan(row["p-value"]) else 'nan'
            
            print(f'{row["Parameter"]:<20}{estimate:>15}{std_error:>15}{t_stat:>15}{p_value:>15}')
        print('=' * 80)
        print(f'{"Regime":<10}{"Type":<15}{"Function":<15}{"Lag":<10}{"Value":<10}{"Data/Name":<20}')
        print('-' * 80)
        for idx, var in enumerate(self.threshold_variables, start=1):
            if var['type'] == 'dependent':
                print(f'{idx:<10}{var["type"]:<15}{var["transition_function"]:<15}{var.get("lag", "-"):<10}{var.get("value", "-"):<10}{"Dependent":<20}')
            elif var['type'] == 'time':
                print(f'{idx:<10}{var["type"]:<15}{var["transition_function"]:<15}{"-":<10}{"-":<10}{"Time variable":<20}')
            elif var['type'] == 'independent':
                print(f'{idx:<10}{var["type"]:<15}{var["transition_function"]:<15}{var.get("lag", "-"):<10}{var.get("value", "-"):<10}{var["name"]:<20}')
            elif var['type'] == 'external':
                if isinstance(var.get("data"), pd.DataFrame):
                    data_desc = ", ".join(var["data"].columns)
                elif isinstance(var.get("data"), pd.Series):
                    data_desc = var["data"].name
                else:
                    data_desc = "External data"
                print(f'{idx:<10}{var["type"]:<15}{var["transition_function"]:<15}{var.get("lag", "-"):<10}{"-":<10}{data_desc:<20}')
        print('=' * 80)


    def predict(self, X_new, y_new=None, threshold_new=None, additional_data = False):
        """
        Predicting new observations.
        
        Parameters
        ----------
        X_new : pd.DataFrame
            New independent variables for prediction.
        y_new : pd.DataFrame or None, optional
            Initial values of the dependent variable. If None, a placeholder is created.
        threshold_new : dict or None, optional
            Dictionary mapping regime numbers to DataFrames containing new transition variables.
        additional_data : bool, default False
            If True, returns the full DataFrame including lagged dependent variables
            and transition variables. Otherwise, returns only predicted y values.

        Returns
        -------
        pd.DataFrame or pd.Series
            Predicted values of the dependent variable, optionally with additional columns that contain utilised variables.
        """

        # Copying the input data
        X_new = X_new[self.exog_names].copy()
        
        # Merging the data containing independent variables with the dependent variable
        if y_new is None:
            y_new = pd.DataFrame(index=X_new.index, columns=['y'])
        X_pred = y_new.merge(X_new, left_index=True, right_index=True, how='outer')

        # Adding the lagged variables if they were used in the estimation process
        for l in self.lags:
            X_pred[f'dependent_L{l}'] = X_pred.iloc[:, 0].shift(l)

        # Exceluding the integer
        n_var = X_pred.shape[1] - 1
        # Treatinng the base scenario as an extra regime
        n_regimes = self.transition_variables.shape[1] + 1

        threshold_list = []
        if threshold_new is not None:
            # Soritng the dictionary by keys (regime numbers) and creating the list of Dataframes
            threshold_list = [threshold_new[str(i + 1)] for i in range(n_regimes - 1)]
            # Mering dataset with transition variables
            for df in threshold_list:
                X_pred = X_pred.merge(df, left_index=True, right_index=True, how='outer')

        # Fixing the column names of the prediction dataset
        X_pred.columns = list(X_pred.columns[:n_var + 1]) + self.transition_names

        # Columns transition variables
        threshold_base = X_pred.iloc[:, n_var + 1:]
        # Columns with the dependent and independent variables
        X_pred = X_pred.iloc[:, :n_var + 1]

        sorted_index = X_pred.index.sort_values()
        # First column is the dependent variable
        y_col = X_pred.iloc[:, 0]
        # Checking which rows have non-missing dependent variable
        isnan = y_col.isna()

        # Find the first row from which all remaining dependent values are missing; if none, stop the function
        start_index = None
        for idx_pos in range(len(y_col)):
            if isnan.iloc[idx_pos]:
                if isnan.iloc[idx_pos:].all():
                    start_index = idx_pos
                    break
        if start_index is None:
            return X_pred

        # Convert the starting position from positional index to label-based index in the sorted index
        start_index = sorted_index.get_loc(y_col.index[start_index])
        pred_index = []
        max_index = len(sorted_index) - 1

        # Loop until there is no available information for the independent variables
        while start_index <= max_index:
            index = sorted_index[start_index]

            integer_predictions = self.params[0]
            linear_predictions = []

            # Calculating the linear predictions for each regime. In this moment, each regime is treated as if its weight is 100%
            for i in range(n_regimes):
                linear_predictions.append(
                    np.dot(
                        self.params[(1 + i * n_var):(1 + i * n_var + n_var)],
                        X_pred.loc[index].iloc[1:]
                    )
                )

            # Compute transition variables, applying lags where specified.
            threshold_variables = threshold_base.copy()
            for i, var in enumerate(self.threshold_variables):
                lag = var.get('lag', 0)
                threshold_variables.iloc[:, i] = threshold_variables.iloc[:, i].shift(lag)

            # Calculating the weights for each of the regimes
            regime_weight = []
            for i in range(n_regimes):
                if i == 0:
                    # Base regime
                    regime_weight.append(1)
                else:
                    # All of the nonlinear regimes
                    delta = threshold_variables.iloc[threshold_variables.index.get_loc(index), i - 1] - self.params[n_regimes * n_var + i + n_regimes - 1]
                    coef = self.params[n_regimes * n_var + i]
                    tf = self.transition_functions[i - 1]
                    if tf == 'logistic':
                        regime_weight.append(
                            1 / (1 + np.exp(-coef * delta))
                            )
                    elif tf == 'exponential':
                        regime_weight.append(  
                            1 - np.exp(-coef * delta ** 2)
                        )
                    else:
                        raise ValueError(f"Unknown transition_function: {tf}")

            # Calculating the weighted sum of predictions across all regimes
            y_pred = integer_predictions + sum(l * w for l, w in zip(linear_predictions, regime_weight))
            X_pred.iloc[X_pred.index.get_loc(index), 0] = y_pred

            # Overwriting recursion variables
            for l in self.lags:
                X_pred[f'dependent_L{l}'] = X_pred.iloc[:, 0].shift(l)
            for i_c in threshold_base.columns:
                if "dependent_" in i_c:
                    threshold_base.loc[index, i_c] = y_pred

            pred_index.append(sorted_index[start_index])
            start_index += 1
            
        if additional_data:
            return X_pred
        else:
            y_col_name = X_pred.columns[0]
            return X_pred.loc[pred_index, y_col_name]


    def regime_weights(self, regime):
        """
        Calculates the weights of the transition variables for the selected regime
        in the modeling sample.

        Parameters
        ----------
        regime : int
            Number of the regime for which to compute transition weights (1-based index).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
                - 'transition_variable': values of the transition variable
                - 'weight': corresponding weight in the selected regime
        """

        n_regimes = self.transition_variables.shape[1] + 1
        if regime <= 0 or regime >= n_regimes:
            raise ValueError(f"Wybrany reżim {regime} jest spoza zakresu [1, {n_regimes-1}]")
        
        col_idx = regime - 1
        trans_var = self.transition_variables[:, col_idx].copy()
        
        gamma = self.params[n_regimes * self.X.shape[1] + regime]
        threshold = self.params[n_regimes * self.X.shape[1] + n_regimes - 1 + regime]

        tf = self.transition_functions[col_idx]
        
        if tf == 'logistic':
            weights = 1 / (1 + np.exp(-gamma * (trans_var - threshold)))
        elif tf == 'exponential':
            weights = 1 - np.exp(-gamma * (trans_var - threshold)**2)
        else:
            raise ValueError(f"Unknown transition_function: {tf}")
        
        df = pd.DataFrame({
            'transition_variable': trans_var,
            'weight': weights
        })
        
        return df





def teravista_test(X, y, threshold_variables, lags, lag_threshold):
    """
    Performs a variant of the Teräsvirta test for detecting nonlinearity in regression models
    with multiple threshold (transition) variables.

    The test examines whether the nonlinearity exists by regressing the residuals of the
    linear model on interactions of X with powers (1st, 2nd, 3rd) of the threshold variables.

    Parameters
    ----------
    X : pd.DataFrame
        Indepentent variables to be included in the model.
    y : pd.DataFrame
        Dependent variable.
    threshold_variables : list of dict
        List of dictionaries defining threshold variables. Each dictionary should contain:
        - 'name': the name of the variable in X, y, or external data,
        - 'type': type of the variable, one of 'dependent', 'independent', 'time', 'external',
        - 'data': additional key, required for 'external' type, a pd.Series or pd.DataFrame.
    lags : list of int
        List of lag orders of the dependent variable to be included in the exogenous variables.
    lag_threshold : int
        Maximum number of lags of the threshold variable to be tested.

    Returns
    -------
    test_results : pd.DataFrame
        DataFrame containing the F-test p-values for each threshold variable and power:
        - 'F1': p-value for first power (linear interaction),
        - 'F2': p-value for second power (quadratic interaction),
        - 'F3': p-value for third power (cubic interaction),
        - 'conclusion': suggestion of the model type based on the significance of F-tests:
          - 'linear' (no nonlinearity),
          - 'lstar' (asymmetric nonlinearity),
          - 'estar' (symmetric nonlinearity).

    Notes
    -----
    - The function automatically drops rows with NaN resulting from lagging.
    - Each threshold variable is tested separately in a loop.
    - Interactions are formed between all regressors X and the powers of the threshold variable.
    - The conclusions are based on standard Teräsvirta logic:
        * If F1 or F3 < 0.05 → 'lstar'
        * If F2 < 0.05 → 'estar'
        * Otherwise → 'linear'
    """
    
    if lags:
        for l in lags:
            X[f'dependent_L{l}'] = y.shift(l)

    transition_df = pd.DataFrame(index=y.index)
    ext = 1
    for var in threshold_variables:
        if var['type'] == 'dependent':
            for lag_value in np.arange(0, lag_threshold+1, 1):
                transition_df[f'dep_L{lag_value}'] = y.shift(lag_value)
        elif var['type'] == 'independent':      
            for lag_value in np.arange(0, lag_threshold+1, 1):    
                independent_series = X[var['name']].shift(lag_value).rename(f'ind_{X[var['name']].name}_L{lag_value}')    
                independent_df = independent_series.to_frame()
                transition_df = transition_df.merge(independent_df, left_index=True, right_index=True, how='left')
        elif var['type'] == 'time':
            transition_df[f'time'] = y.div(y).cumsum()
        elif var['type'] == 'external':
            for lag_value in np.arange(0, lag_threshold+1, 1):
                if 'data' in var and isinstance(var['data'], (pd.Series, pd.DataFrame)):
                    external_series = var['data'].shift(lag_value)
                    if isinstance(external_series, pd.Series):
                        external_series = external_series.rename(f'ext_{external_series.name}_L{lag_value}')
                    else:
                        external_series.columns = [f'external_{external_series.name}_L{lag_value}']
                        ext += 1
                    transition_df = transition_df.merge(external_series, how='left', right_index=True, left_index=True)

    _combined_df = y.merge(X, left_index=True, right_index=True, how='left')
    _combined_df = _combined_df.merge(transition_df, left_index=True, right_index=True, how='left')
    _combined_df = _combined_df.dropna()
    y = _combined_df[y.columns]
    X = _combined_df[X.columns]
    transition_df = _combined_df[transition_df.columns]

    transition_columns = transition_df.columns.tolist()
    transition_df = transition_df.to_numpy()
    y = y.to_numpy()
    X = X.to_numpy()
    del _combined_df

    part_0 = np.concatenate([np.ones((X.shape[0], 1)), X], axis = 1)

    results = {}
    for col in range(transition_df.shape[1]):
        part_1 = X*transition_df[:,col][:, np.newaxis]
        part_2 = X*transition_df[:,col][:, np.newaxis]**2
        part_3 = X*transition_df[:,col][:, np.newaxis]**3

        linear_ols = sm.OLS(y, part_0).fit()
        nonlinear_ols = sm.OLS(
            linear_ols.resid,
            np.concatenate([part_1, part_2, part_3], axis=1)
            ).fit()
        
        R_1 = np.concatenate([
            np.identity(X.shape[1]),
            np.zeros([X.shape[1], X.shape[1]]),
            np.zeros([X.shape[1], X.shape[1]])],
            axis = 1)
        R_2 = np.concatenate([
            np.identity(X.shape[1]*2),
            np.zeros([X.shape[1]*2, X.shape[1]])],
            axis = 1)
        R_3 = np.identity(X.shape[1]*3)
        F1 = nonlinear_ols.f_test(R_1).pvalue
        F2 = nonlinear_ols.f_test(R_2).pvalue
        F3 = nonlinear_ols.f_test(R_2).pvalue

        results[col] = {'variable': transition_columns[col], 'F1': F1, 'F2': F2, 'F3': F3}

        test_results = pd.DataFrame(results).T


        def conclusions_func(x):
            if (x['F1'] < 0.05) or (x['F3'] < 0.05):
                return 'lstar'
            elif (x['F2'] < 0.05):
                return 'estar'
            else:
                return 'linear'

        test_results['conclusion'] = test_results.apply(conclusions_func, axis=1)

    return test_results

