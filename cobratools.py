##
## Go define the competition_scorer
##

# Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML
from sklearn.metrics import log_loss
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.nn import Linear


def calculate_n_weights(layers):
    """ Calculate the number of weights given the layers """
    n_weights = 0
    
    for l, l_next in zip(layers[:-1], layers[1:]):
        n_weights += (l+1)*l_next
    
    return n_weights


def competition_scorer(predictions, labels):
    """ Define the test scorer """
    return torch.sqrt(nn.MLELoss())


def compute_accuracy(predictions, labels, n_round=4):
    """ Compute accuracy """
    # TODO: check if shape vec or array
    # TODO: check if type tensor or np.array
    
    # Get max predictions
    predictions = predictions.argmax(axis=1)
    
    # Cast to numpy format
    # Not anymore, now input is np.array
    #predictions = predictions.numpy()
    #labels = labels.numpy()

    # Compute accuracy
    accuracy = sum(predictions == labels)/len(labels)
    
    # Round result and x100
    accuracy = round(accuracy * 100, n_round)

    return accuracy


def to_one_hot(vec, max_val = .95, min_val = None, to_int=False):
    """
    Transform a vector into a one-hot encoded matrix
    It enables non zero values with min_val and a different val than 1 with max_val
    It constructs it as a distribution with its terms summing to 1 (not)
    """
    if to_int:
        vec = vec.astype(np.int32)
    min_val = (1 - max_val) / vec.max()
    mat = np.zeros((vec.size, vec.max()+1))
    mat += min_val
    mat[np.arange(vec.size), vec] = max_val

    if isinstance(vec, pd.core.series.Series):
        mat = pd.DataFrame(mat, columns=range(vec.max()+1))
    return mat


class Analysis():
    """
    Apply predefined analysis methods and graphs
    """
    def __init__(self, df):
        self.df = df
        self.m = df.shape[0]
        self.n = df.shape[1]

        self.target = None
        self.default_na_vals = None
        
        self.printer = PrinterStyle()


    def create_ft_grant_ratio(self, ft='district', inplace=True):
        """
        Ratio that appreaciates the probability to be granted at 
        least a night given:
        - district: the place of the center that was requested
        - town: the place of leaving of the individual
        """
        # Set feature name
        ft_name = ft + '_grant_ratio'

        # Make alias and check exists
        if self.target:
            target = self.target
        else:
            raise ValueError("Target variable has not been set. "\
                            "Use self.target=<var_name> to set the variable")

        # Number of nights granted for each ft (by request)
        target_by_ft = self.df[[ft, target]].dropna()\
                            .groupby(ft)[[target]].sum()\
                            .sort_values(by=target, ascending=False)

        # Number of requests by ft
        n_rq_by_ft = self.df[ft].dropna().value_counts()

        # Mapping of ft with their ratio of grant
        df_ft = pd.merge(target_by_ft,
                                n_rq_by_ft,
                                left_index=True,
                                right_index=True)

        # Compute sort of proba to be granted by ft (precisely: n_nights/n_requests)
        # where ft means number of requests
        df_grant_ratio_ft = pd.DataFrame({ft_name: round(df_ft[target]/df_ft[ft], 2)})

        # Transform dataframe to dic
        dic = df_grant_ratio_ft.to_dict()[ft_name]

        # Create a list of ratios corresponding to each sample
        new_feat = [dic[name] for name in self.df[ft]]

        if inplace:
            # Add ft_grant_ratio to dataframe
            self.df[ft_name] = new_feat
        else:
            return new_feat


    def create_ft_group_age_bounds(self, obj_full_data, inplace=True):
        """
        Create 2 features: group age min and max
        based on birth_year of each individual in the group

        obj_full_data: the dataframe containing all samples
        both from requests and individuals data sets
        """
        # Set current year
        year_curr = pd.datetime.now().year

        # Compute age for each sample from birth_year
        s_age = year_curr - obj_full_data.df['birth_year']

        # Add the index to the series, and transform it as df
        df_age = pd.DataFrame(s_age).set_index(obj_full_data.df.request_id)

        # Compute max age by request_id
        max_age = df_age.groupby('request_id').max()
        min_age = df_age.groupby('request_id').min()

        # Merge min and max age dfs
        df_age_merged = pd.merge(max_age,
                                min_age,
                                left_index=True,
                                right_index=True)

        # Rename columns
        df_age_merged.columns = ['age_max', 'age_min']

        # Merge with obj_train.df or obj_test.df
        merged_df = pd.merge(self.df,
                                df_age_merged,
                                left_index=True,
                                right_index=True)
        
        # Replace the merged df by current one or return it
        if inplace:
            self.df = merged_df
        else:
            return merged_df


    def describe(self, investigation_level=2, header=False):
        """
        Overview of shape, features, header and statistics
        investigation_level: int in [1, 2]

        header is to display preferably for slim datasets (very few columns)
        """
        # Display shape and list features
        self.printer.title("Properties")
        print("- Shape: {shape}\n\n"
              "- Features: {features}"
              .format(shape=self.df.shape,
              features=self.df.columns.values))
        
        # Display data header
        # TODO: create a nice display
        if investigation_level > 0:
            if header:
                self.printer.title("Header")
                print(self.df.head())
        
        # Display complementary stats
        if investigation_level > 1:
            self.printer.title("Correlations")
            print(self.df.corr())
            self.printer.title("Median")
            print(self.df.median())

        if investigation_level > 2:
            self.printer.title("Skewness")
            print(self.df.skew())
            self.printer.title("Kurtosis")
            print(self.df.kurtosis())


    def export_data(self, file_name='data.csv'):
        """ Export data to csv file """
        self.df.to_csv(file_name)


    def fill_na_most_freq_pair(self, ft1, ft2, mapping, inplace=False):
        """
        Fill a feature (ft1) with a pair feature (ft2) using a mapping
        of most frequent pairs
        """
        # Mask ft1 Nans
        ma_na = self.df[ft1].isna()

        # Get ft2 corresponding to samples where ft1 is NaN
        # (vector of the same size, that will then be merged to the dataframe)
        ft2_na = self.df[ft2] * ma_na

        # Transform dic 'dic_ft1_max' into DataFrame
        df_ft1_max = pd.DataFrame.from_dict(mapping,
                                            orient='index',
                                            columns=[ft1])
        # Name the series
        df_ft1_max.index.name = ft2

        # Match apparent ft2 values with their ft1 pair
        df_matched = pd.merge(pd.DataFrame({ft2:ft2_na}),
                            df_ft1_max, on=ft2, how='left')

        # Extract the original ft2-ft1 vectors
        df_ft2_ft1 = self.df[[ft2, ft1]]

        # Combine the 2 ft1 vectors into one column, keeping the first str (non NaN)
        # Functions 'combine' and 'combine_first' had strange behaviors on my laptop
        # (even though working properly on collab), hence the use of a universal solution
        ft1_filled = [v1 if isinstance(v1, str) else v2 for v1, v2 in zip(df_ft2_ft1[ft1], df_matched[ft1])]

        if inplace:
            # Replace current ft1 in self.df
            self.df[ft1] = ft1_filled
        else:
            # Return result without replacement
            return ft1_filled


    def get_na_counts(self, non_zero=True):
        """
        Returns the number of NAs for each feature 
        Returns only if NA count > 0
        - by feature
        - by sample
        """
        mask_na = self.df.isna()
        na_ft = mask_na.sum(axis=0)
        na_sp = mask_na.sum(axis=1)
        
        if non_zero:
            na_ft = na_ft[na_ft!=0]
            na_sp = na_sp[na_sp!=0]

        return na_ft, na_sp


    def get_col_uniques(self, col, dropna=True):
        """
        Returns the set of unique values for the given column
        """
        if dropna:
            return self.df[col].dropna().unique()
        else:
            return self.df[col].unique()

    
    def get_cols_type(self):
        """
        Detect features' category
        - date
        - num
        - bool
        - cat_few
        - cat_med
        - cat_many
        - (not_recognized)
        - (empty)

        input: df with columns of a single type (can still have NaNs)

        output: list containing the type for each feature
        """
        list_types = []

        for feature in self.df:
            # Retrieve unique values for the feature
            ft_set = self.get_col_uniques(feature)
            
            # Pure boolean
            if isinstance(ft_set[0], np.bool_):
                list_types.append('bool')

            # Numeric types / bool
            elif isinstance(ft_set[0], (np.integer, np.float)):
                if len(ft_set) == 1:
                    list_types.append('empty')
                elif len(ft_set) == 2:
                    list_types.append('bool')
                else:
                    list_types.append('num')
            else:
                try:
                    # Date type
                    isinstance(pd.to_datetime(ft_set[0]), pd.datetime)
                    list_types.append('date')
                except:
                    if isinstance(ft_set[0], str):
                        if len(ft_set) == 1:
                            list_types.append('empty')
                        elif len(ft_set) == 2:
                            list_types.append('bool')
                        elif len(ft_set) <= 10:
                            list_types.append('cat_few')
                        elif len(ft_set) <= 30:
                            list_types.append('cat_med')
                        else:
                            list_types.append('cat_many')
                    else:
                        list_types.append('not_recognized')
        
        return list_types


    def get_features_mapping(self, ft1, ft2):
        """
        Returns a mapping of unique pairs between two features
        When multiple features are corresponding, it selects
        the one with max frequency
        """
        # Get ft2 uniques
        ft1_uniques = self.df[ft1].unique()

        # Get frequency of ft2 for each ft1 (build multi-index df)
        mapping = self.df[[ft1, ft2]].dropna().pivot_table(
                                            index=[ft1,ft2], aggfunc='size')

        # Dic that will contain the mapping
        dic_ft2_max = {}

        # Search the corresponding ft2 for each ft1
        # the selected ft2 is those with max frequency
        for ft1_i in ft1_uniques:
            try:
                # Get the list of ft2_i for the selected ft1_i
                # along with their frequency count
                df_ft1_i = mapping.loc[ft1_i, :]
                
                # Get the index corresponding to the ft2_i with max count
                idx_ft2_i_max = df_ft1_i.argmax()
                
                # Get the ft2_i name given its idx
                ft2_i_max = df_ft1_i.index[idx_ft2_i_max]
                
                # Add key ft1_i and set its correponding ft2_i
                dic_ft2_max[ft2_i_max[0]] = ft2_i_max[1]
            except:
                # Elements with ft2_i==NaNs only can't be assigned a ft2_i
                # Thus, add key along with a unique but dumb value
                dic_ft2_max[ft1_i] = "no_pair_for_" + str(ft1_i)
            
        return dic_ft2_max


    def get_data_chunck(self, df=None, iloc_start=None, iloc_end=None, chunck_size=1000):
        """
        Returns a slice of the dataframe
        """
        # Set iloc_start and iloc_end if not defined
        if not iloc_start:
            iloc_start = 0
        if not iloc_end:
            iloc_end = iloc_start + chunck_size

        # If no df passed as input, return a slice of self.df
        if df is None:
            try:
                return self.df.iloc[iloc_start:iloc_end]
            except:
                return self.df.iloc[iloc_start:]
        
        else:
            # If a df is passed as input, return its slice
            try:
                return df.iloc[iloc_start:iloc_end]
            except:
                return df.iloc[iloc_start:]



    def set_default_na_vals(self):
        """
        Define the default values to replace if no other
        imputation method was defined for the case

        TODO: implement method to define default values
        - poor: overall average based on train set average
        - slightly better: average of a similar group
        """
        self.default_na_vals = self.df.dropna().iloc[0]


    def impute_child_to_come(self, df_indiv):
        """
        Specific method to impute child_to_come NaNs
        > Impute child_to_come from pregnancy in the group of indiv of the request
        """
        # Create mask that says if pregnancy true for each REQUEST (±2min)
        ma_child_t_c = df_indiv['pregnancy'].groupby(df_indiv['request_id']).apply(lambda x: max(x=='t'))

        # Sort request_train to match with the mask
        self.df.sort_index(inplace=True)

        # Control that indexes match perfectly
        if not sum(self.df.index != ma_child_t_c.index) == 0:
            print("Issue in matching of indexes, will certainly corrupt data")

        # Set child_to_come as True if pregnancy 't', False otherwise
        self.df['child_to_come'] = ma_child_t_c

        # Check that no NaN remains
        if not self.df['child_to_come'].isna().sum() == 0:
            print("NaNs remaining")
            

    def impute_nans(self):
        """
        Method built essentially for in-production "test" samples.

        If no method was implemented to impute a given feature' value,
        set a default value to prevent interuption.
        Else, try to apply the methods in order of preference

        TODO: use self.default_na_vals[feature]
        """
        na,_ = self.get_na_counts(non_zero=True)
        
        for col in na.index:
            mask_na_rows = self.df[col].isna()
            self.df.loc[mask_na_rows, col] = self.default_na_vals[col]


    def transform_categories(self, true_val='true', false_val='false', target=None, verbose=False):
        """
        Preprocess all columns

        - bools: replace string 't' 'f' by True False equivalents
        - cat_few: replace strings by one-hot encodings

        Other types are left to manual preprocessing

        Ps: Apply only if database is congruent (always the same strings
        are used for true/false)

        Return:
        - list of columns that failed transformation*
        - (list) mapping between true/false and old cat names*
        * for convert_to_bool only

        """
        list_failed = []
        mapping_true_false_cols = []

        # Transform each categorical column
        for col, col_type in zip(self.df, self.get_cols_type()):

            # Don't modify the target variable
            if not col == target:
                
                # Convert bool values with 
                if col_type == 'bool':
                    mapping_col, failed = self.convert_to_bool(col=col,
                                                    true_val=true_val,
                                                    false_val=false_val,
                                                    verbose=verbose)

                    mapping_true_false_cols.append((mapping_col))

                    if failed:
                        list_failed.append(col)
                
                # Tranform to one-hot categories
                elif col_type in ['cat_few', 'cat_med']:
                    self.convert_to_onehot_enc(col=col)
        
        return mapping_true_false_cols, list_failed



    def convert_to_bool(self, col, true_val='true', false_val='false', verbose=False):
        """
        Replace boolean values of a column by 1 and 0
        
        Ps: inplace operation

        Return:
        - mapping_true_false
        - indication if failed
        """
        recognized = False
        uniques = self.get_col_uniques(col)

        # If bool vaues not recognized, try to recognize frequent ones
        if (not true_val in uniques) or (not false_val in uniques):
        
            # Force df unique values to be lower letters
            uniques_transfo = [str(val).lower() for val in uniques]

            # Frequent bool values to test
            freq_bools = [
                ('true', 'false'),
                ('t', 'f'),
                ('1', '0'),
                ('1.0', '1.0'),
                ('male', 'female'),
                ('yes', 'no')
            ]
            # Try to recognize one of these pairs
            for bools in freq_bools:

                # If pair is recognized => assign its values to true_val/false_val
                if bools[0] in uniques_transfo and bools[1] in uniques_transfo:
                    
                    # If true is in first position
                    if bools[0] == uniques_transfo[0]:
                        true_val = uniques[0]
                        false_val = uniques[1]
                    else:
                        true_val = uniques[1]
                        false_val = uniques[0]
                    
                    if verbose:
                        print("Transform Boolean at col {}: "
                                "True/False={}".format(col, uniques))
                    recognized = True
                    break

            if not recognized:
                if verbose:
                    print("\nERROR - Transform Boolean at col {} "
                            "not recognized: {}\n".format(col, uniques))
                
                return (np.nan, np.nan), True  # True means failed

        # Replace (recognized) bool values
        bools_map = {
            true_val: int(1),
            false_val: int(0)
        }
        self.df[col] = self.df[col].map(bools_map)
        
        # Cast new type
        self.df[col] = self.df[col].astype(np.uint8)

        return (true_val, false_val), False


    def convert_to_onehot_enc(self, col):
        """
        Replace col (contains categories) by one-hot-encodings
        
        Ps: inplace operation
        """
        df_dummies = pd.get_dummies(self.df[col], prefix=col+'_cat', dtype=np.uint8)
        self.df = pd.concat([self.df, df_dummies], axis=1)
        self.df.drop([col], axis=1, inplace=True)


    def visualize(self, investigation_level=1, subplot_n_cols_limit=3, barplot_n_classes_limit=10):
        """
        Plot main features
        """
        # Show histograms of numeric features
        # -----------------------------------

        n_features_numeric = self.df.select_dtypes(include=['number']).shape[1]
        
        # Generate subplot grid
        fig, axes = self.generate_subplot_grid(n_features_numeric, subplot_n_cols_limit)

        # Generate subplots
        for i_feature, str_feature in enumerate(self.df.select_dtypes(include=['number'])):
            n_row = self.get_row_n(i_feature + 1, subplot_n_cols_limit)
            n_col = self.get_col_n(i_feature + 1, subplot_n_cols_limit)
            
            # Plot by referencing to a grid of either 1 or 2 dimensions
            if n_features_numeric > subplot_n_cols_limit:
                self.df[str_feature].plot.hist(ax=axes[n_row-1, n_col-1], title=str_feature)
            else:
                self.df[str_feature].plot.hist(ax=axes[n_col-1], title=str_feature)
        plt.show()

        # Show bar plots of categorical features
        # --------------------------------------  
      
        # Assess total number of features to plot, since some cant be plot
        n_features_class = 0
        for str_feature in self.df.select_dtypes(include=['object']):
            # Only plot if number of classes < plot_limit_n_classes
            if pd.get_dummies(self.df[str_feature]).shape[1] < barplot_n_classes_limit:
                n_features_class += 1

        # Generate subplot grid
        fig, axes = self.generate_subplot_grid(n_features_class, subplot_n_cols_limit)

        # Generate subplots
        n_passed = 0
        for i_feature, str_feature in enumerate(self.df.select_dtypes(include=['object'])):
            
            # Only plot if number of classes < plot_limit_n_classes
            if pd.get_dummies(self.df[str_feature]).shape[1] < barplot_n_classes_limit:
                n_row = self.get_row_n(i_feature + 1 - n_passed, subplot_n_cols_limit)
                n_col = self.get_col_n(i_feature + 1 - n_passed, subplot_n_cols_limit)
                
                # Plot by referencing to a grid of either 1 or 2 dimensions
                if n_features_class > subplot_n_cols_limit:
                    pd.get_dummies(self.df[str_feature]).sum()\
                    .plot.bar(ax=axes[n_row-1, n_col-1], title=str_feature)
                else:
                    pd.get_dummies(self.df[str_feature]).sum()\
                    .plot.bar(ax=axes[n_col-1], title=str_feature)
            else:
                # Not plotted because n_classes > plot_limit_n_classes
                n_passed += 1
        plt.show()


    @staticmethod
    def get_row_n(x, n):
        return 1 + (x - 1) // n


    @staticmethod
    def get_col_n(x, n):
        return x + (-n * ((x - 1) // n))


    def generate_subplot_grid(self, n_features, subplot_n_cols_limit):
        """
        Generate a subplot grid with:
        - number of subplots = to n_features
        - disposition depending on subplot_n_cols_limit
        - grid size proportionnal to the number of subplots
        """
        # Compute limits for subplot grid
        n_rows_total = self.get_row_n(n_features, subplot_n_cols_limit)
        if n_features > subplot_n_cols_limit:
            n_cols_total = subplot_n_cols_limit
        else:
            n_cols_total = n_features

        # Define grid size
        fig_size = (4 + (1.3 * n_cols_total), 4 + (1 * n_rows_total))
        # Generate grid
        fig, axes = plt.subplots(nrows=n_rows_total, ncols=n_cols_total, figsize=fig_size)
        # Define subplots' size
        plt.subplots_adjust(hspace=0.15*n_rows_total, wspace=0.15*n_cols_total)
        
        return fig, axes



class Dataset(Dataset):

    # Constructor
    def __init__(self, X, Y, y_long=True):
        # Cast Y to long for CrossEntropyLoss
        if y_long:
            self.Y = Y.type(torch.LongTensor)
        else:
            self.Y = Y.type(torch.FloatTensor)

        self.X = X
        self.len=len(self.X)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        x=self.X[idx]
        y=self.Y[idx]
        return x, y



class Monitoring:
    """
    Class to monitor models performances

    Uses external metrics functions, which must be defined
    as follows (example):
    
    Example:
    --------
    metrics = {'accuracy': func_accuracy,
               'loss_model': func_criterion}

    TODO: include epochs and batchs handling, 
    removing the need to create two instances,
    removing the need to pass batch object to 
    epoch object to compute aggregates
    """
    def __init__(self, sets, metrics, precision=4):
        self.sets_names = sets
        self.metrics_info = metrics
        self.metrics = self._init_perf(sets, metrics)
        self.precision = precision
        
        # Counters
        self.epoch = 0
        self.batch = 0


    def _init_perf(self, sets=None, metrics=None):
        """
        Initialize a dictionnary to store multiple metrics 
        for each type of data set

        Example:
        --------
        perfs = {'train': {'accuracy': [],
                        'loss': []}
                'test':  {'accuracy': [],
                        'loss': []}}
        """
        # Set default sets and metrics if not passed as input
        if not sets:
            sets = ['train', 'test']
        if not metrics:
            metrics = ['accuracy', 'loss']

        dic_perfs = {}
        
        for set_i in sets:
            # Initialize dictionaries
            dic_metrics = {}

            # Construct the dic of metrics history
            for metric in metrics:
                dic_metrics[metric] = []

            # Construct perf dic
            # - a key for each dset
            # -- each including all metric dics
            dic_perfs[set_i] = dic_metrics

        return dic_perfs


    def evaluate(self, predictions, labels, set_i='train', precision=4):
        """
        Method that calls each metric function and store the result
        to the corresponding set_i and metric history.

        - The score stored is in format float, i.e. has not grad attached
        - The function outputs the model score with its grad attached
        """
        # Evaluate each metric
        for metric in self.metrics_info:

            # Retrieve metric function
            func = self.metrics_info[metric]
            
            # Functions that work np.arrays
            if metric in ['accuracy', 'loss_compet']:

                # Compute score with casting tensors to numpy arrays
                np_score = func(F.softmax(predictions, dim=1).detach().numpy(), labels.numpy())
            else:
                # Below functions need tensors+grad

                # Call metric with generic parameters
                grad_score = func(predictions, labels)

                # Cast a copy to store in numpy
                np_score = grad_score.detach().numpy().item()

            # Round score
            np_score = round(np_score, precision)

            # Add the result to the history
            self.metrics[set_i][metric].append(np_score)

        # After all metrics evaluated, return model score
        if 'loss_model' in self.metrics_info:
            return grad_score


    def compute(self, obj_batchs, sets, precision=4):
        """ Only for epoch: computes the average of each set_i batches """
        # TODO: implement for other types

        for set_i in sets:
            for metric in obj_batchs.metrics_info:
                
                # Retrieve scores
                scores = obj_batchs.metrics[set_i][metric]
                
                if scores:
                    # Compute average and round
                    score_avg = np.array(scores).mean()

                    score_avg = round(score_avg, self.precision)

                    # Add score to the history
                    self.metrics[set_i][metric].append(score_avg)



    def print_scores(self, metrics=None, i_epoch=None, sets=None):
        """ Print metrics' values """
        # TODO: conditional print given inputs

        if not sets:
            sets = self.sets_names
        
        if 'train' in sets:
            print(f"Epoch {i_epoch} train/val: "\
                f"mod_loss {self.metrics['train']['loss_model'][-1]}, "\
                    f"{self.metrics['eval']['loss_model'][-1]}, "\
                f"comp_loss {self.metrics['train']['loss_compet'][-1]}, "\
                    f"{self.metrics['eval']['loss_compet'][-1]}, "\
                f"acc {self.metrics['train']['accuracy'][-1]}, "\
                    f"{self.metrics['eval']['accuracy'][-1]}")
        
        if 'test' in sets:
            print(f"Epoch {i_epoch} test: "\
                f"mod_loss {self.metrics['test']['loss_model'][-1]}, "\
                f"comp_loss {self.metrics['test']['loss_compet'][-1]}, "\
                f"acc {self.metrics['test']['accuracy'][-1]}")


    def reset(self):
        """ Re-init history """
        self.metrics = self._init_perf(self.sets_names, self.metrics_info)



class NN(nn.Module):
    def __init__(self, layers, p=0, seed=500):
        super(NN, self).__init__()
        self.hidden_layers = nn.ModuleList()

        # Set seed to reproduce results
        torch.manual_seed(seed)
        
        for input_size, output_size in zip(layers, layers[1:]):
            self.hidden_layers.append(nn.Linear(input_size, output_size))
            #self.hidden_layers.append(nn.Dropout(p=p))
            self.hidden_layers.append(nn.BatchNorm1d(output_size))

    def forward(self, activation):
        for i_layer, linear_transform in enumerate(self.hidden_layers):
            if i_layer < len(self.hidden_layers) - 1:
                activation = torch.relu(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation



class PrinterStyle():
    """
    Contains functions to apply a custom style to pandas' print outputs
    and functions to print with a predefined style (e.g. titles)
    """
    def __init__(self, n_line_jumps=2):
        self.n_line_jumps = n_line_jumps

    def title(self, str_title):
        """
        Print title with space and a delimiter for
        a clear display of info
        """
        delimiter = "-" * len(str_title)
        line_jumps = "\n" * self.n_line_jumps
        print("{line_jumps}"
            "{title}\n"
            "{delim}"
            "{line_jumps}"
            .format(title=str_title, delim=delimiter, line_jumps=line_jumps))
