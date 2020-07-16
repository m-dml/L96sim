import numpy as np
from delfi.summarystats.BaseSummaryStats import BaseSummaryStats


class Summary_MeanVar(BaseSummaryStats):

    # define summary stats as only mean and variance of observed X_i
    def __init__(self, seed=None):
        super().__init__(seed=seed)
        # should return a matrix n_samples x 1 (mean)
        self.n_summary = 2

    def calc(self, repetition_list):
        # See BaseSummaryStats.py for docstring

        # build a matrix of n_reps x 1
        repetition_stats_matrix = np.zeros((len(repetition_list), self.n_summary))

        # for every repetition, take the mean of the data in the dict
        for rep_idx, rep_dict in enumerate(repetition_list):

            assert len(rep_dict['data']) == 2  # X and Y
            x, y = rep_dict['data'][0], rep_dict['data'][1]

            repetition_stats_matrix[rep_idx, ] = np.array((np.mean(x), np.var(x)))

        return repetition_stats_matrix


class Summary_Schneider2017(BaseSummaryStats):

    # define summary stats as only mean and variance of observe X_i
    def __init__(self, K, J, seed=None):
        super().__init__(seed=seed)
        # should return a matrix n_samples x 1 (mean)
        self.n_summary = 5
        self.K, self.J = K, J

    def calc(self, repetition_list):
        
        # build a matrix of n_reps x 1
        repetition_stats_matrix = np.zeros((len(repetition_list), self.n_summary))

        # for every repetition, take the mean of the data in the dict
        for rep_idx, rep_dict in enumerate(repetition_list):
            
            assert len(rep_dict['data']) == 2 # X and Y
            x, y = rep_dict['data'][0], rep_dict['data'][1]
            
            ymean = y.reshape(-1,self.K,self.J).mean(axis=2).flatten()            
            repetition_stats_matrix[rep_idx, ] = np.array((x.mean(), 
                                                           y.mean(),
                                                           (x**2).mean(),
                                                           (y**2).mean(),
                                                           (x*ymean).mean()))

        return repetition_stats_matrix

    
class Summary_convstats(BaseSummaryStats):
    
    def __init__(self, K, J, obs_nsteps, includes=('X',), seed=None):
        super().__init__(seed=seed)
        # should return a matrix n_samples x 1 (mean)
        assert len(includes) > 0
        self.n_channels = len(includes)
        assert np.all(np.isin(includes, ('X', 'X2', 'Ybar', 'XYbar', 'Y2bar')))
        self.includes = includes        
        self.n_summary = self.n_channels * K * obs_nsteps
        self.K, self.J, self.obs_nsteps = K, J, obs_nsteps
        

    def calc(self, repetition_list):
        # See BaseSummaryStats.py for docstring

        # build a matrix of n_reps x 1
        repetition_stats_matrix = np.zeros((len(repetition_list), self.n_summary))

        # for every repetition, take the mean of the data in the dict

        for rep_idx, rep_dict in enumerate(repetition_list):
            
            assert len(rep_dict['data']) == 2 # X and Y
            x, y = rep_dict['data'][0], rep_dict['data'][1]            

            outlist = []
            if 'X' in self.includes:
                outlist += [x]
            if 'X2' in self.includes:
                outlist += [x**2]
            if 'Ybar' in self.includes:
                outlist += [y.reshape(-1,self.K,self.J).mean(axis=2).flatten()]
            if 'XYbar' in self.includes:
                outlist += [x * y.reshape(-1,self.K,self.J).mean(axis=2).flatten()]
            if 'Y2bar' in self.includes:
                outlist += [(y**2).reshape(-1,self.K,self.J).mean(axis=2).flatten()]
            repetition_stats_matrix[rep_idx, ] = np.concatenate(outlist, axis=0)

        return repetition_stats_matrix

    
class Summary_identity(BaseSummaryStats): 
    """Just apply the identity instead of reducing data.
    Parameters
    ----------
    idx : list or array of int or bool
        Set of data indices to use us sufficient statistics (None for all).
    """

    def __init__(self, seed=None, idx=None):
        super().__init__(seed=seed)
        self.idx = idx

    @copy_ancestor_docstring
    def calc(self, repetition_list):
        # See BaseSummaryStats.py for docstring

        # get the number of samples contained
        n_reps = len(repetition_list)

        # get the size of the data inside a sample
        data0 = repetition_list[0]['data']
        self.n_summary = np.sum([data0[i].size for i in range(len(data0))])

        # build a matrix of n_reps x n_summary
        data_matrix = np.zeros((n_reps, self.n_summary))
        for rep_idx, rep_dict in enumerate(repetition_list):
            data_matrix[rep_idx, :] = np.concatenate(rep_dict['data'])

        return data_matrix if self.idx is None else data_matrix[:, self.idx]
