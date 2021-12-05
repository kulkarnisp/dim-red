from dimred.data.loader import LoadOne
from dimred.models.linear.transform import co_variance, co_kurtosis
from dimred.data.preprocess import MinMaxScalar, ZeroMeanScalar
from dimred.data.preprocess import Scalar
from dimred.models.linear.transform import Kurtosis

import numpy
import matplotlib.pyplot as plot

class DimRedAnalysis:
    '''Class to analyse the datasets for dimensionality reduction.

    Inputs:
        dataset: The combustion dataset to be used (default 1D syngas)
        scaling_type: Type of scaling (default MinMax)
    '''
    def __init__(self, dataset=None, scaling_type=None):
        #instantiate the loader object
        self.loader = LoadOne() if isinstance(dataset, type(None)) \
                            else LoadOne(dataset)
        print("The variables in the dataset are:\n\t%s" %self.loader.info.specs)
        # set the scaling type
        self.scaling_type = MinMaxScalar if isinstance(scaling_type, type(None)) \
                            else scaling_type
        # instantiate the scaler object
        self.scaler = Scalar(scaling_type)
        # create dicts for the covariance, cokurtosis principle vectors
        # and values
        self.pvecs_cov = dict()
        self.pvecs_kur = dict()
        self.pvals_cov = dict()
        self.pvals_kur = dict()

        # get dimensions of dataset
        self.dataset_dimensions = self.read_and_clean_data(0).shape

    def analyse_checkpoint(self, checkpoint_id):
        '''Method to analyse one checkpoint
        
        Inputs:
            1.) Checkpoint Id (int)
        
        Returns:
            1.) Original data at the specified checkpoint
            2.) scaled data at the specified checkpoint
            3.) Principal vectors obtained using Covariance
            4.) Principal values  obtained using Covariance
            5.) Principal vectors obtained using Cokurtosis
            6.) Principal values  obtained using Cokurtosis

        The method also adds the principal values and vectors
        obtained from the Co-variance and co-kurtosis tensors at the
        specified checkpoint to global dictionaries.
        '''
        # read and retain required data at specified checkpoint
        data = self.read_and_clean_data(checkpoint_id)
        data_old = data.copy()
        # scale the data for the analysis
        self.scaler.fit(data)
        scaled_data = self.scaler.transform(data)
        # obtain principal vectors and values using co-variance 
        # and co-kurtosis
        cov_pvecs, cov_pvals = self.get_pcs_and_pvs(data=scaled_data,
                                                    moment_type=co_variance)
        kur_pvecs, kur_pvals = self.get_pcs_and_pvs(data=scaled_data,
                                                    moment_type=co_kurtosis)
        # append the calculated values to the global dicts
        self.pvecs_cov[checkpoint_id] = cov_pvecs
        self.pvecs_kur[checkpoint_id] = kur_pvecs
        self.pvals_cov[checkpoint_id] = cov_pvals
        self.pvals_kur[checkpoint_id] = kur_pvals

        return (data_old, scaled_data, cov_pvecs, cov_pvals,
                kur_pvecs, kur_pvals)

    def get_reconstructed_data(self, scaled_data, cov_pvecs,
            kur_pvecs, retain=-1, reconstruction_type='linear'):
        '''Get reconstructed data from the scaled data. The method
        allows to recontruct the data based on a specified number
        of principal components that are to be retained.

        Inputs:
          1.) scaled_data = scaled form of the input data at a
                              specified checkpoint
          2.) cov_pvecs = principal vectors obtained with co-variance
          3.) kur_pvecs = principal vectors obtained with co-kurtosis
          4.) retain    = number of principal vectors to retain

        Returns:
          1.) full rank reconstructed data using co-variance based
              principal components
          2.) full rank reconstructed data using co-kurtosis based
              principal components
          3.) full rank reconstructed scaled data using co-variance
              based principal components
          4.) full rank reconstructed scaled data using co-kurtosis
              based principal components
        '''
        # isolate retained vectors
        keep_features = self.max_features if retain == -1 else retain
        retain_cv_vecs = cov_pvecs[..., :retain]
        retain_ku_vecs = kur_pvecs[..., :retain]
        # obtain the lower rank data based on the priciple vectors

        cov_lower_rank = numpy.dot(scaled_data, retain_cv_vecs)
        kur_lower_rank = numpy.dot(scaled_data, retain_ku_vecs)
        # reconstruct the data from the lower rank data
        if reconstruction_type == 'linear':
            cov_full_rank_scaled, cov_full_rank = \
                self.linear_reconstruction(cov_lower_rank, retain_cv_vecs)
            kur_full_rank_scaled, kur_full_rank = \
                self.linear_reconstruction(kur_lower_rank, retain_ku_vecs)

        return (cov_full_rank, kur_full_rank, 
                cov_full_rank_scaled, kur_full_rank_scaled)

    def read_and_clean_data(self, idx):
        '''Read and clean the data to return an ndarray of size
        (nGridPoints, nSpecies+1)
        '''
        data = self.loader.readFile(self.loader.flist[idx])
        ### p, vx, vy, vz are the last four features of the dataset ###
        # identify maximum number of features
        self.max_features = data.shape[-1] - 4
        # return data without the last four features
        return data[..., :-4]

    def get_pcs_and_pvs(self, data, moment_type, retain=-1):
        '''Method to obtain the principal components (vectors) & principle
        values from the data using a specified type of moment

        Inputs:
            1.) data = numpy ndarray containing the data with
                       Nspecies + 1(Temp) variables
            2.) moment_type = function to evaluate the covariance/cokurtosis
            3.) retain = number of components to retain
                        (by default retain all components)

        Returns: ndarrays for the principle vectors and values
        '''
        keep_features = self.max_features if retain == -1 else retain
        moment_matrix = moment_type(data)
        u,s,v = numpy.linalg.svd(moment_matrix.T, full_matrices=False)
        vectors = u[...,:keep_features] 
        values = s[:keep_features]
        return vectors, values

    def linear_reconstruction(self, data, vectors):
        # reconstruct the scaled data
        recon_scaled = numpy.dot(data, vectors.T)
        # undo scaling in the reconstructed scaled data
        # !!! transform2 is implemented only for the MinMax Scaling
        reconstructed_data = self.scaler.transform2(recon_scaled)
        return recon_scaled, reconstructed_data
    
    def plot_pvals_vs_modes(self, scaled_data, idx, region=0):
        '''Plot the Principal Components against the Eigen modes.
        Note this method takes in the scaled data and can therefore
        create plots for any region of interest.
        '''
        # obtain principal vectors and values using co-variance 
        # and co-kurtosis
        cov_pvecs, cov_pvals = self.get_pcs_and_pvs(data=scaled_data,
                                                    moment_type=co_variance)
        kur_pvecs, kur_pvals = self.get_pcs_and_pvs(data=scaled_data,
                                                    moment_type=co_kurtosis)

        plot.semilogy(cov_pvals/numpy.sum(cov_pvals), label="Co-Variance")
        plot.semilogy(kur_pvals/numpy.sum(kur_pvals), label="Co-Kurtosis")
        plot.xlabel('Eigen Modes')
        plot.ylabel('Principal Values')
        plot.xticks(numpy.arange(self.max_features))
        plot.legend()
        plot.savefig("pvals_vs_modes_id%s_region%s"%(idx, region))
        
    def plot_inclination(self, scaled_data, idx, region=0):
        '''Plot the inclination of the principal components obtained
        from the co-variance and co-kurtosis tensors.
        Note this method takes in the scaled data and can therefore
        create plots for any region of interest.
        '''
        # obtain principal vectors and values using co-variance 
        # and co-kurtosis
        cov_pvecs, cov_pvals = self.get_pcs_and_pvs(data=scaled_data,
                                                moment_type=co_variance)
        kur_pvecs, kur_pvals = self.get_pcs_and_pvs(data=scaled_data,
                                                moment_type=co_kurtosis)
        if len(cov_pvecs.shape) == 2:
            inclination = numpy.einsum('ij, ij -> j', cov_pvecs,
                    kur_pvecs)
        plot.plot(numpy.absolute(inclination))
        plot.xlabel('Eigen Modes')
        plot.ylabel('Inclination')
        plot.savefig("inclination_id%s_region%s"%(idx, region))
