import numpy as np
from matplotlib import pyplot as plt
# Get the Huber loss function
from scipy.special import huber
from astropy.cosmology import WMAP9 as cosmo
from scipy import interpolate
from sklearn.utils import check_random_state

def make_fake_data(N, func, noise_scale=0.5):
  """Create fake data using a given function

  Arguments
  ---------
      N: int
         The number of data points to return
      func: function
         The function to call to get the data. This must take 
         one argument (the x-value) and return the y-value.

  Keywords
  --------
      noise_scale : float, default 0.5
         The variance of the normal distribution that is
         used to assigned random noise to the data. Set to
         zero to add no noise.

  Returns
  -------
      xvalues : numpy array
         The x-values used for calculating the data points 
      yvalues : numpy array
         The y-values, using true_func and then adding noise.
  """

  rng = np.random.default_rng()
  xvalues = rng.uniform(0, 4, N)
  yvalues = func(xvalues)+rng.normal(0, noise_scale, len(xvalues))

  
  return xvalues, yvalues


def show_data_func(x, y, func, xrange=None, yrange=None, ax=None):
  """A convenience function for showing data points and the true function
  
  Arguments
  ---------
      x : numpy array
          The x-values for the data
      y : numpy array
          The y-values for the data
      func : function
          The true function to overplot the data
          
  Keywords:
  ---------
      xrange : two-element array, optional
          The x-range to show.
      yrange : two-element array, optional
          The y-range to show.
      ax : matplotlib Axis object, optional
          The axis object to plot into, if not 
          provided, an axis object is created and
          returned.
  """
  xplot = np.linspace(np.min(x), np.max(x), 500)
  yplot = func(xplot)
  if (ax is None):
    fig, ax = plt.subplots(ncols=1, nrows=1)
  ax.scatter(x, y, label='Data points')
  ax.plot(xplot, yplot, label='True function', color='orange')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  if xrange is not None:
    ax.set_xlim(xrange)
  if yrange is not None:
    ax.set_ylim(yrange)

  ax.legend()



def loss_Lp(y, y_pred, p):
    """L^p loss function for p > 0"""
    if p == 0:
        print("This loss function is for p > 0 - use loss_L0 for p=0")
        return None

    if p == 1:
        z = np.sum(np.abs(y-y_pred))
    else:
        z = np.sum((y-y_pred)**p)
        
    return z**(1/p)

def loss_L0(coeffs, tol=1e-15):
    """L0 loss function - this just counts the non-zero coefficients"""

    is_zero, = np.where(np.abs(coeffs) < tol)
    loss = len(is_zero)
    return L0

def loss_10(y, y_pred, tol=1e-15):
    """0-1 loss with tolerance"""
    diff = np.abs(y-y_pred)
    return loss_L0(diff)

def loss_MAE(y, y_pred):
    """Median absolute error"""
    N = len(y)
    return loss_Lp(y, y_pred, 1)/N

def loss_MSE(y, y_pred):
    """Mean square error"""
    N = len(y)
    return np.sum((y-y_pred)**2)/N

def loss_Huber(y, y_pred, delta=1.0):
    """Huber loss function"""
    N = len(y)
    return np.sum(huber(delta, y-y_pred))/N

def true_func(x):
  """The true function used by `make_fake_data`"""
  return np.sin(x*np.pi)+x/3


# We will fit up to an order of max_order with no default
def fit_polynomials_to_xy(x, y, max_order=None):
    """Fit a polynomial to input x & y data

    Parameters
    ----------
        x : numpy array
            Input x-data
        y : numpy array
            Input y-data

    Keywords
    --------
        max_order : int
            Maximum polynomial order to consider. If not given 
            this is set to the number of elements in `x`.

    Return
    ------
        orders : numpy.array
            The orders to consider
        MSE : numpy.array
            The mean square error
        best_fit: list
            A list with the polynomial coefficients of each fit.
    """
    if max_order is None:
        max_order = len(x)
    
    orders = np.arange(max_order)
    MSE = np.zeros(max_order)

    
    best_fit = []

    for i, order in enumerate(orders):
        # Fit the training sample using polynomial fitting
        p = Polynomial.fit(x, y, deg=order)
        best_fit.append(p)

        # Calculate the best fit on the training sample
        mu_fit_train = p(x)
        MSE[i] = np.sum((y-mu_fit_train)**2)/len(x)

    return orders, MSE, best_fit


##
# Taken from: https://www.astroml.org/_modules/astroML/datasets/generated.html#generate_mu_z
#
##

def redshift_distribution(z, z0):
    return (z / z0) ** 2 * np.exp(-1.5 * (z / z0))

def generate_mu_z(size=1000, z0=0.3, dmu_0=0.1, dmu_1=0.02,
                  random_state=None, cosmo=None):
    """Generate a dataset of distance modulus vs redshift.

    Parameters
    ----------
    size : int or tuple
        size of generated data

    z0 : float
        parameter in redshift distribution:
        p(z) ~ (z / z0)^2 exp[-1.5 (z / z0)]

    dmu_0, dmu_1 : float
        specify the error in mu, dmu = dmu_0 + dmu_1 * mu

    random_state : None, int, or np.random.RandomState instance
        random seed or random number generator

    cosmo : astropy.cosmology instance specifying cosmology
        to use when generating the sample.  If not provided,
        a Flat Lambda CDM model with H0=71, Om0=0.27, Tcmb=0 is used.

    Returns
    -------
    z, mu, dmu : ndarrays
        arrays of shape ``size``
    """

    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=71, Om0=0.27, Tcmb0=0)

    random_state = check_random_state(random_state)
    zdist = FunctionDistribution(redshift_distribution, func_args=dict(z0=z0),
                                 xmin=0.1 * z0, xmax=10 * z0,
                                 random_state=random_state)

    z_sample = zdist.rvs(size)
    mu_sample = cosmo.distmod(z_sample).value

    dmu = dmu_0 + dmu_1 * mu_sample
    mu_sample = random_state.normal(mu_sample, dmu)

    return z_sample, mu_sample, dmu

import numpy as np

from sklearn.utils import check_random_state


##
# Taken from
#    https://www.astroml.org/_modules/astroML/density_estimation/empirical.html#FunctionDistribution
##
class FunctionDistribution:
    """Generate random variables distributed according to an arbitrary function

    Parameters
    ----------
    func : function
        func should take an array of x values, and return an array
        proportional to the probability density at each value
    xmin : float
        minimum value of interest
    xmax : float
        maximum value of interest
    Nx : int (optional)
        number of samples to draw.  Default is 1000
    random_state : None, int, or np.random.RandomState instance
        random seed or random number generator
    func_args : dictionary (optional)
        additional keyword arguments to be passed to func
    """


    def __init__(self, func, xmin, xmax, Nx=1000,
                 random_state=None, func_args=None):
        self.random_state = check_random_state(random_state)

        if func_args is None:
            func_args = {}

        x = np.linspace(xmin, xmax, Nx)
        Px = func(x, **func_args)

        # if there are too many zeros, interpolation will fail
        positive = (Px > 1E-10 * Px.max())
        x = x[positive]
        Px = Px[positive].cumsum()
        Px /= Px[-1]

        self._tck = interpolate.splrep(Px, x)


    def rvs(self, shape):
        """Draw random variables from the distribution

        Parameters
        ----------
        shape : integer or tuple
            shape of desired array

        Returns
        -------
        rv : ndarray, shape=shape
            random variables
        """
        # generate uniform variables between 0 and 1
        y = self.random_state.random_sample(shape)
        return interpolate.splev(y, self._tck)

