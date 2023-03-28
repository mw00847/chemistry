

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv


#define least square formula
def least_sq(sample_spectrum, components):
   # sample_spectrum (unknown spectrum): array of w values.
   # components (known spectra): array of n (number of components) columns with w values.
   # This def returns an array of n values. Each value is the similarity score for the sample_spectrum and a component spectrum.
   similarity = np.dot(inv(np.dot(components, components.T)) , np.dot(components, sample_spectrum))
   return similarity


# What concentrations we want these components to have in our mixture:
c_a = 0.5
c_b = 0.3
c_c = 0.2


