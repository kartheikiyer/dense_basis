Features
========

- GP-SFH module: A module that can convert SFH-tuples to smooth curves in SFR vs time space, and vice-versa. This module can also be used to create GP-approximations for any input SFH curve (e.g., SFHs from simulations). 

- Prior atlas generator - A set of functions that can generate SEDs corresponding to input stellar population parameters using the FSPS backend. Taken in conjunction with the db.Priors() class, this can be used to sample prior distributions and trade space for time complexity while fitting SEDs.

- SED fitter module - Functions for evaluating the goodness-of-fit given an observed SED with uncertainties, and plotting posterior distributions in parameter- and SFH-space.

