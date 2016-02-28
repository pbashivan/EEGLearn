Data for Visual WM experiment described in:
Bashivan, Pouya, et al. "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).

FeatureMat_timeWin:
FFT power values extracted for three frequency bands (theta, alpha, beta). Features are arranged in band and electrodes order (theta_1, theta_2..., theta_64, alpha_1, alpha_2, ..., beta_64). Last column contains the class labels (load levels).

Neuroscan_locs_orig:
3 dimensional coordinates for electrodes on Neuroscan quik-cap.

trials_subNums:
contains subject numbers associated with each trial (used for leave-subject-out cross validation).