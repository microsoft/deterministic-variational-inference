import numpy as np
import matplotlib.pyplot as plt

def toy_results_plot(data, data_generator, hypers=None, predictions=None):   
    train_x = np.arange(np.min(data[0][0].reshape(-1)),
                        np.max(data[0][0].reshape(-1)), 1/100)
    
    # plot the training data distribution
    plt.plot(train_x, data_generator['mean'](train_x), 'red', label='data mean')
    plt.fill_between(train_x,
                     data_generator['mean'](train_x) - data_generator['std'](train_x),
                     data_generator['mean'](train_x) + data_generator['std'](train_x),
                     color='orange', alpha=1, label='data 1-std')
    plt.plot(data[0][0], data[0][1], 'r.', alpha=0.2, label='train sampl')
     
    # plot the model distribution
    if predictions is not None:
        x = predictions[0]
        y_mean   = predictions[1]['mean'][:,0]
        ell_mean = predictions[1]['mean'][:,1]
        y_var    = predictions[1]['cov'][:,0,0]
        ell_var  = predictions[1]['cov'][:,1,1]
        
        if hypers['style'] != 'heteroskedastic':
            ell_mean = hypers["homo_logvar_scale"]
            ell_var = 0

        heteroskedastic_part = np.exp(0.5 * ell_mean)
        full_std = np.sqrt(y_var + np.exp(ell_mean + 0.5 * ell_var))

        plt.plot(x, y_mean, label='model mean')
        plt.fill_between(x,
                         y_mean - heteroskedastic_part,
                         y_mean + heteroskedastic_part,
                         color='g', alpha = 0.2, label='$\ell$ contrib')
        plt.fill_between(x,
                         y_mean - full_std,
                         y_mean + full_std,
                         color='b', alpha = 0.2, label='model 1-std')
        
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim([-3,2])
    plt.legend()