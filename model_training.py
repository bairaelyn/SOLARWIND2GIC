#!/usr/bin/env python
'''
--------------------------------------------------------------------
model_training.py
Script calling solarwind2gic functions to set up, train and test a
machine learning model for predicting geomagnetically induced
currents (GICs) from solar wind data measured at the L1 point.
 
Tips on bettter model training:
 - https://machinelearningmastery.com/improve-deep-learning-performance/

Created 2017-09-14 by R Bailey, IWF Helio Group (Graz) / ZAMG Conrad Observatory (Vienna).
Last updated April 2019.
--------------------------------------------------------------------
'''

import os
import sys
import getopt

# Special imports
from solarwind2gic import *
from model_training_params import param_dict


# *******************************************************************
#                           MAIN SCRIPT
# *******************************************************************

if __name__ == '__main__':

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,"hvpd:",["findperiods", "trainmodel", "corrwithgic",
                                             "loadmodel", "help", "verbose", "plot",
                                             "traintest", "predstorm", "featuretest"])
    
    find_periods, train_model, load_model, correlate_RMS_with_GIC = False, False, False, False
    verbose, plot_output, get_input, use_predstorm = False, False, True, False
    run_param_tests, test_feature_importance = False, False
    for opt, arg in opts:
        if opt == '-h' or opt == '--help': # Print help for those in need:
            print("")
            print("-----------------------------------------------------------------")
            print("DESCRIPTION:")
            print("Uses keras machine learning algorithms to predict GIC from")
            print("solar wind data at the L1 point. Takes DSCOVR and ACE data and")
            print("can also take PREDSTORM L1 prediction output.")
            print("-------------------------------------")
            print("RUN OPTIONS:")
            print("--findperiods : (step 1) Go through data and pick out periods")
            print("                to train data on. Requires date input.")
            print("                python model_training.py --findperiods -d 1999-01-01--2004-01-01")
            print("--trainmodel  : (step 2) Train model using periods found in")
            print("                step 1.")
            print("                python model_training.py --trainmodel")
            print("--corrwithgic : (step 3) Correlate RMS(dX) and RMS(dY) with RMS(GIC)")
            print("                python model_training.py --corrwithgic")
            print("--loadmodel   : (step 4) Load formerly-trained model from step")
            print("                2 to predict from new data.")
            print("                python model_training.py --loadmodel")
            print("                python model_training.py --loadmodel --predstorm")
            print("GENERAL OPTIONS:")
            print("-h/--help     : print this help data")
            print("-p/--plot     : plot plots")
            print("-v/--verbose  : print runtime info for debugging")
            print("-d            : Provide dates for steps 1/2 in format starttime")
            print("                -endtime, e.g. 1999-01-01--2004-01-01.")
            print("--noinput     : run without user input")
            print("--traintest   : run training (--trainmodel) for parameter testing")
            print("--featuretest : run training (--trainmodel) for feature importance testing")
            print("--predstorm   : use with --loadmodel to use PREDSTORM data")
            print("-------------------------------------")
            print("EXAMPLE USAGE:")
            print("  See model output for most recent DSCOVR data or old data:")
            print("    python model_training.py --loadmodel")
            print("  Train model (training parameters set in code):")
            print("    python model_training.py --trainmodel")
            print("  ... with every step printed and plotted:")
            print("    python model_training.py --trainmodel -v -p")
            print("  Test model training with various parameters (set in code):")
            print("    python model_training.py --trainmodel --traintest")
            print("-----------------------------------------------------------------")
            print("")
            sys.exit()
        elif opt == '-v' or opt == '--verbose': # Print runtime information for debugging
            verbose = True
        elif opt == '-p' or opt == '--plot': # Plot some steps
            plot_output = True
        elif opt == "--findperiods":
            find_periods = True
        elif opt == "--trainmodel":
            train_model = True
        elif opt == "--corrwithgic":
            correlate_RMS_with_GIC = True
        elif opt == "--loadmodel":
            load_model = True
        elif opt == "--noinput":
            get_input = False
        elif opt == "--traintest":
            run_param_tests = True
        elif opt == "--featuretest":
            test_feature_importance = True
        elif opt == "--predstorm":
            use_predstorm = True
        elif opt == '-d':
            dates = arg.split("--")
            starttime = datetime.strptime(dates[0]+" 00:00:00", "%Y-%m-%d %H:%M:%S")
            endtime =   datetime.strptime(dates[1]+" 00:00:00", "%Y-%m-%d %H:%M:%S")
        else:
            print("Try this:")
            print("    python model_training.py --help")
            sys.exit()
            
    if not verbose:  
        verboseprint = lambda *a: None      # Do nothing

    print(param_dict)

    # *******************************************************************
    # 1. Preemptive (do once): find ranges for data set
    # *******************************************************************
    
    # $ python model_training.py --findperiods -d 1999-01-01--2004-01-01
    if find_periods:
        print("Finds periods interesting for training.")
    
    # *******************************************************************
    # 2. Read data and train machine learning model:
    # *******************************************************************
    
    # $ python model_training.py --trainmodel --verbose --plot
    if train_model:
        print("Runs training on model.")
    
    # *******************************************************************
    # FINAL. Load model from former training
    # *******************************************************************
    
    # $ python model_training.py --loadmodel
    if load_model:
        print("Loads and runs prediction model on test input data.")




