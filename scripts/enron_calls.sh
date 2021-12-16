## Enron list of commands

###################
### MLE for tau ###
###################

## No interactions
./enron.py -m -pm -f '../results/enron_results/tau_mle/m_pm' &
./enron.py -m -f '../results/enron_results/tau_mle/m_wm' &
./enron.py -m -hm -f '../results/enron_results/tau_mle/m_hm' &

## No main effects

./enron.py -i -pi -f '../results/enron_results/tau_mle/i_pi' &
./enron.py -i -pi -d 5 -f '../results/enron_results/tau_mle/i_pi_5' &
./enron.py -i -pi -d 10 -f '../results/enron_results/tau_mle/i_pi_10' &

./enron.py -i -f '../results/enron_results/tau_mle/i_wi' &
./enron.py -i -d 5 -f '../results/enron_results/tau_mle/i_wi_5' &
./enron.py -i -d 10 -f '../results/enron_results/tau_mle/i_wi_10' &

./enron.py -i -hi -f '../results/enron_results/tau_mle/i_hi' &
./enron.py -i -hi -d 5 -f '../results/enron_results/tau_mle/i_hi_5' &
./enron.py -i -hi -d 10 -f '../results/enron_results/tau_mle/i_hi_10' &

## Poisson process interactions

./enron.py -m -pm -i -pi -f '../results/enron_results/tau_mle/mi_pm_pi' &
./enron.py -m -i -pi -f '../results/enron_results/tau_mle/mi_wm_pi' &
./enron.py -m -hm -i -pi -f '../results/enron_results/tau_mle/mi_hm_pi' &

./enron.py -m -pm -i -pi -d 5 -f '../results/enron_results/tau_mle/mi_pm_pi_5' &
./enron.py -m -i -pi -d 5 -f '../results/enron_results/tau_mle/mi_wm_pi_5' &
./enron.py -m -hm -i -pi -d 5 -f '../results/enron_results/tau_mle/mi_hm_pi_5' &

./enron.py -m -pm -i -pi -d 10 -f '../results/enron_results/tau_mle/mi_pm_pi_10' &
./enron.py -m -i -pi -d 10 -f '../results/enron_results/tau_mle/mi_wm_pi_10' &
./enron.py -m -hm -i -pi -d 10 -f '../results/enron_results/tau_mle/mi_hm_pi_10' &

## Markov process interactions

./enron.py -m -pm -i -f '../results/enron_results/tau_mle/mi_pm_wi' &
./enron.py -m -i -f '../results/enron_results/tau_mle/mi_wm_wi' &
./enron.py -m -hm -i -f '../results/enron_results/tau_mle/mi_hm_wi' &

./enron.py -m -pm -i -d 5 -f '../results/enron_results/tau_mle/mi_pm_wi_5' &
./enron.py -m -i -d 5 -f '../results/enron_results/tau_mle/mi_wm_wi_5' &
./enron.py -m -hm -i -d 5 -f '../results/enron_results/tau_mle/mi_hm_wi_5' &

./enron.py -m -pm -i -d 10 -f '../results/enron_results/tau_mle/mi_pm_wi_10' &
./enron.py -m -i -d 10 -f '../results/enron_results/tau_mle/mi_wm_wi_10' &
./enron.py -m -hm -i -d 10 -f '../results/enron_results/tau_mle/mi_hm_wi_10' &

## Hawkes process interactions

./enron.py -m -pm -i -hi -f '../results/enron_results/tau_mle/mi_pm_hi' &
./enron.py -m -i -hi -f '../results/enron_results/tau_mle/mi_wm_hi' &
./enron.py -m -hm -i -hi -f '../results/enron_results/tau_mle/mi_hm_hi' &

./enron.py -m -pm -i -hi -d 5 -f '../results/enron_results/tau_mle/mi_pm_hi_5' &
./enron.py -m -i -hi -d 5 -f '../results/enron_results/tau_mle/mi_wm_hi_5' &
./enron.py -m -hm -i -hi -d 5 -f '../results/enron_results/tau_mle/mi_hm_hi_5' &

./enron.py -m -pm -i -hi -d 10 -f '../results/enron_results/tau_mle/mi_pm_hi_10' &
./enron.py -m -i -hi -d 10 -f '../results/enron_results/tau_mle/mi_wm_hi_10' &
./enron.py -m -hm -i -hi -d 10 -f '../results/enron_results/tau_mle/mi_hm_hi_10' &

################################################
### tau set to training set adjacency matrix ###
################################################

## No interactions
./enron.py -m -pm -z -f '../results/enron_results/tau_Aij/m_pm' &
./enron.py -m -z -f '../results/enron_results/tau_Aij/m_wm' &
./enron.py -m -hm -z -f '../results/enron_results/tau_Aij/m_hm' &

## No main effects

./enron.py -i -pi -z -f '../results/enron_results/tau_Aij/i_pi' &
./enron.py -i -pi -d 5 -z -f '../results/enron_results/tau_Aij/i_pi_5' &
./enron.py -i -pi -d 10 -z -f '../results/enron_results/tau_Aij/i_pi_10' &

./enron.py -i -z -f '../results/enron_results/tau_Aij/i_wi' &
./enron.py -i -d 5 -z -f '../results/enron_results/tau_Aij/i_wi_5' &
./enron.py -i -d 10 -z -f '../results/enron_results/tau_Aij/i_wi_10' &

./enron.py -i -hi -z -f '../results/enron_results/tau_Aij/i_hi' &
./enron.py -i -hi -d 5 -z -f '../results/enron_results/tau_Aij/i_hi_5' &
./enron.py -i -hi -d 10 -z -f '../results/enron_results/tau_Aij/i_hi_10' &

## Poisson process interactions

./enron.py -m -pm -i -pi -z -f '../results/enron_results/tau_Aij/mi_pm_pi' &
./enron.py -m -i -pi -z -f '../results/enron_results/tau_Aij/mi_wm_pi' &
./enron.py -m -hm -i -pi -z -f '../results/enron_results/tau_Aij/mi_hm_pi' &

./enron.py -m -pm -i -pi -d 5 -z -f '../results/enron_results/tau_Aij/mi_pm_pi_5' &
./enron.py -m -i -pi -d 5 -z -f '../results/enron_results/tau_Aij/mi_wm_pi_5' &
./enron.py -m -hm -i -pi -d 5 -z -f '../results/enron_results/tau_Aij/mi_hm_pi_5' &

./enron.py -m -pm -i -pi -d 10 -z -f '../results/enron_results/tau_Aij/mi_pm_pi_10' &
./enron.py -m -i -pi -d 10 -z -f '../results/enron_results/tau_Aij/mi_wm_pi_10' &
./enron.py -m -hm -i -pi -d 10 -z -f '../results/enron_results/tau_Aij/mi_hm_pi_10' &

## Markov process interactions

./enron.py -m -pm -i -z -f '../results/enron_results/tau_Aij/mi_pm_wi' &
./enron.py -m -i -z -f '../results/enron_results/tau_Aij/mi_wm_wi' &
./enron.py -m -hm -i -z -f '../results/enron_results/tau_Aij/mi_hm_wi' &

./enron.py -m -pm -i -d 5 -z -f '../results/enron_results/tau_Aij/mi_pm_wi_5' &
./enron.py -m -i -d 5 -z -f '../results/enron_results/tau_Aij/mi_wm_wi_5' &
./enron.py -m -hm -i -d 5 -z -f '../results/enron_results/tau_Aij/mi_hm_wi_5' &

./enron.py -m -pm -i -d 10 -z -f '../results/enron_results/tau_Aij/mi_pm_wi_10' &
./enron.py -m -i -d 10 -z -f '../results/enron_results/tau_Aij/mi_wm_wi_10' &
./enron.py -m -hm -i -d 10 -z -f '../results/enron_results/tau_Aij/mi_hm_wi_10' &

## Hawkes process interactions

./enron.py -m -pm -i -hi -z -f '../results/enron_results/tau_Aij/mi_pm_hi' &
./enron.py -m -i -hi -z -f '../results/enron_results/tau_Aij/mi_wm_hi' &
./enron.py -m -hm -i -hi -z -f '../results/enron_results/tau_Aij/mi_hm_hi' &

./enron.py -m -pm -i -hi -d 5 -z -f '../results/enron_results/tau_Aij/mi_pm_hi_5' &
./enron.py -m -i -hi -d 5 -z -f '../results/enron_results/tau_Aij/mi_wm_hi_5' &
./enron.py -m -hm -i -hi -d 5 -z -f '../results/enron_results/tau_Aij/mi_hm_hi_5' &

./enron.py -m -pm -i -hi -d 10 -z -f '../results/enron_results/tau_Aij/mi_pm_hi_10' &
./enron.py -m -i -hi -d 10 -z -f '../results/enron_results/tau_Aij/mi_wm_hi_10' &
./enron.py -m -hm -i -hi -d 10 -z -f '../results/enron_results/tau_Aij/mi_hm_hi_10' &

###############################
### tau set to 0 everywhere ###
###############################

## No interactions
./enron.py -m -pm -z -fl -f '../results/enron_results/tau_zero/m_pm' &
./enron.py -m -z -fl -f '../results/enron_results/tau_zero/m_wm' &
./enron.py -m -hm -z -fl -f '../results/enron_results/tau_zero/m_hm' &

## No main effects

./enron.py -i -pi -z -fl -f '../results/enron_results/tau_zero/i_pi' &
./enron.py -i -pi -d 5 -z -fl -f '../results/enron_results/tau_zero/i_pi_5' &
./enron.py -i -pi -d 10 -z -fl -f '../results/enron_results/tau_zero/i_pi_10' &

./enron.py -i -z -fl -f '../results/enron_results/tau_zero/i_wi' &
./enron.py -i -d 5 -z -fl -f '../results/enron_results/tau_zero/i_wi_5' &
./enron.py -i -d 10 -z -fl -f '../results/enron_results/tau_zero/i_wi_10' &

./enron.py -i -hi -z -fl -f '../results/enron_results/tau_zero/i_hi' &
./enron.py -i -hi -d 5 -z -fl -f '../results/enron_results/tau_zero/i_hi_5' &
./enron.py -i -hi -d 10 -z -fl -f '../results/enron_results/tau_zero/i_hi_10' &

## Poisson process interactions

./enron.py -m -pm -i -pi -z -fl -f '../results/enron_results/tau_zero/mi_pm_pi' &
./enron.py -m -i -pi -z -fl -f '../results/enron_results/tau_zero/mi_wm_pi' &
./enron.py -m -hm -i -pi -z -fl -f '../results/enron_results/tau_zero/mi_hm_pi' &

./enron.py -m -pm -i -pi -d 5 -z -fl -f '../results/enron_results/tau_zero/mi_pm_pi_5' &
./enron.py -m -i -pi -d 5 -z -fl -f '../results/enron_results/tau_zero/mi_wm_pi_5' &
./enron.py -m -hm -i -pi -d 5 -z -fl -f '../results/enron_results/tau_zero/mi_hm_pi_5' &

./enron.py -m -pm -i -pi -d 10 -z -fl -f '../results/enron_results/tau_zero/mi_pm_pi_10' &
./enron.py -m -i -pi -d 10 -z -fl -f '../results/enron_results/tau_zero/mi_wm_pi_10' &
./enron.py -m -hm -i -pi -d 10 -z -fl -f '../results/enron_results/tau_zero/mi_hm_pi_10' &

## Markov process interactions

./enron.py -m -pm -i -z -fl -f '../results/enron_results/tau_zero/mi_pm_wi' &
./enron.py -m -i -z -fl -f '../results/enron_results/tau_zero/mi_wm_wi' &
./enron.py -m -hm -i -z -fl -f '../results/enron_results/tau_zero/mi_hm_wi' &

./enron.py -m -pm -i -d 5 -z -fl -f '../results/enron_results/tau_zero/mi_pm_wi_5' &
./enron.py -m -i -d 5 -z -fl -f '../results/enron_results/tau_zero/mi_wm_wi_5' &
./enron.py -m -hm -i -d 5 -z -fl -f '../results/enron_results/tau_zero/mi_hm_wi_5' &

./enron.py -m -pm -i -d 10 -z -fl -f '../results/enron_results/tau_zero/mi_pm_wi_10' &
./enron.py -m -i -d 10 -z -fl -f '../results/enron_results/tau_zero/mi_wm_wi_10' &
./enron.py -m -hm -i -d 10 -z -fl -f '../results/enron_results/tau_zero/mi_hm_wi_10' &

## Hawkes process interactions

./enron.py -m -pm -i -hi -z -fl -f '../results/enron_results/tau_zero/mi_pm_hi' &
./enron.py -m -i -hi -z -fl -f '../results/enron_results/tau_zero/mi_wm_hi' &
./enron.py -m -hm -i -hi -z -fl -f '../results/enron_results/tau_zero/mi_hm_hi' &

./enron.py -m -pm -i -hi -d 5 -z -fl -f '../results/enron_results/tau_zero/mi_pm_hi_5' &
./enron.py -m -i -hi -d 5 -z -fl -f '../results/enron_results/tau_zero/mi_wm_hi_5' &
./enron.py -m -hm -i -hi -d 5 -z -fl -f '../results/enron_results/tau_zero/mi_hm_hi_5' &

./enron.py -m -pm -i -hi -d 10 -z -fl -f '../results/enron_results/tau_zero/mi_pm_hi_10' &
./enron.py -m -i -hi -d 10 -z -fl -f '../results/enron_results/tau_zero/mi_wm_hi_10' &
./enron.py -m -hm -i -hi -d 10 -z -fl -f '../results/enron_results/tau_zero/mi_hm_hi_10' &