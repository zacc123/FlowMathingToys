Sorry Mathew!!!


Run it like:

python train_posterior_toy.py --out-dir <what ever you want>  \
                              --run-name <whatever dir you want inside out-dir> \
                              --mode <ecrf or rf> \ 
                              --coupling <indendent or mbot or nn> 

### A couple notes:
if you want to just rerun the posterior sampling, do it with --skip-train-prior and --resume-checkpoint <path to your weights>
                      
