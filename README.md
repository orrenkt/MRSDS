
# MR-SDS

Code for the MRSDS model \
Modeling state-dependent communication between brain regions with switching nonlinear dynamical systems \
https://openreview.net/forum?id=WQwV7Y8qwa

To install MRSDS and its requirements: \
1 - clone this repo and run `bash create_conda_env.sh` to set up the enviornment and requirements \
    Note that the cudatoolkit and cudnn versions will need to be appropriate for your system \
    This shell script creates a conda enviornment `tfp-gpu` which you'll need to activate to run mrsds.

To run MRSDS on your data:

1 - Implement a simple function for loading your data in utils_data.py \
    -see `load_mydata()` example function, you'll need to return: \
    ys: list of arrays or array of size `[trials, time, neurons]` \
    us: list of arrays or array of size `[trials, time, inputs]` \
    region_sizes: list with number of neurons per region in ys, eg `[num, num, num]` \
    num_regions: number of regions, eg `len(region_sizes)` 

2 - Add your data source name to `load_prep_data()` in `utils_data.py`. This name will also go in your config.

3 - Create a config file in the `run-configs` folder which specifies the dataset, model and training hyperparameters. 
    See two included example configs for you to modify: \
    `toy-config.yaml` : good starting point for running on a small simulation \
    `large-config.yaml` : for a larger dataset with neural data.

4 - Call `run_training.py`, modifying the command line args to point to your directories. \
    Note the script output will stream out a run log to a text file which should have the same output directory \
    and name as your run (note the `-u` arg makes stdout unbuffered). See argparsing at the end of \
    `run_training.py` for more argument details. You'll need to provide: \
    -the gpu id to run on \
    -the number of discrete states (k) and latent continuous dimensions per region (d) \
    -model inference type: svae (structured), s-svae (switching structured), or mrsds (mean field). \
     For simplicity it's best to start with svae (k=1). For switching dynamics (k > 1) use s-svae. \
     mrsds (mean field) can handle both and works better in some cases, but may be less good for \
     generation vs inference (eg cosmoothing). \
    -random seeds for the data loader, model initialization, and model training for reproducability.\
     it's a good idea to add these seeds to the run name you specify but you don't have to. 

```
python -u run_training.py -datapath /mydata -resultpath /myresults -configpath run-configs/my_config.yaml
-name myrun-seeds0-0-0 -gpu 0 -k 1 -d 2 -mtype svae -dataseed 0 -modelseed 0 -trainseed 0
> /myresults/myrun-seeds0-0-0-log.txt
```

   the script will create the following directories and files in your resultpath: \
   `myrun-log.txt` : the run log text file specified above. \
   `myrun-logs`    : directory with tensorflow logfile \
   `myrun-models`  : directory with tf model checkpoints needed to load trained model, see eg notebooks below \
   `myrun-models-do` : only created when using mean field (mrsds) inference, not svae or s-svae. \
    -By default after regular training, the model is trained with the inference network frozen to ensure good dynamics generation.
    These later checkpoints are put in a second directory. See config examples for how this second training stage is specified. \
   `myrun_latents.mat` : a mat file with a dictionary of saved arrays. You can set how often this file is created during training 
   in your config file. A few important keys: \
    `xs_train`, `xs_test`: samples from inferred approximate posterior over latents xs \
    `zs_logprob`, `zs_logprob_test`: log posterior over discrete states z, when k > 1 \
    `ys_recon_train`, `ys_recon_test`: reconstructions under the model.
    saved r2, mse, cosmooth etc.

5 - use a notebook to load and visualize the results.
 
  see example notebook. Some things you can do: \
  -specify a run log file, eg myrun-log.txt, to directly load: \
   its associated config.yaml \
   saved latents.mat file \
   the saved tf model \
  -get and visualize the gradient field (dynamics) of the learned model for each region (only for 2d latents) \
  -run inference on a batch of samples \
  -get inferred messages and message norms under the model \
  -visualize messages \

