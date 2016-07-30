# Various-Boltzmann
Variation of Boltzmann Machines implemented in Matlab.<br><br>

All the codes is based on "https://github.com/rasmusbergpalm/DeepLearnToolbox.git".<br><br>

These codes has implemented a RBM's training mathods in terms of the paper "Training Restricted Boltzmann Machines via the Thouless-Anderson-Palmer Free Energy" by matlab.In the course of implementation,I also reference to the Julia language version codes "https://github.com/sphinxteam/Boltzmann.jl". We call this method "tap2"(second order) and "tap3"(third order).<br><br>

These codes has implemented a RBM's training mathods in terms of the paper "Modeling image patches with a directed hierarchy of Markov random fields".We call these method  "semi".
Implementing the Matrix RBM in terms of the paper"Matrix Variate RBM and Its Applications".<br><br>

Usage:<br>
    Enter the test directory,and edit the test_example_DBN.m.<br>
    opts.approx="semi":use semi method to do sampling.<br>
    opts.approx="tap2":use tap2 method to do sampling.<br>
    opts.approx="tap3":use tap3 method to do sampling<br>
    opts.approx="CD":use CD method to do sampling.<br>
    
    dbn.sizes = [100];
    opts.numepochs =   10;
    opts.batchsize = 100;
    opts.momentum  =   0.5;
    opts.alpha     =   0.005;
    opts.approx = 'semi'
    opts.regularize=0.01
    opts.weight_decay='l1'
    opts.iterations=1
