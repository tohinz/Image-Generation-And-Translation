# Image-Generation-And-Translation
Implementation and additional results of ["Image Generation and Translation with Disentangled Representations"](https://arxiv.org/abs/1803.10567).

The work presented here was done at the [Knowledge Technology Research Group](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/ "Knowledge Technology Research Group") at the University of Hamburg and accepted as a conference contribution to the [IJCNN 2018](http://www.ecomp.poli.br/~wcci2018/).

## To Run the Experiment on MNIST Data Set
Requirements:
* Tensorflow 1.5.0
* Numpy 1.14.1

Run Experiment:
* run `python code/train_mnist.py`
Results are stored in code/log_dir/mnist_sem_sup
* to visualize results: `tensorboard --logdir code/log_dir/mnist_sem_sup`
* for each experiment run log_dir also contains a file info.txt detailing the generator and encoder accuracy after X iterations, as well as other information about the training progress

To evaluate the results:
* run `python evaluate_model.py --model_dir dir-where-model-weights-are-stored` with one of the following flags:
    * `--evaluate`: evaluates the encodings' accuracy on the MNIST test set. Needs a file "encodings.txt" (stored in the same dir as --model_dir) which gives the mappings from the learned encodings to the actual labels. Use for example the images from the directory samples_disc to find the right mappings. E.g. for the following image the encodings.txt file would look as following: `6,5,9,2,1,8,3,0,4,7`, since the sixth row encodes zeros, the fifth row encodes ones, the last (ninth) row encodes twos, etc
![Example Image of Encodings](code/example_encoding/example.png)
    * `--sample`: samples images from the test set according to their values in the categorical and continuous variables of z. Samples are stored in model_dir/samples (samples_categorical.png, sample_cont_c1.png, sample_cont_c2.png)
    * `--reconstruct`: samples twenty images X of each class from the test set and reconstructs them, i.e. G(E(X)). Result is stored in model_dir/samples/samples_reconstruction.png, where the uneven columns (first, third, ...) depict the original images and the even columns depict the respective reconstructions
    * `--generate`: generates new samples according to the different categorical values. Results are stored in model_dir/samples/generated_imgs_categorical.png
