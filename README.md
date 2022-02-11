# ElectroMagneticShowerSimulator

Embryo of prototype to study GPU acceleration techniques for Electro-Magnetic Shower simulations in particle physics. <br>
Started in 2014 but never completed...

## To setup CUDA
```
export PATH=/usr/local/cuda-6.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-6.0/lib64:$LD_LIBRARY_PATH
```

## To compile and run
```
nvcc test.cu -o test 
./test

nvcc hello_world.cu -o hello_world
./hello_world

nvcc -DDEBUG debug_errors.cu -o debug_errors
./debug_errors

export DEBUG=''
export DEBUG='-DDEBUG'
nvcc $DEBUG -arch=compute_11 -code=sm_11,compute_11 -c Queue.cu -o lib/Queue.o
nvcc $DEBUG -arch=compute_11 -code=sm_11,compute_11 -c Master.cu -o lib/Master.o
nvcc $DEBUG -arch=compute_11 -code=sm_11,compute_11 -c Particle.cu -o lib/Particle.o
nvcc $DEBUG -arch=compute_11 -code=sm_11,compute_11 -c Electron.cu -o lib/Electron.o
nvcc $DEBUG -arch=compute_11 -code=sm_11,compute_11 -c Photon.cu -o lib/Photon.o
nvcc $DEBUG -arch=compute_11 -code=sm_11,compute_11 lib/Master.o lib/Particle.o lib/Photon.o lib/Electron.o lib/Queue.o shower_simulator.cu -o bin/shower_simulator_cuda.exe
```

## Note
The optimal number of threads per block is almost always a multiple of 32, and at least 64, 
because of how the thread scheduling hardware works. A good choice for a first attempt is 128 or 256.
Then adjust numbers of blocks to map the size of problem, such that `n_blocks*n_threads=full_problem`.


## Useful link
https://code.google.com/p/stanford-cs193g-sp2010/wiki/TutorialPrerequisites

