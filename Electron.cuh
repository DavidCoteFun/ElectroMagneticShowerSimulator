#include "Common.cuh"
#include "Particle.cuh"

class Master;

//Energy unit: MeV
//Distance unit: micrometer

class Electron : public Particle {
public:
    CUDA_HOST Electron();
    CUDA_HOST Electron(float energy, float pos_x, float pos_y, float dir_x, float dir_y);
    ~Electron() {}
    CUDA_HOST virtual void simulate_until_decay(Master* mgr);	
};
