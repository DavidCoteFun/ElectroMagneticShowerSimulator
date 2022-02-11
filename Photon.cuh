#include "Common.cuh"
#include "Particle.cuh"

class Master;

//Energy unit: MeV
//Distance unit: micrometer

class Photon : public Particle {
public:
    CUDA_HOST Photon();
    CUDA_HOST Photon(float energy, float pos_x, float pos_y, float dir_x, float dir_y);
    ~Photon() {}
    CUDA_HOST virtual void simulate_until_decay(Master* mgr);	
};
