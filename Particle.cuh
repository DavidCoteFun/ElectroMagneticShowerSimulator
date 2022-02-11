#ifndef PARTICLE_CUH
#define PARTICLE_CUH

#include "Common.cuh"
class Master;

//Energy unit: MeV
//Distance unit: micrometer

class Particle {
public:
    CUDA_HOST Particle();
    CUDA_HOST Particle(float energy, float pos_x, float pos_y, float dir_x, float dir_y);
    virtual ~Particle() {}
    CUDA_HOST virtual void simulate_until_decay(Master*)=0;
    float m_x;
    float m_y;
    float m_e;
    float m_dx;
    float m_dy;
    Particle* next;
    Particle* prev;
};

#endif
