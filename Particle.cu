#include "Particle.cuh"

CUDA_HOST Particle::Particle(){
  m_e=0;
  m_x=0;
  m_y=0;
  m_dx=1.0;
  m_dy=0;
}

CUDA_HOST Particle::Particle(float energy, float pos_x, float pos_y, float dir_x, float dir_y){
  m_e=energy;
  m_x=pos_x;
  m_y=pos_y;
  m_dx=dir_x;
  m_dy=dir_y;
}
