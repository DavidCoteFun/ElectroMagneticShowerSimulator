#include "Electron.cuh"
#include "Master.cuh"

CUDA_HOST Electron::Electron() : Particle() {}

CUDA_HOST Electron::Electron(float energy, float pos_x, float pos_y, float dir_x, float dir_y) : Particle(energy,pos_x,pos_y,dir_x,dir_y) {}

CUDA_HOST void Electron::simulate_until_decay(Master* mgr){
  //mimic radiation of a photon with 1% of electron's energy
  //make it computationally expensive on purpose...

  float angle1 = atan(m_dy/m_dx);
  float angle2 = 3.1415926535897931 / 180.0;  //1 degree
  float hyp = sqrt(m_dx*m_dx + m_dy*m_dy);
  float angle3 = angle1+angle2;
  float outDy = hyp*sin(angle3);
  float outDx = hyp*cos(angle3);
  float outX = m_x+outDx;
  float outY = m_y+outDy;

  if(m_e > mgr->m_threshold){
    mgr->add_photon(m_e*0.01,outX,outY,outDx,outDy);
    mgr->add_electron(m_e*0.99,m_x,m_y,m_dx,m_dy);
  }
  return;
}
