#include "Photon.cuh"
#include "Master.cuh"
#include <cmath>

CUDA_HOST Photon::Photon() : Particle() {
}

CUDA_HOST Photon::Photon(float energy, float pos_x, float pos_y, float dir_x, float dir_y) : Particle(energy,pos_x,pos_y,dir_x,dir_y) {
}

CUDA_HOST void Photon::simulate_until_decay(Master* mgr){
 //mimic creation of e+e- that would deviate in magnetic field
 //make it computationally expensive on purpose...

 //e+
 float angle1 = atan(m_dy/m_dx);
 float angle2 = 3.1415926535897931 / 180.0;  //1 degree
 float hyp = sqrt(m_dx*m_dx + m_dy*m_dy);
 float angle3 = angle1+angle2;
 float outDy = hyp*sin(angle3);
 float outDx = hyp*cos(angle3);
 mgr->add_electron(m_e/2.0,m_x+m_dx,m_y+m_dy,outDx,outDy);

 //e-
 angle1 = atan(m_dy/m_dx);
 angle2 = 3.1415926535897931 / 180.0;  //1 degree
 hyp = sqrt(m_dx*m_dx + m_dy*m_dy);
 angle3 = angle1-angle2;
 outDy = hyp*sin(angle3);
 outDx = hyp*cos(angle3);
 mgr->add_electron(m_e/2.0,m_x+m_dx,m_y+m_dy,outDx,outDy);

 return;
}

