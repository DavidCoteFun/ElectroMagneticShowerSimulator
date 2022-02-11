#include "Master.cuh"
#include "Queue.cuh"
#include "Photon.cuh"
#include "Electron.cuh"
#include <iostream>

using namespace std;

__global__ void simulate_electron(float *input, float *output){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  float m_e=input[5*index];
  float m_x=input[5*index+1];
  float m_y=input[5*index+2];
  float m_dx=input[5*index+3];
  float m_dy=input[5*index+4];

  float angle1 = atan(m_dy/m_dx);
  float angle2 = 3.1415926535897931 / 180.0;  //1 degree
  float hyp = sqrt(m_dx*m_dx + m_dy*m_dy);
  float angle3 = angle1+angle2;
  float outDy = hyp*sin(angle3);
  float outDx = hyp*cos(angle3);
  float outX = m_x+outDx;
  float outY = m_y+outDy;

  if(m_e > 0.001){
    output[10*index]=m_e*0.01;
    output[10*index+1]=outX;
    output[10*index+2]=outY;
    output[10*index+3]=outDx;
    output[10*index+4]=outDy;
    output[10*index+5]=m_e*0.99;
    output[10*index+6]=m_x;
    output[10*index+7]=m_y;
    output[10*index+8]=m_dx;
    output[10*index+9]=m_dy;
  }
  else{
    output[10*index]=NULL;
    output[10*index+1]=NULL;
    output[10*index+2]=NULL;
    output[10*index+3]=NULL;
    output[10*index+4]=NULL;
    output[10*index+5]=NULL;
    output[10*index+6]=NULL;
    output[10*index+7]=NULL;
    output[10*index+8]=NULL;
    output[10*index+9]=NULL;
  }

  return;
}

__global__ void simulate_photon(float *input, float *output){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  float m_e=input[5*index];
  float m_x=input[5*index+1];
  float m_y=input[5*index+2];
  float m_dx=input[5*index+3];
  float m_dy=input[5*index+4];

  //e+
  float angle1 = atan(m_dy/m_dx);
  float angle2 = 3.1415926535897931 / 180.0;  //1 degree
  float hyp = sqrt(m_dx*m_dx + m_dy*m_dy);
  float angle3 = angle1+angle2;
  float outDy = hyp*sin(angle3);
  float outDx = hyp*cos(angle3);
  output[10*index]=m_e*0.5;
  output[10*index+1]=m_x+m_dx;
  output[10*index+2]=m_y+m_dy;
  output[10*index+3]=outDx;
  output[10*index+4]=outDy;
   
  //e-
  angle1 = atan(m_dy/m_dx);
  angle2 = 3.1415926535897931 / 180.0;  //1 degree
  hyp = sqrt(m_dx*m_dx + m_dy*m_dy);
  angle3 = angle1-angle2;
  outDy = hyp*sin(angle3);
  outDx = hyp*cos(angle3);
  output[10*index+5]=m_e*0.5;
  output[10*index+6]=m_x+m_dx;
  output[10*index+7]=m_y+m_dy;
  output[10*index+8]=outDx;
  output[10*index+9]=outDy;

  return;
}

Master::Master(){
  m_qel = new Queue();
  m_qph = new Queue();
  m_threshold=0.001; //1 kev
  m_nb_tot=0;
  m_e_tot=0;
}

Master::~Master(){
  delete m_qel;
  delete m_qph;
}

void Master::start(){
  unsigned int n_threads=128;

  //Process Electrons
  while(!m_qel->empty()){
    unsigned int size=m_qel->size();
    if(size<n_threads){
      Particle* p = m_qel->get();
      m_nb_tot+=1;
      m_e_tot+=p->m_e;
      p->simulate_until_decay(this);
      delete p;
    }
    else{
      //execute on GPU
      cout<<"Electrons GPU"<<endl;
      unsigned int n_blocks=size/n_threads;
      unsigned int n_elements=n_blocks*n_threads;
      unsigned int num_bytes_input = n_elements * 5 * sizeof(float);
      unsigned int num_bytes_output = num_bytes_input*2;

      //allocate host and device memory
      float *device_input = 0;
      float *device_output = 0;
      float host_input[5*n_elements];
      float host_output[2*5*n_elements];
      cudaMalloc((void**)&device_input, num_bytes_input);
      cudaMalloc((void**)&device_output, num_bytes_output);

      //fill input array
      unsigned int i=0;
      while(i<n_elements){
        Particle* tmp = m_qel->get();
        host_input[5*i]=tmp->m_e;
        host_input[5*i+1]=tmp->m_x;
        host_input[5*i+2]=tmp->m_y;
        host_input[5*i+3]=tmp->m_dx;
        host_input[5*i+4]=tmp->m_dy;
        i+=1;
        m_nb_tot+=1;
        m_e_tot+=tmp->m_e;
        delete tmp;
      }

      //copy input to device memory
      cudaMemcpy(device_input, host_input, num_bytes_input, cudaMemcpyHostToDevice); 

      //do work on GPU device
      simulate_electron<<<n_blocks,n_threads>>>(device_input,device_output); 

      //copy device memory back to host
      cudaMemcpy(host_output, device_output, num_bytes_output, cudaMemcpyDeviceToHost);

      //re-build output particles and put in queues
      i=0;
      float e1,x1,y1,dx1,dy1,e2,x2,y2,dx2,dy2;
      while(i<n_elements){
        e1=host_output[5*i];
	if(e1){
	  x1=host_output[5*i+1];
	  y1=host_output[5*i+2];
	  dx1=host_output[5*i+3];
	  dy1=host_output[5*i+4];
	  add_photon(e1,x1,y1,dx1,dy1);
	  e2=host_output[5*i+5];
	  x2=host_output[5*i+6];
	  y2=host_output[5*i+7];
	  dx2=host_output[5*i+8];
	  dy2=host_output[5*i+9];
	  add_electron(e2,x2,y2,dx2,dy2);	  
	}
        i+=1;
      }

      //deallocate host and device memory
      cudaFree(device_input);
      cudaFree(device_output);
    }
  }


  //Process Photons
  while(!m_qph->empty()){
    unsigned int size=m_qph->size();
    if(size<n_threads){
      Particle* p = m_qph->get();
      m_nb_tot+=1;
      m_e_tot+=p->m_e;
      p->simulate_until_decay(this);
      delete p;
    }
    else{
      //execute on GPU
      cout<<"Photons GPU"<<endl;
      unsigned int n_blocks=size/n_threads;
      unsigned int n_elements=n_blocks*n_threads;
      unsigned int num_bytes_input = n_elements * 5 * sizeof(float);
      unsigned int num_bytes_output = num_bytes_input*2;

      //allocate host and device memory
      float *device_input = 0;
      float *device_output = 0;
      float host_input[5*n_elements];
      float host_output[2*5*n_elements];
      cudaMalloc((void**)&device_input, num_bytes_input);
      cudaMalloc((void**)&device_output, num_bytes_output);

      //fill input array
      unsigned int i=0;
      while(i<n_elements){
        Particle* tmp = m_qph->get();
        host_input[5*i]=tmp->m_e;
        host_input[5*i+1]=tmp->m_x;
        host_input[5*i+2]=tmp->m_y;
        host_input[5*i+3]=tmp->m_dx;
        host_input[5*i+4]=tmp->m_dy;
        i+=1;
        m_nb_tot+=1;
        m_e_tot+=tmp->m_e;
        delete tmp;
      }

      //copy input to device memory
      cudaMemcpy(device_input, host_input, num_bytes_input, cudaMemcpyHostToDevice); 

      //do work on GPU device
      simulate_photon<<<n_blocks,n_threads>>>(device_input,device_output); 

      //copy device memory back to host
      cudaMemcpy(host_output, device_output, num_bytes_output, cudaMemcpyDeviceToHost);

      //re-build output particles and put in queues
      i=0;
      float e1,x1,y1,dx1,dy1,e2,x2,y2,dx2,dy2;
      while(i<n_elements){
        e1=host_output[5*i];
	if(e1){
	  x1=host_output[5*i+1];
	  y1=host_output[5*i+2];
	  dx1=host_output[5*i+3];
	  dy1=host_output[5*i+4];
	  add_electron(e1,x1,y1,dx1,dy1);
	  e2=host_output[5*i+5];
	  x2=host_output[5*i+6];
	  y2=host_output[5*i+7];
	  dx2=host_output[5*i+8];
	  dy2=host_output[5*i+9];
	  add_electron(e2,x2,y2,dx2,dy2);	  
	}
        i+=1;
      }

      //deallocate host and device memory
      cudaFree(device_input);
      cudaFree(device_output);
    }
  }


}

void Master::print(){
  std::cout<<"** Master Summary **"<<std::endl;
  std::cout<<"nb tot: "<< m_nb_tot <<std::endl;
  std::cout<<"e tot : "<< m_e_tot <<std::endl;
  return;
}

void Master::add_electron(float energy, float pos_x, float pos_y, float dir_x, float dir_y){
  Electron *el = new Electron(energy, pos_x, pos_y, dir_x, dir_y);
  m_qel->put(el);
  return;
}

void Master::add_photon(float energy, float pos_x, float pos_y, float dir_x, float dir_y){
  Photon *ph = new Photon(energy, pos_x, pos_y, dir_x, dir_y);
  m_qph->put(ph);
  return;
}


