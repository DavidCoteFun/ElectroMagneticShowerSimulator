#include "Queue.cuh"

Queue::Queue(){
  m_size=0;
}

Queue::~Queue(){
 //call recursive deletion of particles
}

Particle* Queue::get(){
  if(m_size==0){ return NULL; }

  Particle* out=m_last;
  if(m_size==1){
    m_first=NULL;
    m_last=NULL;
  }
  else{
    Particle* tmp=m_last->prev;
    tmp->next=NULL;
    m_last=tmp;
  }
  m_size-=1;
  return out;
}

void Queue::put(Particle *p){
  if(m_size==0){
    m_first=p;
    m_last=p;
  }
  else{
    p->prev=m_last;
    m_last->next=p;
    m_last=p;
  }
  m_size+=1;
  return;
}

bool Queue::empty(){
  return (m_size==0);
}

unsigned int Queue::size(){
  return m_size;
}
