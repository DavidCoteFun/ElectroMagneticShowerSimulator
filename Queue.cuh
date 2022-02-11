#include "Common.cuh"
#include "Particle.cuh"

class Queue {
public:
  Queue();
  ~Queue();
  Particle* get();
  void put(Particle *p);
  bool empty();
  unsigned int size();

private:
  unsigned int m_size;
  Particle *m_first;
  Particle *m_last;
};
