#ifndef MASTER_CUH
#define MASTER_CUH

#include "Common.cuh"
class Queue;

class Master {
  public:
    Master();
    ~Master();
    void start();
    void print();

    void add_electron(float energy, float pos_x, float pos_y, float dir_x, float dir_y);
    void add_photon(float energy, float pos_x, float pos_y, float dir_x, float dir_y);

    float m_threshold;

  private:
    Queue *m_qel;
    Queue *m_qph;
    unsigned int m_nb_tot;
    float m_e_tot;
};

#endif
