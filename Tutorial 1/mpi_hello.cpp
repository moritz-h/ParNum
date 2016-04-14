#include <iostream>
#include <mpi.h>

/*
  Auf gleiche Lib bei compile und run achten (OpenMPI, mpich)
  mpic++ mpi_hello.cpp
  mpirun -n 4 a.out 
*/

int main(int argc, char **argv)
{
  int size, rank;
  MPI_Init(&argc, &argv);
  
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::cout << "Size = " << size << " Rank = " << rank << std::endl;
  
  MPI_Finalize();
   
  return 0;
}
