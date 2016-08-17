#include <iostream>
#include <string>
#include <mpi.h>

/*
  Auf gleiche Lib bei compile und run achten (OpenMPI, mpich)
  mpic++ mpi_hello2.cpp
  mpirun -n 4 a.out 
*/

using std::cout;
using std::endl;

int main(int argc, char **argv)
{
  int size, rank;
  MPI_Init(&argc, &argv);
  
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    cout << "This is rank 0 talking to my " << size << " friends." << endl << endl; 
    std::string hello("Hello my fellow processes!");
    for (int i = 1; i < size; i++) { // Start at one, we don't want to send a message to ourself, see footnote
      //         MPI_Send(void* data,                       int count,    MPI_Datatype datatype, int destination, int tag, MPI_Comm communicator);
      int ierr = MPI_Send(const_cast<char*>(hello.c_str()), hello.size(), MPI_CHAR,              i,               0,       MPI_COMM_WORLD);
      cout << "Sent: " << hello << " with return code " << ierr << endl;
    }
  }
  else {
    int size;
    // We are receiving a string, which is char array of unknown size.
    // Due to that we first check the size and reserve an array of appropriate size
    MPI_Status stat;
    // MPI_Probe(int source,     int tag,     MPI_Comm comm,  MPI_Status *status)
    MPI_Probe(   MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
    MPI_Get_count(&stat, MPI_CHAR, &size);
    cout << "Awaiting message of size " << size << endl;
      
    char buf[size];
    // MPI_Recv(void* data, int count, MPI_Datatype datatype, int source,     int tag,     MPI_Comm communicator, MPI_Status* status)
    MPI_Recv(   buf,        size,      MPI_CHAR,              MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,        &stat);
    std::string rec(buf, size);
    cout << "Received: " << rec << endl;
  }  
  
  MPI_Finalize();   
  return 0;
}

// In a standard mode send (i.e. MPI_Send()), it is up to the MPI
// implementation to determine whether to buffer the message or
// not. It is reasonable to assume that any implementation, or at
// least the popular ones, will recognize a send to self, and decide
// to buffer the message. Execution will then continue, and once the
// matching receive call is made, the message will be read from the
// buffer. If you want to be absolutely certain, you can use
// MPI_Bsend(), but then it may be your responsibility to manage the
// buffer via MPI_Buffer_attach() and MPI_Buffer_detach().

// Therefore behavior is implementation specific.

// http://stackoverflow.com/questions/11385395/is-the-behavior-of-mpi-communication-of-a-rank-with-itself-well-defined
