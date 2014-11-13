#ifndef _array_hpp
#define _array_hpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>      // std::setfill, std::setw
#include <string.h>
#include <cuda.h>

using namespace std;

template<typename T>class Array
{
	private:
		T 		*h_ptr;		//CPU memory pointer
		T 		*d_ptr;		//GPU memory pointer
		int  	dimx;
		int 	dimy;
		int 	dimz;		// Currently supports up to 3 dimensions
		//int 	dimt;		
		int 	dimxy;
		int 	dimxyz;
		
		__forceinline__ 
		void cudaCheckLastError() 
		{                                          		
			cudaError_t error = cudaGetLastError();                               			
			int id; cudaGetDevice(&id);                                                     
			if(error != cudaSuccess) 
			{                                                      
				printf("Cuda failure error in file '%s' in line %i: '%s' at device %d \n",	
					__FILE__,__LINE__, cudaGetErrorString(error), id);                      
				exit(EXIT_FAILURE);                                                         
			}                                                                               
		}
		
	public:
		Array()
		{
			//Initialize array dimension
			this->dimx = 1;	
			this->dimy = 1;	
			this->dimz = 1;
			
			this->dimxy 	= dimx*dimy;
			this->dimxyz 	= dimx*dimy*dimz;
			
			h_ptr = new T[dimxyz];
			cudaMalloc((void**)&d_ptr, (dimxyz)*sizeof(T));
		};		
		
		Array(int dimx)
		{
			//Initialize array dimension
			this->dimx = dimx;	
			this->dimy = 1;	
			this->dimz = 1;
			
			this->dimxy 	= dimx*dimy;
			this->dimxyz 	= dimx*dimy*dimz;
			
			h_ptr = new T[dimxyz];
			cudaMalloc((void**)&d_ptr, (dimxyz)*sizeof(T));
		};
		
		Array(int dimx, int dimy)
		{
			//Initialize array dimension
			this->dimx = dimx;	
			this->dimy = dimy;	
			this->dimz = 1;
			
			this->dimxy 	= dimx*dimy;
			this->dimxyz 	= dimx*dimy*dimz;
			
			h_ptr = new T[dimxyz];
			cudaMalloc((void**)&d_ptr, (dimxyz)*sizeof(T));
		};
		
		Array(int dimx, int dimy, int dimz)
		{
			//Initialize array dimension
			this->dimx = dimx;	
			this->dimy = dimy;	
			this->dimz = dimz;
			
			this->dimxy 	= dimx*dimy;
			this->dimxyz 	= dimx*dimy*dimz;
			
			h_ptr = new T[dimxyz];
			cudaMalloc((void**)&d_ptr, (dimxyz)*sizeof(T));
		};
		
		~Array()
		{
			free(h_ptr);
			cudaFree(d_ptr);
		}
		void DeviceToHost()
		{
			cudaMemcpy(h_ptr, d_ptr, (dimxyz)*sizeof(T), cudaMemcpyDeviceToHost);
			cudaCheckLastError();
		};
		
		void HostToDevice()
		{	
			cudaMemcpy(d_ptr, h_ptr, (dimxyz)*sizeof(T), cudaMemcpyHostToDevice);
			cudaCheckLastError();
		};
		
		void ReadFile(char* filename, size_t size)
		{
			fstream *fs = new fstream;											
			fs->open(filename, ios::in|ios::binary);							
			if (!fs->is_open())															
			{																			
				printf("Cannot open file '%s' in file '%s' at line %i\n",				
				filename, __FILE__, __LINE__);											
				return;																
			}																			
			fs->read(reinterpret_cast<char*>(h_ptr), size);								
			fs->close();																
			delete fs;			
			
			//Send to GPU immediately
			HostToDevice();
		};
		
		void SaveFile(char* filename, size_t size)
		{
			//Send to CPU immediately
			this->DeviceToHost();
			
			fstream *fs = new fstream;													
			fs->open(filename, ios::out|ios::binary);									
			if (!fs->is_open())															
			{																			
				fprintf(stderr, "Cannot open file '%s' in file '%s' at line %i\n",		
				filename, __FILE__, __LINE__);											
				return;																
			}																			
			fs->write(reinterpret_cast<char*>(h_ptr), size);							
			fs->close();																
			delete fs;		
		}
		
		T* hPtr()		{return h_ptr;};	//Return the CPU pointer
		T* dPtr()		{return d_ptr;};	//Return the GPU pointer
};
#endif