#include "objects.h"
#include "code_objects/statemonitor_2_codeobject.h"
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include "brianlib/stdint_compat.h"
#include <cmath>
#include <stdint.h>
#include <ctime>
#include <stdio.h>

////// SUPPORT CODE ///////
namespace {
    // Implement dummy functions such that the host compiled code of binomial
    // functions works. Hacky, hacky ...
    double host_rand(const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `host_rand` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }
    double host_randn(const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `host_rand` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }


        
    template < typename T1, typename T2 > struct _higher_type;
    template < > struct _higher_type<int,int> { typedef int type; };
    template < > struct _higher_type<int,long> { typedef long type; };
    template < > struct _higher_type<int,long long> { typedef long long type; };
    template < > struct _higher_type<int,float> { typedef float type; };
    template < > struct _higher_type<int,double> { typedef double type; };
    template < > struct _higher_type<long,int> { typedef long type; };
    template < > struct _higher_type<long,long> { typedef long type; };
    template < > struct _higher_type<long,long long> { typedef long long type; };
    template < > struct _higher_type<long,float> { typedef float type; };
    template < > struct _higher_type<long,double> { typedef double type; };
    template < > struct _higher_type<long long,int> { typedef long long type; };
    template < > struct _higher_type<long long,long> { typedef long long type; };
    template < > struct _higher_type<long long,long long> { typedef long long type; };
    template < > struct _higher_type<long long,float> { typedef float type; };
    template < > struct _higher_type<long long,double> { typedef double type; };
    template < > struct _higher_type<float,int> { typedef float type; };
    template < > struct _higher_type<float,long> { typedef float type; };
    template < > struct _higher_type<float,long long> { typedef float type; };
    template < > struct _higher_type<float,float> { typedef float type; };
    template < > struct _higher_type<float,double> { typedef double type; };
    template < > struct _higher_type<double,int> { typedef double type; };
    template < > struct _higher_type<double,long> { typedef double type; };
    template < > struct _higher_type<double,long long> { typedef double type; };
    template < > struct _higher_type<double,float> { typedef double type; };
    template < > struct _higher_type<double,double> { typedef double type; };
    template < typename T1, typename T2 >
    __host__ __device__ static inline typename _higher_type<T1,T2>::type
    _brian_mod(T1 x, T2 y)
    {{
        return x-y*floor(1.0*x/y);
    }}
    template < typename T1, typename T2 >
    __host__ __device__ static inline typename _higher_type<T1,T2>::type
    _brian_floordiv(T1 x, T2 y)
    {{
        return floor(1.0*x/y);
    }}
    #ifdef _MSC_VER
    #define _brian_pow(x, y) (pow((double)(x), (y)))
    #else
    #define _brian_pow(x, y) (pow((x), (y)))
    #endif
                inline __device__ int _brian_atomicAdd(int* address, int val)
                {
                    // hardware implementation
                    return atomicAdd(address, val);
                }
                inline __device__ float _brian_atomicAdd(float* address, float val)
                {
                    // hardware implementation
                    return atomicAdd(address, val);
                }
                inline __device__ double _brian_atomicAdd(double* address, double val)
                {
                    // software implementation
                    unsigned long long int* address_as_int = (unsigned long long int*)address;
                    unsigned long long int old = *address_as_int, assumed;
                    do {
                        assumed = old;
                        old = atomicCAS(address_as_int, assumed,
                                        __double_as_longlong(val +
                                               __longlong_as_double(assumed)));
                    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
                    } while (assumed != old);
                    return __longlong_as_double(old);
                }
                inline __device__ int _brian_atomicMul(int* address, int val)
                {
                    // software implementation
                    int old = *address, assumed;
                    do {
                        assumed = old;
                        old = atomicCAS(address, assumed, val * assumed);
                    } while (assumed != old);
                    return old;
                }
                inline __device__ float _brian_atomicMul(float* address, float val)
                {
                    // software implementation
                    int* address_as_int = (int*)address;
                    int old = *address_as_int, assumed;
                    do {
                        assumed = old;
                        old = atomicCAS(address_as_int, assumed,
                                        __float_as_int(val *
                                               __int_as_float(assumed)));
                    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
                    } while (assumed != old);
                    return __int_as_float(old);
                }
                inline __device__ double _brian_atomicMul(double* address, double val)
                {
                    // software implementation
                    unsigned long long int* address_as_int = (unsigned long long int*)address;
                    unsigned long long int old = *address_as_int, assumed;
                    do {
                        assumed = old;
                        old = atomicCAS(address_as_int, assumed,
                                        __double_as_longlong(val *
                                               __longlong_as_double(assumed)));
                    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
                    } while (assumed != old);
                    return __longlong_as_double(old);
                }
                inline __device__ int _brian_atomicDiv(int* address, int val)
                {
                    // software implementation
                    int old = *address, assumed;
                    do {
                        assumed = old;
                        old = atomicCAS(address, assumed, val / assumed);
                    } while (assumed != old);
                    return old;
                }
                inline __device__ float _brian_atomicDiv(float* address, float val)
                {
                    // software implementation
                    int* address_as_int = (int*)address;
                    int old = *address_as_int, assumed;
                    do {
                        assumed = old;
                        old = atomicCAS(address_as_int, assumed,
                                        __float_as_int(val /
                                               __int_as_float(assumed)));
                    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
                    } while (assumed != old);
                    return __int_as_float(old);
                }
                inline __device__ double _brian_atomicDiv(double* address, double val)
                {
                    // software implementation
                    unsigned long long int* address_as_int = (unsigned long long int*)address;
                    unsigned long long int old = *address_as_int, assumed;
                    do {
                        assumed = old;
                        old = atomicCAS(address_as_int, assumed,
                                        __double_as_longlong(val /
                                               __longlong_as_double(assumed)));
                    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
                    } while (assumed != old);
                    return __longlong_as_double(old);
                }

}




__global__ void
kernel_statemonitor_2_codeobject(
    int _num_indices,
    int32_t* indices,
    int current_iteration,
    double** monitor_V,
    ///// KERNEL_PARAMETERS /////
    double* _ptr_array_neurongroup_3_V,
	double* _ptr_array_statemonitor_2_t,
	const int _numt,
	int32_t* _ptr_array_statemonitor_2_N
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    if(tid > _num_indices)
    {
        return;
    }
    int32_t _idx = indices[tid];

    ///// KERNEL_CONSTANTS /////
    const int _num_source_V = 10;
	const int _numN = 1;

    ///// kernel_lines /////
        


    ///// scalar_code /////
        


    // need different scope here since scalar_code and vector_code can
    // declare the same variables
    {
        ///// vector_code /////
                
        const double _source_V = _ptr_array_neurongroup_3_V[_idx];
        const double _to_record_V = _source_V;


        monitor_V[tid][current_iteration] = _to_record_V;
    }
}

void _run_statemonitor_2_codeobject()
{
    using namespace brian;



    ///// HOST_CONSTANTS ///////////
    const int _num_source_V = 10;
		double* const _array_statemonitor_2_t = thrust::raw_pointer_cast(&dev_dynamic_array_statemonitor_2_t[0]);
		const int _numt = dev_dynamic_array_statemonitor_2_t.size();
		const int _numN = 1;

// TODO: this pushes a new value to the device each time step? Looks
// inefficient, can we keep the t values on the host instead? Do we need them
// on the device?
dev_dynamic_array_statemonitor_2_t.push_back(defaultclock.t[0]);

int num_iterations = defaultclock.i_end;
int current_iteration = defaultclock.timestep[0];
static int start_offset = current_iteration - _numt;

    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
addresses_monitor__dynamic_array_statemonitor_2_V.clear();
for(int i = 0; i < _num__array_statemonitor_2__indices; i++)
{
    _dynamic_array_statemonitor_2_V[i].resize(_numt + num_iterations - current_iteration);
    addresses_monitor__dynamic_array_statemonitor_2_V.push_back(thrust::raw_pointer_cast(&_dynamic_array_statemonitor_2_V[i][0]));
}
// Print a warning when the monitor is not going to work (#50)
if (_num__array_statemonitor_2__indices > 1024)
{
    printf("ERROR in statemonitor_2: Too many neurons recorded. Due to a bug (brian-team/brian2cuda#50), "
            "currently only as many neurons can be recorded as threads can be called from a single block!\n");
}



        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_statemonitor_2_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_statemonitor_2_codeobject "
                   "with maximum possible threads per block (%u). "
                   "Reducing num_threads to %u. (Kernel needs %i "
                   "registers per block, %i bytes of "
                   "statically-allocated shared memory per block, %i "
                   "bytes of local memory per thread and a total of %i "
                   "bytes of user-allocated constant memory)\n",
                   max_threads_per_block, num_threads, funcAttrib.numRegs,
                   funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                   funcAttrib.constSizeBytes);

        }
        first_run = false;
    }


// If the StateMonitor is run outside the MagicNetwork, we need to resize it.
// Happens e.g. when StateMonitor.record_single_timestep() is called.
if(current_iteration >= num_iterations)
{
    for(int i = 0; i < _num__array_statemonitor_2__indices; i++)
    {
        _dynamic_array_statemonitor_2_V[i].resize(_numt + 1);
        addresses_monitor__dynamic_array_statemonitor_2_V[i] = thrust::raw_pointer_cast(&_dynamic_array_statemonitor_2_V[i][0]);
    }
}

if (_num__array_statemonitor_2__indices > 0)
// TODO we get invalid launch configuration if this is 0, which happens e.g. for StateMonitor(..., variables=[])
{
    kernel_statemonitor_2_codeobject<<<1, _num__array_statemonitor_2__indices>>>(
        _num__array_statemonitor_2__indices,
        dev_array_statemonitor_2__indices,
        current_iteration - start_offset,
        thrust::raw_pointer_cast(&addresses_monitor__dynamic_array_statemonitor_2_V[0]),
        ///// HOST_PARAMETERS /////
        dev_array_neurongroup_3_V,
			_array_statemonitor_2_t,
			_numt,
			dev_array_statemonitor_2_N
        );

    CUDA_CHECK_ERROR("kernel_statemonitor_2_codeobject");
}


}


