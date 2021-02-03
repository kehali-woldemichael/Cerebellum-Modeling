#include "objects.h"
#include "code_objects/ratemonitor_3_codeobject_1.h"
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
kernel_ratemonitor_3_codeobject_1(
    int32_t current_iteration,
    double* ratemonitor_rate,
    double* ratemonitor_t,
    ///// KERNEL_PARAMETERS /////
    const double _value_array_defaultclock_t,
	int32_t* _ptr_array_neurongroup_1__spikespace,
	double* _ptr_array_ratemonitor_3_rate,
	const int _numrate,
	const double _value_array_defaultclock_dt,
	double* _ptr_array_ratemonitor_3_t,
	const int _numt
    )
{
    using namespace brian;

    ///// KERNEL_CONSTANTS /////
    const int _num_spikespace = 11;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;
    const double* _ptr_array_defaultclock_dt = &_value_array_defaultclock_dt;


    int num_spikes = 0;

    if (_num_spikespace-1 != 10)  // we have a subgroup
    {
        // TODO shouldn't this be 'i < _num_spikespace -1'?
        for (int i=0; i < _num_spikespace; i++)
        {
            const int spiking_neuron = _ptr_array_neurongroup_1__spikespace[i];
            if (spiking_neuron != -1)
            {
                // check if spiking neuron is in this subgroup
                if (0 <= spiking_neuron && spiking_neuron < 10)
                    num_spikes++;
            }
            else  // end of spiking neurons
            {
                break;
            }
        }
    }
    else  // we don't have a subgroup
    {
        num_spikes = _ptr_array_neurongroup_1__spikespace[10];
    }

    // TODO: we should be able to use _ptr_array_ratemonitor_3_rate and _ptr_array_ratemonitor_3_t here instead of passing these
    //       additional pointers. But this results in thrust::system_error illegal memory access.
    //       Don't know why... _ptr_array_ratemonitor_3_rate and ratemonitor_rate should be the same...
    ratemonitor_rate[current_iteration] = 1.0*num_spikes/_ptr_array_defaultclock_dt[0]/10;
    ratemonitor_t[current_iteration] = _ptr_array_defaultclock_t[0];
}

void _run_ratemonitor_3_codeobject_1()
{
    using namespace brian;

    const std::clock_t _start_time = std::clock();


    ///// HOST_CONSTANTS ///////////
    const int _num_spikespace = 11;
		double* const _array_ratemonitor_3_rate = thrust::raw_pointer_cast(&dev_dynamic_array_ratemonitor_3_rate[0]);
		const int _numrate = dev_dynamic_array_ratemonitor_3_rate.size();
		double* const _array_ratemonitor_3_t = thrust::raw_pointer_cast(&dev_dynamic_array_ratemonitor_3_t[0]);
		const int _numt = dev_dynamic_array_ratemonitor_3_t.size();

int current_iteration = defaultclock.timestep[0];
static int start_offset = current_iteration;

    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
int num_iterations = defaultclock.i_end;
int size_till_now = dev_dynamic_array_ratemonitor_3_t.size();
THRUST_CHECK_ERROR(
        dev_dynamic_array_ratemonitor_3_t.resize(num_iterations + size_till_now - start_offset)
        );
THRUST_CHECK_ERROR(
        dev_dynamic_array_ratemonitor_3_rate.resize(num_iterations + size_till_now - start_offset)
        );
num_threads = 1;
num_blocks = 1;

        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_ratemonitor_3_codeobject_1, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_ratemonitor_3_codeobject_1)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_ratemonitor_3_codeobject_1 "
                   "with maximum possible threads per block (%u). "
                   "Reducing num_threads to %u. (Kernel needs %i "
                   "registers per block, %i bytes of "
                   "statically-allocated shared memory per block, %i "
                   "bytes of local memory per thread and a total of %i "
                   "bytes of user-allocated constant memory)\n",
                   max_threads_per_block, num_threads, funcAttrib.numRegs,
                   funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                   funcAttrib.constSizeBytes);

            // calculate theoretical occupancy for new num_threads
            CUDA_SAFE_CALL(
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                        kernel_ratemonitor_3_codeobject_1, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
        else
        {
            printf("INFO kernel_ratemonitor_3_codeobject_1\n"
                   "\t%u blocks\n"
                   "\t%u threads\n"
                   "\t%i registers per block\n"
                   "\t%i bytes statically-allocated shared memory per block\n"
                   "\t%i bytes local memory per thread\n"
                   "\t%i bytes user-allocated constant memory\n"
                   "\t%.3f theoretical occupancy\n",
                   num_blocks, num_threads, funcAttrib.numRegs,
                   funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                   funcAttrib.constSizeBytes, occupancy);
        }
        first_run = false;
    }


kernel_ratemonitor_3_codeobject_1<<<num_blocks, num_threads>>>(
    current_iteration - start_offset,
    thrust::raw_pointer_cast(&(dev_dynamic_array_ratemonitor_3_rate[0])),
    thrust::raw_pointer_cast(&(dev_dynamic_array_ratemonitor_3_t[0])),
    ///// HOST_PARAMETERS /////
    _array_defaultclock_t[0],
			dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace],
			_array_ratemonitor_3_rate,
			_numrate,
			_array_defaultclock_dt[0],
			_array_ratemonitor_3_t,
			_numt);

CUDA_CHECK_ERROR("kernel_ratemonitor_3_codeobject_1");


    CUDA_SAFE_CALL(
            cudaDeviceSynchronize()
            );
    const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    ratemonitor_3_codeobject_1_profiling_info += _run_time;
}


