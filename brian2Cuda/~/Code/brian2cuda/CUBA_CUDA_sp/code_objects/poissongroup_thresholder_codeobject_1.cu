#include "objects.h"
#include "code_objects/poissongroup_thresholder_codeobject_1.h"
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

        __global__ void
        _reset_poissongroup_thresholder_codeobject_1(
            int32_t* eventspace
            )
        {
            using namespace brian;

            int _idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (_idx == 0)
            {
                // reset eventspace counter
                eventspace[100] = 0;
            }

            if (_idx < 100)
            {
                // reset eventspace
                eventspace[_idx] = -1;
            }
        }

        
    #define _rand(vectorisation_idx) (_ptr_array_poissongroup_thresholder_codeobject_1_rand[vectorisation_idx])
    template <typename T>
    __host__ __device__
    double _brian_cos(T value)
    {
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
        return cos((double)value);
    #else
        return cos(value);
    #endif
    }
    inline __host__ __device__
    float _brian_cos(float value)
    {
        return cos(value);
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
kernel_poissongroup_thresholder_codeobject_1(
    int _N,
    int THREADS_PER_BLOCK,
    ///// KERNEL_PARAMETERS /////
    int32_t* _ptr_array_poissongroup__spikespace,
	const double _value_array_defaultclock_t,
	const double _value_array_defaultclock_dt,
	double* _ptr_array_poissongroup_thresholder_codeobject_1_rand
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * THREADS_PER_BLOCK + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    const int _num_spikespace = 101;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;
    const double* _ptr_array_defaultclock_dt = &_value_array_defaultclock_dt;


    assert(THREADS_PER_BLOCK == blockDim.x);


    if(_idx >= _N)
    {
        return;
    }


    ///// scalar_code /////
        
    const double t = _ptr_array_defaultclock_t[0];
    const double _lio_1 = (22.5 + (7.5 * _brian_cos((2.0 * ((3.141592653589793 * t) * 0.5)) + 3.141592653589793))) * 1.0;



    {// there might be the same variable defined in scalar and vector code
    ///// vector_code /////
        
    const double dt = _ptr_array_defaultclock_dt[0];
    const double rates = _lio_1;
    const char _cond = _rand(_vectorisation_idx + 0 * _N) < (dt * rates);


    if (_cond)
    {
        int32_t spike_index = atomicAdd(&_ptr_array_poissongroup__spikespace[_N], 1);
        _ptr_array_poissongroup__spikespace[spike_index] = _idx;
    }
    }
}

void _run_poissongroup_thresholder_codeobject_1()
{
    using namespace brian;

    const std::clock_t _start_time = std::clock();

    const int _N = 100;

    ///// HOST_CONSTANTS ///////////
    const int _num_spikespace = 101;


    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
        // get number of blocks and threads
        int min_num_threads; // The minimum grid size needed to achieve the
                             // maximum occupancy for a full device launch

        CUDA_SAFE_CALL(
                cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
                    kernel_poissongroup_thresholder_codeobject_1, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;

        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_poissongroup_thresholder_codeobject_1, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_poissongroup_thresholder_codeobject_1)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_poissongroup_thresholder_codeobject_1 "
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
                        kernel_poissongroup_thresholder_codeobject_1, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
        else
        {
            printf("INFO kernel_poissongroup_thresholder_codeobject_1\n"
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

        _reset_poissongroup_thresholder_codeobject_1<<<num_blocks, num_threads>>>(
                dev_array_poissongroup__spikespace[current_idx_array_poissongroup__spikespace]
            );

        CUDA_CHECK_ERROR("_reset_poissongroup_thresholder_codeobject_1");

    kernel_poissongroup_thresholder_codeobject_1<<<num_blocks, num_threads>>>(
            _N,
            num_threads,
            ///// HOST_PARAMETERS /////
            dev_array_poissongroup__spikespace[current_idx_array_poissongroup__spikespace],
			_array_defaultclock_t[0],
			_array_defaultclock_dt[0],
			dev_poissongroup_thresholder_codeobject_1_rand
        );

    CUDA_CHECK_ERROR("kernel_poissongroup_thresholder_codeobject_1");


    CUDA_SAFE_CALL(
            cudaDeviceSynchronize()
            );
    const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    poissongroup_thresholder_codeobject_1_profiling_info += _run_time;
}


