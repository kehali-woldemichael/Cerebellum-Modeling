#include "objects.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
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
kernel_neurongroup_stateupdater_codeobject(
    int _N,
    int THREADS_PER_BLOCK,
    ///// KERNEL_PARAMETERS /////
    double* _ptr_array_neurongroup_s_GABA_1,
	double* _ptr_array_neurongroup_s_GABA_2,
	double* _ptr_array_neurongroup_s_AMPA,
	double* _ptr_array_neurongroup_V,
	const double _value_array_defaultclock_dt,
	double* _ptr_array_neurongroup_s_NMDA,
	double* _ptr_array_neurongroup_s_ahp_GR
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * THREADS_PER_BLOCK + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    const int _nums_GABA_1 = 100;
	const int _nums_GABA_2 = 100;
	const int _nums_AMPA = 100;
	const int _numV = 100;
	const int _nums_NMDA = 100;
	const int _nums_ahp_GR = 100;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_dt = &_value_array_defaultclock_dt;


    assert(THREADS_PER_BLOCK == blockDim.x);


    if(_idx >= _N)
    {
        return;
    }


    ///// scalar_code /////
        
    const double dt = _ptr_array_defaultclock_dt[0];
    const double _lio_1 = 1.0f*(- dt)/0.0012;
    const double _lio_2 = 1.0f*(- dt)/0.052000000000000005;
    const double _lio_3 = 1.0f*(- dt)/0.007;
    const double _lio_4 = 1.0f*(- dt)/0.059000000000000004;
    const double _lio_5 = 1.0f*dt/3.1e-12;
    const double _lio_6 = - 4.3e-10;
    const double _lio_7 = - (-0.058);
    const double _lio_8 = - 0.0;
    const double _lio_9 = 0.43 * 2.8000000000000003e-11;
    const double _lio_10 = - (-0.082);
    const double _lio_11 = 0.57 * 2.8000000000000003e-11;
    const double _lio_12 = - (-0.082);
    const double _lio_13 = 1.0f*(- dt)/0.005;


    {
        ///// vector_code /////
                
        double s_AMPA = _ptr_array_neurongroup_s_AMPA[_idx];
        double s_NMDA = _ptr_array_neurongroup_s_NMDA[_idx];
        double s_ahp_GR = _ptr_array_neurongroup_s_ahp_GR[_idx];
        double s_GABA_2 = _ptr_array_neurongroup_s_GABA_2[_idx];
        double V = _ptr_array_neurongroup_V[_idx];
        double s_GABA_1 = _ptr_array_neurongroup_s_GABA_1[_idx];
        const double _s_AMPA = (_lio_1 * s_AMPA) + s_AMPA;
        const double _s_NMDA = (_lio_2 * s_NMDA) + s_NMDA;
        const double _s_GABA_1 = (_lio_3 * s_GABA_1) + s_GABA_1;
        const double _s_GABA_2 = (_lio_4 * s_GABA_2) + s_GABA_2;
        const double _V = V + (_lio_5 * ((_lio_6 * (_lio_7 + V)) - (((((1.8e-10 * (s_AMPA * (_lio_8 + V))) + (_lio_9 * (s_GABA_1 * (_lio_10 + V)))) + (_lio_11 * (s_GABA_2 * (_lio_10 + V)))) + (2.5000000000000004e-11 * (s_NMDA * (_lio_8 + V)))) + (1e-09 * (s_ahp_GR * (_lio_12 + V))))));
        const double _s_ahp_GR = (_lio_13 * s_ahp_GR) + s_ahp_GR;
        s_AMPA = _s_AMPA;
        s_NMDA = _s_NMDA;
        s_GABA_1 = _s_GABA_1;
        s_GABA_2 = _s_GABA_2;
        V = _V;
        s_ahp_GR = _s_ahp_GR;
        _ptr_array_neurongroup_s_AMPA[_idx] = s_AMPA;
        _ptr_array_neurongroup_s_NMDA[_idx] = s_NMDA;
        _ptr_array_neurongroup_s_ahp_GR[_idx] = s_ahp_GR;
        _ptr_array_neurongroup_s_GABA_2[_idx] = s_GABA_2;
        _ptr_array_neurongroup_V[_idx] = V;
        _ptr_array_neurongroup_s_GABA_1[_idx] = s_GABA_1;


    }
}

void _run_neurongroup_stateupdater_codeobject()
{
    using namespace brian;


    const int _N = 100;

    ///// HOST_CONSTANTS ///////////
    const int _nums_GABA_1 = 100;
		const int _nums_GABA_2 = 100;
		const int _nums_AMPA = 100;
		const int _numV = 100;
		const int _nums_NMDA = 100;
		const int _nums_ahp_GR = 100;


    static int num_threads, num_blocks;
    static bool first_run = true;
    if (first_run)
    {
        // get number of blocks and threads
        int min_num_threads; // The minimum grid size needed to achieve the
                             // maximum occupancy for a full device launch

        CUDA_SAFE_CALL(
                cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
                    kernel_neurongroup_stateupdater_codeobject, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;

        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    kernel_neurongroup_stateupdater_codeobject, num_threads, 0)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, kernel_neurongroup_stateupdater_codeobject)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "kernel_neurongroup_stateupdater_codeobject "
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
                        kernel_neurongroup_stateupdater_codeobject, num_threads, 0)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }
        else
        {
            printf("INFO kernel_neurongroup_stateupdater_codeobject\n"
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


    kernel_neurongroup_stateupdater_codeobject<<<num_blocks, num_threads>>>(
            _N,
            num_threads,
            ///// HOST_PARAMETERS /////
            dev_array_neurongroup_s_GABA_1,
			dev_array_neurongroup_s_GABA_2,
			dev_array_neurongroup_s_AMPA,
			dev_array_neurongroup_V,
			_array_defaultclock_dt[0],
			dev_array_neurongroup_s_NMDA,
			dev_array_neurongroup_s_ahp_GR
        );

    CUDA_CHECK_ERROR("kernel_neurongroup_stateupdater_codeobject");


}


