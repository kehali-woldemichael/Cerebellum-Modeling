#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>
#include "run.h"
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include "rand.h"

#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/synapses_5_post_codeobject.h"
#include "code_objects/synapses_2_post_initialise_queue.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/neurongroup_2_resetter_codeobject.h"
#include "code_objects/neurongroup_2_stateupdater_codeobject.h"
#include "code_objects/synapses_4_pre_initialise_queue.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_6.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_7.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_4.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_5.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_2.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_3.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_1.h"
#include "code_objects/synapses_1_post_push_spikes.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_8.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_9.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject.h"
#include "code_objects/synapses_post_initialise_queue.h"
#include "code_objects/spikemonitor_3_codeobject.h"
#include "code_objects/synapses_2_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_5_pre_codeobject.h"
#include "code_objects/synapses_1_pre_push_spikes.h"
#include "code_objects/neurongroup_3_resetter_codeobject.h"
#include "code_objects/synapses_2_post_push_spikes.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject.h"
#include "code_objects/ratemonitor_1_codeobject.h"
#include "code_objects/neurongroup_2_thresholder_codeobject.h"
#include "code_objects/synapses_3_pre_codeobject.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_3.h"
#include "code_objects/synapses_1_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_4_pre_codeobject.h"
#include "code_objects/synapses_2_post_codeobject.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject.h"
#include "code_objects/synapses_post_codeobject.h"
#include "code_objects/neurongroup_1_resetter_codeobject.h"
#include "code_objects/synapses_1_pre_initialise_queue.h"
#include "code_objects/ratemonitor_3_codeobject.h"
#include "code_objects/synapses_1_post_initialise_queue.h"
#include "code_objects/synapses_6_pre_codeobject.h"
#include "code_objects/statemonitor_2_codeobject.h"
#include "code_objects/poissongroup_thresholder_codeobject.h"
#include "code_objects/synapses_pre_initialise_queue.h"
#include "code_objects/neurongroup_resetter_codeobject.h"
#include "code_objects/synapses_4_post_initialise_queue.h"
#include "code_objects/synapses_5_post_initialise_queue.h"
#include "code_objects/statemonitor_3_codeobject.h"
#include "code_objects/synapses_5_pre_initialise_queue.h"
#include "code_objects/statemonitor_1_codeobject.h"
#include "code_objects/ratemonitor_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/ratemonitor_2_codeobject.h"
#include "code_objects/spikemonitor_1_codeobject.h"
#include "code_objects/poissongroup_1_thresholder_codeobject.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"
#include "code_objects/synapses_6_pre_initialise_queue.h"
#include "code_objects/synapses_4_pre_push_spikes.h"
#include "code_objects/synapses_6_pre_push_spikes.h"
#include "code_objects/neurongroup_3_thresholder_codeobject.h"
#include "code_objects/synapses_3_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_post_push_spikes.h"
#include "code_objects/synapses_5_post_push_spikes.h"
#include "code_objects/synapses_2_pre_initialise_queue.h"
#include "code_objects/synapses_3_pre_initialise_queue.h"
#include "code_objects/synapses_6_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_4_post_codeobject.h"
#include "code_objects/neurongroup_1_thresholder_codeobject.h"
#include "code_objects/synapses_2_pre_codeobject.h"
#include "code_objects/synapses_2_pre_push_spikes.h"
#include "code_objects/statemonitor_codeobject.h"
#include "code_objects/synapses_3_pre_push_spikes.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_5.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_4.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_7.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_6.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_1.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_2.h"
#include "code_objects/synapses_5_pre_push_spikes.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_9.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_8.h"
#include "code_objects/synapses_1_post_codeobject.h"
#include "code_objects/synapses_4_post_push_spikes.h"
#include "code_objects/synapses_synapses_create_generator_codeobject.h"
#include "code_objects/neurongroup_3_stateupdater_codeobject.h"
#include "code_objects/spikemonitor_2_codeobject.h"
#include "code_objects/synapses_1_pre_codeobject.h"


#include <iostream>
#include <fstream>
#include "cuda_profiler_api.h"




int main(int argc, char **argv)
{
    // seed variable set in Python through brian2.seed() calls can use this
    // variable (see device.py CUDAStandaloneDevice.generate_main_source())
    unsigned long long seed;

    const std::clock_t _start_time = std::clock();

    const std::clock_t _start_time2 = std::clock();

    cudaDeviceProp props;
    CUDA_SAFE_CALL(
            cudaGetDeviceProperties(&props, 0)
            );
    size_t limit = 128 * 1024 * 1024;
    CUDA_SAFE_CALL(
            cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit)
            );
    CUDA_SAFE_CALL(
            cudaDeviceSynchronize()
            );

    const double _run_time2 = (double)(std::clock() -_start_time2)/CLOCKS_PER_SEC;
    printf("INFO: setting cudaDevice stuff took %f seconds\n", _run_time2);

    brian_start();

    const std::clock_t _start_time3 = std::clock();
    {
        using namespace brian;

                
                        for(int i=0; i<_num__array_neurongroup_1__spikespace; i++)
                        {
                            _array_neurongroup_1__spikespace[i] = -1;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace], &_array_neurongroup_1__spikespace[0],
                                        sizeof(_array_neurongroup_1__spikespace[0])*_num__array_neurongroup_1__spikespace, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_3__spikespace; i++)
                        {
                            _array_neurongroup_3__spikespace[i] = -1;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_3__spikespace[current_idx_array_neurongroup_3__spikespace], &_array_neurongroup_3__spikespace[0],
                                        sizeof(_array_neurongroup_3__spikespace[0])*_num__array_neurongroup_3__spikespace, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup__spikespace; i++)
                        {
                            _array_neurongroup__spikespace[i] = -1;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup__spikespace[current_idx_array_neurongroup__spikespace], &_array_neurongroup__spikespace[0],
                                        sizeof(_array_neurongroup__spikespace[0])*_num__array_neurongroup__spikespace, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_poissongroup_1__spikespace; i++)
                        {
                            _array_poissongroup_1__spikespace[i] = -1;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_poissongroup_1__spikespace[current_idx_array_poissongroup_1__spikespace], &_array_poissongroup_1__spikespace[0],
                                        sizeof(_array_poissongroup_1__spikespace[0])*_num__array_poissongroup_1__spikespace, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_poissongroup__spikespace; i++)
                        {
                            _array_poissongroup__spikespace[i] = -1;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_poissongroup__spikespace[current_idx_array_poissongroup__spikespace], &_array_poissongroup__spikespace[0],
                                        sizeof(_array_poissongroup__spikespace[0])*_num__array_poissongroup__spikespace, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_2__spikespace; i++)
                        {
                            _array_neurongroup_2__spikespace[i] = -1;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_2__spikespace[current_idx_array_neurongroup_2__spikespace], &_array_neurongroup_2__spikespace[0],
                                        sizeof(_array_neurongroup_2__spikespace[0])*_num__array_neurongroup_2__spikespace, cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_defaultclock_dt[0] = 0.0001;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(&dev_array_defaultclock_dt[0], &_array_defaultclock_dt[0],
                                        sizeof(_array_defaultclock_dt[0]), cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_defaultclock_dt[0] = 0.0001;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(&dev_array_defaultclock_dt[0], &_array_defaultclock_dt[0],
                                        sizeof(_array_defaultclock_dt[0]), cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_defaultclock_dt[0] = 0.0001;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(&dev_array_defaultclock_dt[0], &_array_defaultclock_dt[0],
                                        sizeof(_array_defaultclock_dt[0]), cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_V; i++)
                        {
                            _array_neurongroup_V[i] = -0.058;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_V, &_array_neurongroup_V[0],
                                        sizeof(_array_neurongroup_V[0])*_num__array_neurongroup_V, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_synapses_create_generator_codeobject();
        
                        for(int i=0; i<_dynamic_array_synapses_w_MFGR.size(); i++)
                        {
                            _dynamic_array_synapses_w_MFGR[i] = 4;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_synapses_w_MFGR[0]), &_dynamic_array_synapses_w_MFGR[0],
                                        sizeof(_dynamic_array_synapses_w_MFGR[0])*_dynamic_array_synapses_w_MFGR.size(), cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_1_V; i++)
                        {
                            _array_neurongroup_1_V[i] = -0.055;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_1_V, &_array_neurongroup_1_V[0],
                                        sizeof(_array_neurongroup_1_V[0])*_num__array_neurongroup_1_V, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_2_V; i++)
                        {
                            _array_neurongroup_2_V[i] = -0.068;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_2_V, &_array_neurongroup_2_V[0],
                                        sizeof(_array_neurongroup_2_V[0])*_num__array_neurongroup_2_V, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_1_synapses_create_generator_codeobject();
        
                        for(int i=0; i<_dynamic_array_synapses_1_w_CFPKJ.size(); i++)
                        {
                            _dynamic_array_synapses_1_w_CFPKJ[i] = 1.0;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1_w_CFPKJ[0]), &_dynamic_array_synapses_1_w_CFPKJ[0],
                                        sizeof(_dynamic_array_synapses_1_w_CFPKJ[0])*_dynamic_array_synapses_1_w_CFPKJ.size(), cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__array_neurongroup_3_V; i++)
                        {
                            _array_neurongroup_3_V[i] = -0.068;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_neurongroup_3_V, &_array_neurongroup_3_V[0],
                                        sizeof(_array_neurongroup_3_V[0])*_num__array_neurongroup_3_V, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_2_synapses_create_generator_codeobject();
        
                        for(int i=0; i<_dynamic_array_synapses_2_w_GRGO.size(); i++)
                        {
                            _dynamic_array_synapses_2_w_GRGO[i] = 4e-05;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2_w_GRGO[0]), &_dynamic_array_synapses_2_w_GRGO[0],
                                        sizeof(_dynamic_array_synapses_2_w_GRGO[0])*_dynamic_array_synapses_2_w_GRGO.size(), cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_3_synapses_create_generator_codeobject();
        
                        for(int i=0; i<_dynamic_array_synapses_3_w_GOGR.size(); i++)
                        {
                            _dynamic_array_synapses_3_w_GOGR[i] = 10;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_synapses_3_w_GOGR[0]), &_dynamic_array_synapses_3_w_GOGR[0],
                                        sizeof(_dynamic_array_synapses_3_w_GOGR[0])*_dynamic_array_synapses_3_w_GOGR.size(), cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_4_sources; i++)
                        {
                            _array_synapses_4_sources[i] = _static_array__array_synapses_4_sources[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_sources, &_array_synapses_4_sources[0],
                                        sizeof(_array_synapses_4_sources[0])*_num__array_synapses_4_sources, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_4_targets; i++)
                        {
                            _array_synapses_4_targets[i] = _static_array__array_synapses_4_targets[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_targets, &_array_synapses_4_targets[0],
                                        sizeof(_array_synapses_4_targets[0])*_num__array_synapses_4_targets, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_4_synapses_create_array_codeobject();
        
                        for(int i=0; i<_num__static_array__array_synapses_4_sources_1; i++)
                        {
                            _array_synapses_4_sources_1[i] = _static_array__array_synapses_4_sources_1[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_sources_1, &_array_synapses_4_sources_1[0],
                                        sizeof(_array_synapses_4_sources_1[0])*_num__array_synapses_4_sources_1, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_4_targets_1; i++)
                        {
                            _array_synapses_4_targets_1[i] = _static_array__array_synapses_4_targets_1[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_targets_1, &_array_synapses_4_targets_1[0],
                                        sizeof(_array_synapses_4_targets_1[0])*_num__array_synapses_4_targets_1, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_4_synapses_create_array_codeobject_1();
        
                        for(int i=0; i<_num__static_array__array_synapses_4_sources_2; i++)
                        {
                            _array_synapses_4_sources_2[i] = _static_array__array_synapses_4_sources_2[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_sources_2, &_array_synapses_4_sources_2[0],
                                        sizeof(_array_synapses_4_sources_2[0])*_num__array_synapses_4_sources_2, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_4_targets_2; i++)
                        {
                            _array_synapses_4_targets_2[i] = _static_array__array_synapses_4_targets_2[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_targets_2, &_array_synapses_4_targets_2[0],
                                        sizeof(_array_synapses_4_targets_2[0])*_num__array_synapses_4_targets_2, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_4_synapses_create_array_codeobject_2();
        
                        for(int i=0; i<_num__static_array__array_synapses_4_sources_3; i++)
                        {
                            _array_synapses_4_sources_3[i] = _static_array__array_synapses_4_sources_3[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_sources_3, &_array_synapses_4_sources_3[0],
                                        sizeof(_array_synapses_4_sources_3[0])*_num__array_synapses_4_sources_3, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_4_targets_3; i++)
                        {
                            _array_synapses_4_targets_3[i] = _static_array__array_synapses_4_targets_3[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_targets_3, &_array_synapses_4_targets_3[0],
                                        sizeof(_array_synapses_4_targets_3[0])*_num__array_synapses_4_targets_3, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_4_synapses_create_array_codeobject_3();
        
                        for(int i=0; i<_num__static_array__array_synapses_4_sources_4; i++)
                        {
                            _array_synapses_4_sources_4[i] = _static_array__array_synapses_4_sources_4[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_sources_4, &_array_synapses_4_sources_4[0],
                                        sizeof(_array_synapses_4_sources_4[0])*_num__array_synapses_4_sources_4, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_4_targets_4; i++)
                        {
                            _array_synapses_4_targets_4[i] = _static_array__array_synapses_4_targets_4[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_targets_4, &_array_synapses_4_targets_4[0],
                                        sizeof(_array_synapses_4_targets_4[0])*_num__array_synapses_4_targets_4, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_4_synapses_create_array_codeobject_4();
        
                        for(int i=0; i<_num__static_array__array_synapses_4_sources_5; i++)
                        {
                            _array_synapses_4_sources_5[i] = _static_array__array_synapses_4_sources_5[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_sources_5, &_array_synapses_4_sources_5[0],
                                        sizeof(_array_synapses_4_sources_5[0])*_num__array_synapses_4_sources_5, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_4_targets_5; i++)
                        {
                            _array_synapses_4_targets_5[i] = _static_array__array_synapses_4_targets_5[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_targets_5, &_array_synapses_4_targets_5[0],
                                        sizeof(_array_synapses_4_targets_5[0])*_num__array_synapses_4_targets_5, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_4_synapses_create_array_codeobject_5();
        
                        for(int i=0; i<_num__static_array__array_synapses_4_sources_6; i++)
                        {
                            _array_synapses_4_sources_6[i] = _static_array__array_synapses_4_sources_6[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_sources_6, &_array_synapses_4_sources_6[0],
                                        sizeof(_array_synapses_4_sources_6[0])*_num__array_synapses_4_sources_6, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_4_targets_6; i++)
                        {
                            _array_synapses_4_targets_6[i] = _static_array__array_synapses_4_targets_6[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_targets_6, &_array_synapses_4_targets_6[0],
                                        sizeof(_array_synapses_4_targets_6[0])*_num__array_synapses_4_targets_6, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_4_synapses_create_array_codeobject_6();
        
                        for(int i=0; i<_num__static_array__array_synapses_4_sources_7; i++)
                        {
                            _array_synapses_4_sources_7[i] = _static_array__array_synapses_4_sources_7[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_sources_7, &_array_synapses_4_sources_7[0],
                                        sizeof(_array_synapses_4_sources_7[0])*_num__array_synapses_4_sources_7, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_4_targets_7; i++)
                        {
                            _array_synapses_4_targets_7[i] = _static_array__array_synapses_4_targets_7[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_targets_7, &_array_synapses_4_targets_7[0],
                                        sizeof(_array_synapses_4_targets_7[0])*_num__array_synapses_4_targets_7, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_4_synapses_create_array_codeobject_7();
        
                        for(int i=0; i<_num__static_array__array_synapses_4_sources_8; i++)
                        {
                            _array_synapses_4_sources_8[i] = _static_array__array_synapses_4_sources_8[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_sources_8, &_array_synapses_4_sources_8[0],
                                        sizeof(_array_synapses_4_sources_8[0])*_num__array_synapses_4_sources_8, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_4_targets_8; i++)
                        {
                            _array_synapses_4_targets_8[i] = _static_array__array_synapses_4_targets_8[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_targets_8, &_array_synapses_4_targets_8[0],
                                        sizeof(_array_synapses_4_targets_8[0])*_num__array_synapses_4_targets_8, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_4_synapses_create_array_codeobject_8();
        
                        for(int i=0; i<_num__static_array__array_synapses_4_sources_9; i++)
                        {
                            _array_synapses_4_sources_9[i] = _static_array__array_synapses_4_sources_9[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_sources_9, &_array_synapses_4_sources_9[0],
                                        sizeof(_array_synapses_4_sources_9[0])*_num__array_synapses_4_sources_9, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_4_targets_9; i++)
                        {
                            _array_synapses_4_targets_9[i] = _static_array__array_synapses_4_targets_9[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_4_targets_9, &_array_synapses_4_targets_9[0],
                                        sizeof(_array_synapses_4_targets_9[0])*_num__array_synapses_4_targets_9, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_4_synapses_create_array_codeobject_9();
        
                        for(int i=0; i<_dynamic_array_synapses_4_w_GRPKJ.size(); i++)
                        {
                            _dynamic_array_synapses_4_w_GRPKJ[i] = 0.003;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_synapses_4_w_GRPKJ[0]), &_dynamic_array_synapses_4_w_GRPKJ[0],
                                        sizeof(_dynamic_array_synapses_4_w_GRPKJ[0])*_dynamic_array_synapses_4_w_GRPKJ.size(), cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_5_sources; i++)
                        {
                            _array_synapses_5_sources[i] = _static_array__array_synapses_5_sources[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_sources, &_array_synapses_5_sources[0],
                                        sizeof(_array_synapses_5_sources[0])*_num__array_synapses_5_sources, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_5_targets; i++)
                        {
                            _array_synapses_5_targets[i] = _static_array__array_synapses_5_targets[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_targets, &_array_synapses_5_targets[0],
                                        sizeof(_array_synapses_5_targets[0])*_num__array_synapses_5_targets, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_5_synapses_create_array_codeobject();
        
                        for(int i=0; i<_num__static_array__array_synapses_5_sources_1; i++)
                        {
                            _array_synapses_5_sources_1[i] = _static_array__array_synapses_5_sources_1[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_sources_1, &_array_synapses_5_sources_1[0],
                                        sizeof(_array_synapses_5_sources_1[0])*_num__array_synapses_5_sources_1, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_5_targets_1; i++)
                        {
                            _array_synapses_5_targets_1[i] = _static_array__array_synapses_5_targets_1[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_targets_1, &_array_synapses_5_targets_1[0],
                                        sizeof(_array_synapses_5_targets_1[0])*_num__array_synapses_5_targets_1, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_5_synapses_create_array_codeobject_1();
        
                        for(int i=0; i<_num__static_array__array_synapses_5_sources_2; i++)
                        {
                            _array_synapses_5_sources_2[i] = _static_array__array_synapses_5_sources_2[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_sources_2, &_array_synapses_5_sources_2[0],
                                        sizeof(_array_synapses_5_sources_2[0])*_num__array_synapses_5_sources_2, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_5_targets_2; i++)
                        {
                            _array_synapses_5_targets_2[i] = _static_array__array_synapses_5_targets_2[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_targets_2, &_array_synapses_5_targets_2[0],
                                        sizeof(_array_synapses_5_targets_2[0])*_num__array_synapses_5_targets_2, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_5_synapses_create_array_codeobject_2();
        
                        for(int i=0; i<_num__static_array__array_synapses_5_sources_3; i++)
                        {
                            _array_synapses_5_sources_3[i] = _static_array__array_synapses_5_sources_3[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_sources_3, &_array_synapses_5_sources_3[0],
                                        sizeof(_array_synapses_5_sources_3[0])*_num__array_synapses_5_sources_3, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_5_targets_3; i++)
                        {
                            _array_synapses_5_targets_3[i] = _static_array__array_synapses_5_targets_3[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_targets_3, &_array_synapses_5_targets_3[0],
                                        sizeof(_array_synapses_5_targets_3[0])*_num__array_synapses_5_targets_3, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_5_synapses_create_array_codeobject_3();
        
                        for(int i=0; i<_num__static_array__array_synapses_5_sources_4; i++)
                        {
                            _array_synapses_5_sources_4[i] = _static_array__array_synapses_5_sources_4[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_sources_4, &_array_synapses_5_sources_4[0],
                                        sizeof(_array_synapses_5_sources_4[0])*_num__array_synapses_5_sources_4, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_5_targets_4; i++)
                        {
                            _array_synapses_5_targets_4[i] = _static_array__array_synapses_5_targets_4[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_targets_4, &_array_synapses_5_targets_4[0],
                                        sizeof(_array_synapses_5_targets_4[0])*_num__array_synapses_5_targets_4, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_5_synapses_create_array_codeobject_4();
        
                        for(int i=0; i<_num__static_array__array_synapses_5_sources_5; i++)
                        {
                            _array_synapses_5_sources_5[i] = _static_array__array_synapses_5_sources_5[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_sources_5, &_array_synapses_5_sources_5[0],
                                        sizeof(_array_synapses_5_sources_5[0])*_num__array_synapses_5_sources_5, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_5_targets_5; i++)
                        {
                            _array_synapses_5_targets_5[i] = _static_array__array_synapses_5_targets_5[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_targets_5, &_array_synapses_5_targets_5[0],
                                        sizeof(_array_synapses_5_targets_5[0])*_num__array_synapses_5_targets_5, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_5_synapses_create_array_codeobject_5();
        
                        for(int i=0; i<_num__static_array__array_synapses_5_sources_6; i++)
                        {
                            _array_synapses_5_sources_6[i] = _static_array__array_synapses_5_sources_6[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_sources_6, &_array_synapses_5_sources_6[0],
                                        sizeof(_array_synapses_5_sources_6[0])*_num__array_synapses_5_sources_6, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_5_targets_6; i++)
                        {
                            _array_synapses_5_targets_6[i] = _static_array__array_synapses_5_targets_6[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_targets_6, &_array_synapses_5_targets_6[0],
                                        sizeof(_array_synapses_5_targets_6[0])*_num__array_synapses_5_targets_6, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_5_synapses_create_array_codeobject_6();
        
                        for(int i=0; i<_num__static_array__array_synapses_5_sources_7; i++)
                        {
                            _array_synapses_5_sources_7[i] = _static_array__array_synapses_5_sources_7[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_sources_7, &_array_synapses_5_sources_7[0],
                                        sizeof(_array_synapses_5_sources_7[0])*_num__array_synapses_5_sources_7, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_5_targets_7; i++)
                        {
                            _array_synapses_5_targets_7[i] = _static_array__array_synapses_5_targets_7[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_targets_7, &_array_synapses_5_targets_7[0],
                                        sizeof(_array_synapses_5_targets_7[0])*_num__array_synapses_5_targets_7, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_5_synapses_create_array_codeobject_7();
        
                        for(int i=0; i<_num__static_array__array_synapses_5_sources_8; i++)
                        {
                            _array_synapses_5_sources_8[i] = _static_array__array_synapses_5_sources_8[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_sources_8, &_array_synapses_5_sources_8[0],
                                        sizeof(_array_synapses_5_sources_8[0])*_num__array_synapses_5_sources_8, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_5_targets_8; i++)
                        {
                            _array_synapses_5_targets_8[i] = _static_array__array_synapses_5_targets_8[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_targets_8, &_array_synapses_5_targets_8[0],
                                        sizeof(_array_synapses_5_targets_8[0])*_num__array_synapses_5_targets_8, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_5_synapses_create_array_codeobject_8();
        
                        for(int i=0; i<_num__static_array__array_synapses_5_sources_9; i++)
                        {
                            _array_synapses_5_sources_9[i] = _static_array__array_synapses_5_sources_9[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_sources_9, &_array_synapses_5_sources_9[0],
                                        sizeof(_array_synapses_5_sources_9[0])*_num__array_synapses_5_sources_9, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_synapses_5_targets_9; i++)
                        {
                            _array_synapses_5_targets_9[i] = _static_array__array_synapses_5_targets_9[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_synapses_5_targets_9, &_array_synapses_5_targets_9[0],
                                        sizeof(_array_synapses_5_targets_9[0])*_num__array_synapses_5_targets_9, cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_5_synapses_create_array_codeobject_9();
        
                        for(int i=0; i<_dynamic_array_synapses_5_w_GRBS.size(); i++)
                        {
                            _dynamic_array_synapses_5_w_GRBS[i] = 0.003;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_synapses_5_w_GRBS[0]), &_dynamic_array_synapses_5_w_GRBS[0],
                                        sizeof(_dynamic_array_synapses_5_w_GRBS[0])*_dynamic_array_synapses_5_w_GRBS.size(), cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_6_synapses_create_generator_codeobject();
        
                        for(int i=0; i<_dynamic_array_synapses_6_w_BSPKJ.size(); i++)
                        {
                            _dynamic_array_synapses_6_w_BSPKJ[i] = 5.3;
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(thrust::raw_pointer_cast(&dev_dynamic_array_synapses_6_w_BSPKJ[0]), &_dynamic_array_synapses_6_w_BSPKJ[0],
                                        sizeof(_dynamic_array_synapses_6_w_BSPKJ[0])*_dynamic_array_synapses_6_w_BSPKJ.size(), cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_statemonitor__indices; i++)
                        {
                            _array_statemonitor__indices[i] = _static_array__array_statemonitor__indices[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_statemonitor__indices, &_array_statemonitor__indices[0],
                                        sizeof(_array_statemonitor__indices[0])*_num__array_statemonitor__indices, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_statemonitor_1__indices; i++)
                        {
                            _array_statemonitor_1__indices[i] = _static_array__array_statemonitor_1__indices[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_statemonitor_1__indices, &_array_statemonitor_1__indices[0],
                                        sizeof(_array_statemonitor_1__indices[0])*_num__array_statemonitor_1__indices, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_statemonitor_2__indices; i++)
                        {
                            _array_statemonitor_2__indices[i] = _static_array__array_statemonitor_2__indices[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_statemonitor_2__indices, &_array_statemonitor_2__indices[0],
                                        sizeof(_array_statemonitor_2__indices[0])*_num__array_statemonitor_2__indices, cudaMemcpyHostToDevice)
                                );
                        
        
                        for(int i=0; i<_num__static_array__array_statemonitor_3__indices; i++)
                        {
                            _array_statemonitor_3__indices[i] = _static_array__array_statemonitor_3__indices[i];
                        }
                        
        
                        CUDA_SAFE_CALL(
                                cudaMemcpy(dev_array_statemonitor_3__indices, &_array_statemonitor_3__indices[0],
                                        sizeof(_array_statemonitor_3__indices[0])*_num__array_statemonitor_3__indices, cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_defaultclock_timestep[0] = 0;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(&dev_array_defaultclock_timestep[0], &_array_defaultclock_timestep[0],
                                        sizeof(_array_defaultclock_timestep[0]), cudaMemcpyHostToDevice)
                                );
                        
        
                        _array_defaultclock_t[0] = 0.0;
                        CUDA_SAFE_CALL(
                                cudaMemcpy(&dev_array_defaultclock_t[0], &_array_defaultclock_t[0],
                                        sizeof(_array_defaultclock_t[0]), cudaMemcpyHostToDevice)
                                );
                        
        _run_synapses_1_pre_initialise_queue();
        _run_synapses_2_pre_initialise_queue();
        _run_synapses_3_pre_initialise_queue();
        _run_synapses_4_pre_initialise_queue();
        _run_synapses_5_pre_initialise_queue();
        _run_synapses_6_pre_initialise_queue();
        _run_synapses_pre_initialise_queue();
        _run_synapses_1_post_initialise_queue();
        _run_synapses_2_post_initialise_queue();
        _run_synapses_4_post_initialise_queue();
        _run_synapses_5_post_initialise_queue();
        _run_synapses_post_initialise_queue();
        
                                    dev_dynamic_array_synapses_5__synaptic_pre.clear();
                                    dev_dynamic_array_synapses_5__synaptic_pre.shrink_to_fit();
                                    
        
                                    dev_dynamic_array_synapses__synaptic_pre.clear();
                                    dev_dynamic_array_synapses__synaptic_pre.shrink_to_fit();
                                    
        
                                    dev_dynamic_array_synapses_6__synaptic_pre.clear();
                                    dev_dynamic_array_synapses_6__synaptic_pre.shrink_to_fit();
                                    
        
                                    dev_dynamic_array_synapses_4__synaptic_pre.clear();
                                    dev_dynamic_array_synapses_4__synaptic_pre.shrink_to_fit();
                                    
        
                                    dev_dynamic_array_synapses_2__synaptic_pre.clear();
                                    dev_dynamic_array_synapses_2__synaptic_pre.shrink_to_fit();
                                    
        
                                    dev_dynamic_array_synapses_3__synaptic_pre.clear();
                                    dev_dynamic_array_synapses_3__synaptic_pre.shrink_to_fit();
                                    
        
                                    dev_dynamic_array_synapses_1__synaptic_pre.clear();
                                    dev_dynamic_array_synapses_1__synaptic_pre.shrink_to_fit();
                                    
        magicnetwork.clear();
        magicnetwork.add(&defaultclock, _run_random_number_buffer);
        magicnetwork.add(&defaultclock, _run_statemonitor_codeobject);
        magicnetwork.add(&defaultclock, _run_statemonitor_1_codeobject);
        magicnetwork.add(&defaultclock, _run_statemonitor_2_codeobject);
        magicnetwork.add(&defaultclock, _run_statemonitor_3_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_1_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_2_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_3_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_1_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_2_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_3_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_poissongroup_1_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_poissongroup_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_spikemonitor_codeobject);
        magicnetwork.add(&defaultclock, _run_spikemonitor_1_codeobject);
        magicnetwork.add(&defaultclock, _run_spikemonitor_2_codeobject);
        magicnetwork.add(&defaultclock, _run_spikemonitor_3_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_1_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_1_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_2_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_2_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_3_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_3_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_4_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_4_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_5_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_5_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_6_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_6_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_1_post_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_1_post_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_2_post_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_2_post_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_4_post_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_4_post_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_5_post_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_5_post_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_post_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_post_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_1_resetter_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_2_resetter_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_3_resetter_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_resetter_codeobject);
        magicnetwork.add(&defaultclock, _run_ratemonitor_codeobject);
        magicnetwork.add(&defaultclock, _run_ratemonitor_1_codeobject);
        magicnetwork.add(&defaultclock, _run_ratemonitor_2_codeobject);
        magicnetwork.add(&defaultclock, _run_ratemonitor_3_codeobject);
        CUDA_SAFE_CALL(cudaProfilerStart());
        magicnetwork.run(2.0, NULL, 10.0);
        random_number_buffer.run_finished();
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        CUDA_SAFE_CALL(cudaProfilerStop());
        _debugmsg_synapses_5_post_codeobject();
        
        _debugmsg_synapses_pre_codeobject();
        
        _copyToHost_spikemonitor_3_codeobject();
        _debugmsg_spikemonitor_3_codeobject();
        
        _debugmsg_synapses_5_pre_codeobject();
        
        _debugmsg_synapses_3_pre_codeobject();
        
        _debugmsg_synapses_4_pre_codeobject();
        
        _debugmsg_synapses_2_post_codeobject();
        
        _debugmsg_synapses_post_codeobject();
        
        _debugmsg_synapses_6_pre_codeobject();
        
        _copyToHost_spikemonitor_1_codeobject();
        _debugmsg_spikemonitor_1_codeobject();
        
        _debugmsg_synapses_4_post_codeobject();
        
        _debugmsg_synapses_2_pre_codeobject();
        
        _copyToHost_spikemonitor_codeobject();
        _debugmsg_spikemonitor_codeobject();
        
        _debugmsg_synapses_1_post_codeobject();
        
        _copyToHost_spikemonitor_2_codeobject();
        _debugmsg_spikemonitor_2_codeobject();
        
        _debugmsg_synapses_1_pre_codeobject();

    }

    const double _run_time3 = (double)(std::clock() -_start_time3)/CLOCKS_PER_SEC;
    printf("INFO: main_lines took %f seconds\n", _run_time3);

    brian_end();

    // Profiling
    const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    printf("INFO: main function took %f seconds\n", _run_time);

    return 0;
}