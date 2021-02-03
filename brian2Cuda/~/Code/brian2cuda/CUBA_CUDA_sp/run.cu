#include<stdlib.h>
#include "brianlib/cuda_utils.h"
#include "objects.h"
#include<ctime>

#include "code_objects/neurongroup_1_resetter_codeobject.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject.h"
#include "code_objects/neurongroup_1_thresholder_codeobject.h"
#include "code_objects/neurongroup_2_resetter_codeobject.h"
#include "code_objects/neurongroup_2_stateupdater_codeobject.h"
#include "code_objects/neurongroup_2_thresholder_codeobject.h"
#include "code_objects/neurongroup_3_resetter_codeobject.h"
#include "code_objects/neurongroup_3_stateupdater_codeobject.h"
#include "code_objects/neurongroup_3_thresholder_codeobject.h"
#include "code_objects/neurongroup_resetter_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"
#include "code_objects/poissongroup_1_thresholder_codeobject.h"
#include "code_objects/poissongroup_thresholder_codeobject.h"
#include "code_objects/ratemonitor_1_codeobject.h"
#include "code_objects/ratemonitor_2_codeobject.h"
#include "code_objects/ratemonitor_3_codeobject.h"
#include "code_objects/ratemonitor_codeobject.h"
#include "code_objects/spikemonitor_1_codeobject.h"
#include "code_objects/spikemonitor_2_codeobject.h"
#include "code_objects/spikemonitor_3_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/statemonitor_1_codeobject.h"
#include "code_objects/statemonitor_2_codeobject.h"
#include "code_objects/statemonitor_3_codeobject.h"
#include "code_objects/statemonitor_codeobject.h"
#include "code_objects/synapses_1_post_codeobject.h"
#include "code_objects/synapses_1_post_initialise_queue.h"
#include "code_objects/synapses_1_post_push_spikes.h"
#include "code_objects/synapses_1_pre_codeobject.h"
#include "code_objects/synapses_1_pre_initialise_queue.h"
#include "code_objects/synapses_1_pre_push_spikes.h"
#include "code_objects/synapses_1_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_2_post_codeobject.h"
#include "code_objects/synapses_2_post_initialise_queue.h"
#include "code_objects/synapses_2_post_push_spikes.h"
#include "code_objects/synapses_2_pre_codeobject.h"
#include "code_objects/synapses_2_pre_initialise_queue.h"
#include "code_objects/synapses_2_pre_push_spikes.h"
#include "code_objects/synapses_2_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_3_pre_codeobject.h"
#include "code_objects/synapses_3_pre_initialise_queue.h"
#include "code_objects/synapses_3_pre_push_spikes.h"
#include "code_objects/synapses_3_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_4_post_codeobject.h"
#include "code_objects/synapses_4_post_initialise_queue.h"
#include "code_objects/synapses_4_post_push_spikes.h"
#include "code_objects/synapses_4_pre_codeobject.h"
#include "code_objects/synapses_4_pre_initialise_queue.h"
#include "code_objects/synapses_4_pre_push_spikes.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_1.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_2.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_3.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_4.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_5.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_6.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_7.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_8.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject_9.h"
#include "code_objects/synapses_5_post_codeobject.h"
#include "code_objects/synapses_5_post_initialise_queue.h"
#include "code_objects/synapses_5_post_push_spikes.h"
#include "code_objects/synapses_5_pre_codeobject.h"
#include "code_objects/synapses_5_pre_initialise_queue.h"
#include "code_objects/synapses_5_pre_push_spikes.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_1.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_2.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_3.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_4.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_5.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_6.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_7.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_8.h"
#include "code_objects/synapses_5_synapses_create_array_codeobject_9.h"
#include "code_objects/synapses_6_pre_codeobject.h"
#include "code_objects/synapses_6_pre_initialise_queue.h"
#include "code_objects/synapses_6_pre_push_spikes.h"
#include "code_objects/synapses_6_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_post_codeobject.h"
#include "code_objects/synapses_post_initialise_queue.h"
#include "code_objects/synapses_post_push_spikes.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/synapses_pre_initialise_queue.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/synapses_synapses_create_generator_codeobject.h"


void brian_start()
{
    _init_arrays();
    _load_arrays();
    srand(time(NULL));

    // Initialize clocks (link timestep and dt to the respective arrays)
    brian::defaultclock.timestep = brian::_array_defaultclock_timestep;
    brian::defaultclock.dt = brian::_array_defaultclock_dt;
    brian::defaultclock.t = brian::_array_defaultclock_t;
}

void brian_end()
{
    _write_arrays();
    _dealloc_arrays();
}


