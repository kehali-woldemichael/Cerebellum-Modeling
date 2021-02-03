
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/cuda_utils.h"
#include "network.h"
#include "rand.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <utility>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>
#include <curand_kernel.h>

size_t brian::used_device_memory = 0;

//////////////// clocks ///////////////////
Clock brian::defaultclock;

//////////////// networks /////////////////
Network brian::magicnetwork;

//////////////// arrays ///////////////////
double * brian::_array_defaultclock_dt;
double * brian::dev_array_defaultclock_dt;
__device__ double * brian::d_array_defaultclock_dt;
const int brian::_num__array_defaultclock_dt = 1;

double * brian::_array_defaultclock_t;
double * brian::dev_array_defaultclock_t;
__device__ double * brian::d_array_defaultclock_t;
const int brian::_num__array_defaultclock_t = 1;

int64_t * brian::_array_defaultclock_timestep;
int64_t * brian::dev_array_defaultclock_timestep;
__device__ int64_t * brian::d_array_defaultclock_timestep;
const int brian::_num__array_defaultclock_timestep = 1;

int32_t * brian::_array_neurongroup_1_i;
int32_t * brian::dev_array_neurongroup_1_i;
__device__ int32_t * brian::d_array_neurongroup_1_i;
const int brian::_num__array_neurongroup_1_i = 10;

double * brian::_array_neurongroup_1_s_ahp_GO;
double * brian::dev_array_neurongroup_1_s_ahp_GO;
__device__ double * brian::d_array_neurongroup_1_s_ahp_GO;
const int brian::_num__array_neurongroup_1_s_ahp_GO = 10;

double * brian::_array_neurongroup_1_s_AMPA;
double * brian::dev_array_neurongroup_1_s_AMPA;
__device__ double * brian::d_array_neurongroup_1_s_AMPA;
const int brian::_num__array_neurongroup_1_s_AMPA = 10;

double * brian::_array_neurongroup_1_s_NMDA_1;
double * brian::dev_array_neurongroup_1_s_NMDA_1;
__device__ double * brian::d_array_neurongroup_1_s_NMDA_1;
const int brian::_num__array_neurongroup_1_s_NMDA_1 = 10;

double * brian::_array_neurongroup_1_s_NMDA_2;
double * brian::dev_array_neurongroup_1_s_NMDA_2;
__device__ double * brian::d_array_neurongroup_1_s_NMDA_2;
const int brian::_num__array_neurongroup_1_s_NMDA_2 = 10;

double * brian::_array_neurongroup_1_V;
double * brian::dev_array_neurongroup_1_V;
__device__ double * brian::d_array_neurongroup_1_V;
const int brian::_num__array_neurongroup_1_V = 10;

double * brian::_array_neurongroup_1_x;
double * brian::dev_array_neurongroup_1_x;
__device__ double * brian::d_array_neurongroup_1_x;
const int brian::_num__array_neurongroup_1_x = 10;

double * brian::_array_neurongroup_1_y;
double * brian::dev_array_neurongroup_1_y;
__device__ double * brian::d_array_neurongroup_1_y;
const int brian::_num__array_neurongroup_1_y = 10;

int32_t * brian::_array_neurongroup_2_i;
int32_t * brian::dev_array_neurongroup_2_i;
__device__ int32_t * brian::d_array_neurongroup_2_i;
const int brian::_num__array_neurongroup_2_i = 10;

double * brian::_array_neurongroup_2_s_AHP_PKJ;
double * brian::dev_array_neurongroup_2_s_AHP_PKJ;
__device__ double * brian::d_array_neurongroup_2_s_AHP_PKJ;
const int brian::_num__array_neurongroup_2_s_AHP_PKJ = 10;

double * brian::_array_neurongroup_2_s_AMPA;
double * brian::dev_array_neurongroup_2_s_AMPA;
__device__ double * brian::d_array_neurongroup_2_s_AMPA;
const int brian::_num__array_neurongroup_2_s_AMPA = 10;

double * brian::_array_neurongroup_2_s_GABA;
double * brian::dev_array_neurongroup_2_s_GABA;
__device__ double * brian::d_array_neurongroup_2_s_GABA;
const int brian::_num__array_neurongroup_2_s_GABA = 10;

double * brian::_array_neurongroup_2_V;
double * brian::dev_array_neurongroup_2_V;
__device__ double * brian::d_array_neurongroup_2_V;
const int brian::_num__array_neurongroup_2_V = 10;

int32_t * brian::_array_neurongroup_3_i;
int32_t * brian::dev_array_neurongroup_3_i;
__device__ int32_t * brian::d_array_neurongroup_3_i;
const int brian::_num__array_neurongroup_3_i = 10;

double * brian::_array_neurongroup_3_s_AHP_BS;
double * brian::dev_array_neurongroup_3_s_AHP_BS;
__device__ double * brian::d_array_neurongroup_3_s_AHP_BS;
const int brian::_num__array_neurongroup_3_s_AHP_BS = 10;

double * brian::_array_neurongroup_3_s_AMPA;
double * brian::dev_array_neurongroup_3_s_AMPA;
__device__ double * brian::d_array_neurongroup_3_s_AMPA;
const int brian::_num__array_neurongroup_3_s_AMPA = 10;

double * brian::_array_neurongroup_3_V;
double * brian::dev_array_neurongroup_3_V;
__device__ double * brian::d_array_neurongroup_3_V;
const int brian::_num__array_neurongroup_3_V = 10;

int32_t * brian::_array_neurongroup_i;
int32_t * brian::dev_array_neurongroup_i;
__device__ int32_t * brian::d_array_neurongroup_i;
const int brian::_num__array_neurongroup_i = 100;

double * brian::_array_neurongroup_s_ahp_GR;
double * brian::dev_array_neurongroup_s_ahp_GR;
__device__ double * brian::d_array_neurongroup_s_ahp_GR;
const int brian::_num__array_neurongroup_s_ahp_GR = 100;

double * brian::_array_neurongroup_s_AMPA;
double * brian::dev_array_neurongroup_s_AMPA;
__device__ double * brian::d_array_neurongroup_s_AMPA;
const int brian::_num__array_neurongroup_s_AMPA = 100;

double * brian::_array_neurongroup_s_GABA_1;
double * brian::dev_array_neurongroup_s_GABA_1;
__device__ double * brian::d_array_neurongroup_s_GABA_1;
const int brian::_num__array_neurongroup_s_GABA_1 = 100;

double * brian::_array_neurongroup_s_GABA_2;
double * brian::dev_array_neurongroup_s_GABA_2;
__device__ double * brian::d_array_neurongroup_s_GABA_2;
const int brian::_num__array_neurongroup_s_GABA_2 = 100;

double * brian::_array_neurongroup_s_NMDA;
double * brian::dev_array_neurongroup_s_NMDA;
__device__ double * brian::d_array_neurongroup_s_NMDA;
const int brian::_num__array_neurongroup_s_NMDA = 100;

double * brian::_array_neurongroup_V;
double * brian::dev_array_neurongroup_V;
__device__ double * brian::d_array_neurongroup_V;
const int brian::_num__array_neurongroup_V = 100;

double * brian::_array_neurongroup_x;
double * brian::dev_array_neurongroup_x;
__device__ double * brian::d_array_neurongroup_x;
const int brian::_num__array_neurongroup_x = 100;

double * brian::_array_neurongroup_y;
double * brian::dev_array_neurongroup_y;
__device__ double * brian::d_array_neurongroup_y;
const int brian::_num__array_neurongroup_y = 100;

int32_t * brian::_array_poissongroup_1_i;
int32_t * brian::dev_array_poissongroup_1_i;
__device__ int32_t * brian::d_array_poissongroup_1_i;
const int brian::_num__array_poissongroup_1_i = 10;

int32_t * brian::_array_poissongroup_i;
int32_t * brian::dev_array_poissongroup_i;
__device__ int32_t * brian::d_array_poissongroup_i;
const int brian::_num__array_poissongroup_i = 100;

int32_t * brian::_array_ratemonitor_1_N;
int32_t * brian::dev_array_ratemonitor_1_N;
__device__ int32_t * brian::d_array_ratemonitor_1_N;
const int brian::_num__array_ratemonitor_1_N = 1;

int32_t * brian::_array_ratemonitor_2_N;
int32_t * brian::dev_array_ratemonitor_2_N;
__device__ int32_t * brian::d_array_ratemonitor_2_N;
const int brian::_num__array_ratemonitor_2_N = 1;

int32_t * brian::_array_ratemonitor_3_N;
int32_t * brian::dev_array_ratemonitor_3_N;
__device__ int32_t * brian::d_array_ratemonitor_3_N;
const int brian::_num__array_ratemonitor_3_N = 1;

int32_t * brian::_array_ratemonitor_N;
int32_t * brian::dev_array_ratemonitor_N;
__device__ int32_t * brian::d_array_ratemonitor_N;
const int brian::_num__array_ratemonitor_N = 1;

int32_t * brian::_array_spikemonitor_1__source_idx;
int32_t * brian::dev_array_spikemonitor_1__source_idx;
__device__ int32_t * brian::d_array_spikemonitor_1__source_idx;
const int brian::_num__array_spikemonitor_1__source_idx = 10;

int32_t * brian::_array_spikemonitor_1_count;
int32_t * brian::dev_array_spikemonitor_1_count;
__device__ int32_t * brian::d_array_spikemonitor_1_count;
const int brian::_num__array_spikemonitor_1_count = 10;

int32_t * brian::_array_spikemonitor_1_N;
int32_t * brian::dev_array_spikemonitor_1_N;
__device__ int32_t * brian::d_array_spikemonitor_1_N;
const int brian::_num__array_spikemonitor_1_N = 1;

int32_t * brian::_array_spikemonitor_2__source_idx;
int32_t * brian::dev_array_spikemonitor_2__source_idx;
__device__ int32_t * brian::d_array_spikemonitor_2__source_idx;
const int brian::_num__array_spikemonitor_2__source_idx = 10;

int32_t * brian::_array_spikemonitor_2_count;
int32_t * brian::dev_array_spikemonitor_2_count;
__device__ int32_t * brian::d_array_spikemonitor_2_count;
const int brian::_num__array_spikemonitor_2_count = 10;

int32_t * brian::_array_spikemonitor_2_N;
int32_t * brian::dev_array_spikemonitor_2_N;
__device__ int32_t * brian::d_array_spikemonitor_2_N;
const int brian::_num__array_spikemonitor_2_N = 1;

int32_t * brian::_array_spikemonitor_3__source_idx;
int32_t * brian::dev_array_spikemonitor_3__source_idx;
__device__ int32_t * brian::d_array_spikemonitor_3__source_idx;
const int brian::_num__array_spikemonitor_3__source_idx = 10;

int32_t * brian::_array_spikemonitor_3_count;
int32_t * brian::dev_array_spikemonitor_3_count;
__device__ int32_t * brian::d_array_spikemonitor_3_count;
const int brian::_num__array_spikemonitor_3_count = 10;

int32_t * brian::_array_spikemonitor_3_N;
int32_t * brian::dev_array_spikemonitor_3_N;
__device__ int32_t * brian::d_array_spikemonitor_3_N;
const int brian::_num__array_spikemonitor_3_N = 1;

int32_t * brian::_array_spikemonitor__source_idx;
int32_t * brian::dev_array_spikemonitor__source_idx;
__device__ int32_t * brian::d_array_spikemonitor__source_idx;
const int brian::_num__array_spikemonitor__source_idx = 100;

int32_t * brian::_array_spikemonitor_count;
int32_t * brian::dev_array_spikemonitor_count;
__device__ int32_t * brian::d_array_spikemonitor_count;
const int brian::_num__array_spikemonitor_count = 100;

int32_t * brian::_array_spikemonitor_N;
int32_t * brian::dev_array_spikemonitor_N;
__device__ int32_t * brian::d_array_spikemonitor_N;
const int brian::_num__array_spikemonitor_N = 1;

int32_t * brian::_array_statemonitor_1__indices;
int32_t * brian::dev_array_statemonitor_1__indices;
__device__ int32_t * brian::d_array_statemonitor_1__indices;
const int brian::_num__array_statemonitor_1__indices = 10;

int32_t * brian::_array_statemonitor_1_N;
int32_t * brian::dev_array_statemonitor_1_N;
__device__ int32_t * brian::d_array_statemonitor_1_N;
const int brian::_num__array_statemonitor_1_N = 1;

double * brian::_array_statemonitor_1_V;
double * brian::dev_array_statemonitor_1_V;
__device__ double * brian::d_array_statemonitor_1_V;
const int brian::_num__array_statemonitor_1_V = (0, 10);

int32_t * brian::_array_statemonitor_2__indices;
int32_t * brian::dev_array_statemonitor_2__indices;
__device__ int32_t * brian::d_array_statemonitor_2__indices;
const int brian::_num__array_statemonitor_2__indices = 10;

int32_t * brian::_array_statemonitor_2_N;
int32_t * brian::dev_array_statemonitor_2_N;
__device__ int32_t * brian::d_array_statemonitor_2_N;
const int brian::_num__array_statemonitor_2_N = 1;

double * brian::_array_statemonitor_2_V;
double * brian::dev_array_statemonitor_2_V;
__device__ double * brian::d_array_statemonitor_2_V;
const int brian::_num__array_statemonitor_2_V = (0, 10);

int32_t * brian::_array_statemonitor_3__indices;
int32_t * brian::dev_array_statemonitor_3__indices;
__device__ int32_t * brian::d_array_statemonitor_3__indices;
const int brian::_num__array_statemonitor_3__indices = 10;

int32_t * brian::_array_statemonitor_3_N;
int32_t * brian::dev_array_statemonitor_3_N;
__device__ int32_t * brian::d_array_statemonitor_3_N;
const int brian::_num__array_statemonitor_3_N = 1;

double * brian::_array_statemonitor_3_V;
double * brian::dev_array_statemonitor_3_V;
__device__ double * brian::d_array_statemonitor_3_V;
const int brian::_num__array_statemonitor_3_V = (0, 10);

int32_t * brian::_array_statemonitor__indices;
int32_t * brian::dev_array_statemonitor__indices;
__device__ int32_t * brian::d_array_statemonitor__indices;
const int brian::_num__array_statemonitor__indices = 100;

int32_t * brian::_array_statemonitor_N;
int32_t * brian::dev_array_statemonitor_N;
__device__ int32_t * brian::d_array_statemonitor_N;
const int brian::_num__array_statemonitor_N = 1;

double * brian::_array_statemonitor_V;
double * brian::dev_array_statemonitor_V;
__device__ double * brian::d_array_statemonitor_V;
const int brian::_num__array_statemonitor_V = (0, 100);

int32_t * brian::_array_synapses_1_N;
int32_t * brian::dev_array_synapses_1_N;
__device__ int32_t * brian::d_array_synapses_1_N;
const int brian::_num__array_synapses_1_N = 1;

int32_t * brian::_array_synapses_2_N;
int32_t * brian::dev_array_synapses_2_N;
__device__ int32_t * brian::d_array_synapses_2_N;
const int brian::_num__array_synapses_2_N = 1;

int32_t * brian::_array_synapses_3_N;
int32_t * brian::dev_array_synapses_3_N;
__device__ int32_t * brian::d_array_synapses_3_N;
const int brian::_num__array_synapses_3_N = 1;

int32_t * brian::_array_synapses_4_N;
int32_t * brian::dev_array_synapses_4_N;
__device__ int32_t * brian::d_array_synapses_4_N;
const int brian::_num__array_synapses_4_N = 1;

int32_t * brian::_array_synapses_4_sources;
int32_t * brian::dev_array_synapses_4_sources;
__device__ int32_t * brian::d_array_synapses_4_sources;
const int brian::_num__array_synapses_4_sources = 10;

int32_t * brian::_array_synapses_4_sources_1;
int32_t * brian::dev_array_synapses_4_sources_1;
__device__ int32_t * brian::d_array_synapses_4_sources_1;
const int brian::_num__array_synapses_4_sources_1 = 10;

int32_t * brian::_array_synapses_4_sources_2;
int32_t * brian::dev_array_synapses_4_sources_2;
__device__ int32_t * brian::d_array_synapses_4_sources_2;
const int brian::_num__array_synapses_4_sources_2 = 10;

int32_t * brian::_array_synapses_4_sources_3;
int32_t * brian::dev_array_synapses_4_sources_3;
__device__ int32_t * brian::d_array_synapses_4_sources_3;
const int brian::_num__array_synapses_4_sources_3 = 10;

int32_t * brian::_array_synapses_4_sources_4;
int32_t * brian::dev_array_synapses_4_sources_4;
__device__ int32_t * brian::d_array_synapses_4_sources_4;
const int brian::_num__array_synapses_4_sources_4 = 10;

int32_t * brian::_array_synapses_4_sources_5;
int32_t * brian::dev_array_synapses_4_sources_5;
__device__ int32_t * brian::d_array_synapses_4_sources_5;
const int brian::_num__array_synapses_4_sources_5 = 10;

int32_t * brian::_array_synapses_4_sources_6;
int32_t * brian::dev_array_synapses_4_sources_6;
__device__ int32_t * brian::d_array_synapses_4_sources_6;
const int brian::_num__array_synapses_4_sources_6 = 10;

int32_t * brian::_array_synapses_4_sources_7;
int32_t * brian::dev_array_synapses_4_sources_7;
__device__ int32_t * brian::d_array_synapses_4_sources_7;
const int brian::_num__array_synapses_4_sources_7 = 10;

int32_t * brian::_array_synapses_4_sources_8;
int32_t * brian::dev_array_synapses_4_sources_8;
__device__ int32_t * brian::d_array_synapses_4_sources_8;
const int brian::_num__array_synapses_4_sources_8 = 10;

int32_t * brian::_array_synapses_4_sources_9;
int32_t * brian::dev_array_synapses_4_sources_9;
__device__ int32_t * brian::d_array_synapses_4_sources_9;
const int brian::_num__array_synapses_4_sources_9 = 10;

int32_t * brian::_array_synapses_4_targets;
int32_t * brian::dev_array_synapses_4_targets;
__device__ int32_t * brian::d_array_synapses_4_targets;
const int brian::_num__array_synapses_4_targets = 10;

int32_t * brian::_array_synapses_4_targets_1;
int32_t * brian::dev_array_synapses_4_targets_1;
__device__ int32_t * brian::d_array_synapses_4_targets_1;
const int brian::_num__array_synapses_4_targets_1 = 10;

int32_t * brian::_array_synapses_4_targets_2;
int32_t * brian::dev_array_synapses_4_targets_2;
__device__ int32_t * brian::d_array_synapses_4_targets_2;
const int brian::_num__array_synapses_4_targets_2 = 10;

int32_t * brian::_array_synapses_4_targets_3;
int32_t * brian::dev_array_synapses_4_targets_3;
__device__ int32_t * brian::d_array_synapses_4_targets_3;
const int brian::_num__array_synapses_4_targets_3 = 10;

int32_t * brian::_array_synapses_4_targets_4;
int32_t * brian::dev_array_synapses_4_targets_4;
__device__ int32_t * brian::d_array_synapses_4_targets_4;
const int brian::_num__array_synapses_4_targets_4 = 10;

int32_t * brian::_array_synapses_4_targets_5;
int32_t * brian::dev_array_synapses_4_targets_5;
__device__ int32_t * brian::d_array_synapses_4_targets_5;
const int brian::_num__array_synapses_4_targets_5 = 10;

int32_t * brian::_array_synapses_4_targets_6;
int32_t * brian::dev_array_synapses_4_targets_6;
__device__ int32_t * brian::d_array_synapses_4_targets_6;
const int brian::_num__array_synapses_4_targets_6 = 10;

int32_t * brian::_array_synapses_4_targets_7;
int32_t * brian::dev_array_synapses_4_targets_7;
__device__ int32_t * brian::d_array_synapses_4_targets_7;
const int brian::_num__array_synapses_4_targets_7 = 10;

int32_t * brian::_array_synapses_4_targets_8;
int32_t * brian::dev_array_synapses_4_targets_8;
__device__ int32_t * brian::d_array_synapses_4_targets_8;
const int brian::_num__array_synapses_4_targets_8 = 10;

int32_t * brian::_array_synapses_4_targets_9;
int32_t * brian::dev_array_synapses_4_targets_9;
__device__ int32_t * brian::d_array_synapses_4_targets_9;
const int brian::_num__array_synapses_4_targets_9 = 10;

int32_t * brian::_array_synapses_5_N;
int32_t * brian::dev_array_synapses_5_N;
__device__ int32_t * brian::d_array_synapses_5_N;
const int brian::_num__array_synapses_5_N = 1;

int32_t * brian::_array_synapses_5_sources;
int32_t * brian::dev_array_synapses_5_sources;
__device__ int32_t * brian::d_array_synapses_5_sources;
const int brian::_num__array_synapses_5_sources = 10;

int32_t * brian::_array_synapses_5_sources_1;
int32_t * brian::dev_array_synapses_5_sources_1;
__device__ int32_t * brian::d_array_synapses_5_sources_1;
const int brian::_num__array_synapses_5_sources_1 = 10;

int32_t * brian::_array_synapses_5_sources_2;
int32_t * brian::dev_array_synapses_5_sources_2;
__device__ int32_t * brian::d_array_synapses_5_sources_2;
const int brian::_num__array_synapses_5_sources_2 = 10;

int32_t * brian::_array_synapses_5_sources_3;
int32_t * brian::dev_array_synapses_5_sources_3;
__device__ int32_t * brian::d_array_synapses_5_sources_3;
const int brian::_num__array_synapses_5_sources_3 = 10;

int32_t * brian::_array_synapses_5_sources_4;
int32_t * brian::dev_array_synapses_5_sources_4;
__device__ int32_t * brian::d_array_synapses_5_sources_4;
const int brian::_num__array_synapses_5_sources_4 = 10;

int32_t * brian::_array_synapses_5_sources_5;
int32_t * brian::dev_array_synapses_5_sources_5;
__device__ int32_t * brian::d_array_synapses_5_sources_5;
const int brian::_num__array_synapses_5_sources_5 = 10;

int32_t * brian::_array_synapses_5_sources_6;
int32_t * brian::dev_array_synapses_5_sources_6;
__device__ int32_t * brian::d_array_synapses_5_sources_6;
const int brian::_num__array_synapses_5_sources_6 = 10;

int32_t * brian::_array_synapses_5_sources_7;
int32_t * brian::dev_array_synapses_5_sources_7;
__device__ int32_t * brian::d_array_synapses_5_sources_7;
const int brian::_num__array_synapses_5_sources_7 = 10;

int32_t * brian::_array_synapses_5_sources_8;
int32_t * brian::dev_array_synapses_5_sources_8;
__device__ int32_t * brian::d_array_synapses_5_sources_8;
const int brian::_num__array_synapses_5_sources_8 = 10;

int32_t * brian::_array_synapses_5_sources_9;
int32_t * brian::dev_array_synapses_5_sources_9;
__device__ int32_t * brian::d_array_synapses_5_sources_9;
const int brian::_num__array_synapses_5_sources_9 = 10;

int32_t * brian::_array_synapses_5_targets;
int32_t * brian::dev_array_synapses_5_targets;
__device__ int32_t * brian::d_array_synapses_5_targets;
const int brian::_num__array_synapses_5_targets = 10;

int32_t * brian::_array_synapses_5_targets_1;
int32_t * brian::dev_array_synapses_5_targets_1;
__device__ int32_t * brian::d_array_synapses_5_targets_1;
const int brian::_num__array_synapses_5_targets_1 = 10;

int32_t * brian::_array_synapses_5_targets_2;
int32_t * brian::dev_array_synapses_5_targets_2;
__device__ int32_t * brian::d_array_synapses_5_targets_2;
const int brian::_num__array_synapses_5_targets_2 = 10;

int32_t * brian::_array_synapses_5_targets_3;
int32_t * brian::dev_array_synapses_5_targets_3;
__device__ int32_t * brian::d_array_synapses_5_targets_3;
const int brian::_num__array_synapses_5_targets_3 = 10;

int32_t * brian::_array_synapses_5_targets_4;
int32_t * brian::dev_array_synapses_5_targets_4;
__device__ int32_t * brian::d_array_synapses_5_targets_4;
const int brian::_num__array_synapses_5_targets_4 = 10;

int32_t * brian::_array_synapses_5_targets_5;
int32_t * brian::dev_array_synapses_5_targets_5;
__device__ int32_t * brian::d_array_synapses_5_targets_5;
const int brian::_num__array_synapses_5_targets_5 = 10;

int32_t * brian::_array_synapses_5_targets_6;
int32_t * brian::dev_array_synapses_5_targets_6;
__device__ int32_t * brian::d_array_synapses_5_targets_6;
const int brian::_num__array_synapses_5_targets_6 = 10;

int32_t * brian::_array_synapses_5_targets_7;
int32_t * brian::dev_array_synapses_5_targets_7;
__device__ int32_t * brian::d_array_synapses_5_targets_7;
const int brian::_num__array_synapses_5_targets_7 = 10;

int32_t * brian::_array_synapses_5_targets_8;
int32_t * brian::dev_array_synapses_5_targets_8;
__device__ int32_t * brian::d_array_synapses_5_targets_8;
const int brian::_num__array_synapses_5_targets_8 = 10;

int32_t * brian::_array_synapses_5_targets_9;
int32_t * brian::dev_array_synapses_5_targets_9;
__device__ int32_t * brian::d_array_synapses_5_targets_9;
const int brian::_num__array_synapses_5_targets_9 = 10;

int32_t * brian::_array_synapses_6_N;
int32_t * brian::dev_array_synapses_6_N;
__device__ int32_t * brian::d_array_synapses_6_N;
const int brian::_num__array_synapses_6_N = 1;

int32_t * brian::_array_synapses_N;
int32_t * brian::dev_array_synapses_N;
__device__ int32_t * brian::d_array_synapses_N;
const int brian::_num__array_synapses_N = 1;


//////////////// eventspaces ///////////////
// we dynamically create multiple eventspaces in no_or_const_delay_mode
// for initiating the first spikespace, we need a host pointer
// for choosing the right spikespace, we need a global index variable
int32_t * brian::_array_neurongroup_1__spikespace;
const int brian::_num__array_neurongroup_1__spikespace = 11;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_1__spikespace(1);
int brian::current_idx_array_neurongroup_1__spikespace = 0;
int32_t * brian::_array_neurongroup_2__spikespace;
const int brian::_num__array_neurongroup_2__spikespace = 11;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_2__spikespace(1);
int brian::current_idx_array_neurongroup_2__spikespace = 0;
int32_t * brian::_array_neurongroup_3__spikespace;
const int brian::_num__array_neurongroup_3__spikespace = 11;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_3__spikespace(1);
int brian::current_idx_array_neurongroup_3__spikespace = 0;
int32_t * brian::_array_neurongroup__spikespace;
const int brian::_num__array_neurongroup__spikespace = 101;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup__spikespace(1);
int brian::current_idx_array_neurongroup__spikespace = 0;
int32_t * brian::_array_poissongroup_1__spikespace;
const int brian::_num__array_poissongroup_1__spikespace = 11;
thrust::host_vector<int32_t*> brian::dev_array_poissongroup_1__spikespace(1);
int brian::current_idx_array_poissongroup_1__spikespace = 0;
int32_t * brian::_array_poissongroup__spikespace;
const int brian::_num__array_poissongroup__spikespace = 101;
thrust::host_vector<int32_t*> brian::dev_array_poissongroup__spikespace(1);
int brian::current_idx_array_poissongroup__spikespace = 0;

//////////////// dynamic arrays 1d /////////
thrust::host_vector<double> brian::_dynamic_array_ratemonitor_1_rate;
thrust::device_vector<double> brian::dev_dynamic_array_ratemonitor_1_rate;
thrust::host_vector<double> brian::_dynamic_array_ratemonitor_1_t;
thrust::device_vector<double> brian::dev_dynamic_array_ratemonitor_1_t;
thrust::host_vector<double> brian::_dynamic_array_ratemonitor_2_rate;
thrust::device_vector<double> brian::dev_dynamic_array_ratemonitor_2_rate;
thrust::host_vector<double> brian::_dynamic_array_ratemonitor_2_t;
thrust::device_vector<double> brian::dev_dynamic_array_ratemonitor_2_t;
thrust::host_vector<double> brian::_dynamic_array_ratemonitor_3_rate;
thrust::device_vector<double> brian::dev_dynamic_array_ratemonitor_3_rate;
thrust::host_vector<double> brian::_dynamic_array_ratemonitor_3_t;
thrust::device_vector<double> brian::dev_dynamic_array_ratemonitor_3_t;
thrust::host_vector<double> brian::_dynamic_array_ratemonitor_rate;
thrust::device_vector<double> brian::dev_dynamic_array_ratemonitor_rate;
thrust::host_vector<double> brian::_dynamic_array_ratemonitor_t;
thrust::device_vector<double> brian::dev_dynamic_array_ratemonitor_t;
thrust::host_vector<int32_t> brian::_dynamic_array_spikemonitor_1_i;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikemonitor_1_i;
thrust::host_vector<double> brian::_dynamic_array_spikemonitor_1_t;
thrust::device_vector<double> brian::dev_dynamic_array_spikemonitor_1_t;
thrust::host_vector<int32_t> brian::_dynamic_array_spikemonitor_2_i;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikemonitor_2_i;
thrust::host_vector<double> brian::_dynamic_array_spikemonitor_2_t;
thrust::device_vector<double> brian::dev_dynamic_array_spikemonitor_2_t;
thrust::host_vector<int32_t> brian::_dynamic_array_spikemonitor_3_i;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikemonitor_3_i;
thrust::host_vector<double> brian::_dynamic_array_spikemonitor_3_t;
thrust::device_vector<double> brian::dev_dynamic_array_spikemonitor_3_t;
thrust::host_vector<int32_t> brian::_dynamic_array_spikemonitor_i;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikemonitor_i;
thrust::host_vector<double> brian::_dynamic_array_spikemonitor_t;
thrust::device_vector<double> brian::dev_dynamic_array_spikemonitor_t;
thrust::host_vector<double> brian::_dynamic_array_statemonitor_1_t;
thrust::device_vector<double> brian::dev_dynamic_array_statemonitor_1_t;
thrust::host_vector<double> brian::_dynamic_array_statemonitor_2_t;
thrust::device_vector<double> brian::dev_dynamic_array_statemonitor_2_t;
thrust::host_vector<double> brian::_dynamic_array_statemonitor_3_t;
thrust::device_vector<double> brian::dev_dynamic_array_statemonitor_3_t;
thrust::host_vector<double> brian::_dynamic_array_statemonitor_t;
thrust::device_vector<double> brian::dev_dynamic_array_statemonitor_t;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_delay;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_delay_1;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_delay_1;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_w_CFPKJ;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_w_CFPKJ;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_2_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_2_delay;
thrust::host_vector<double> brian::_dynamic_array_synapses_2_delay_1;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_2_delay_1;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_2_w_GRGO;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_2_w_GRGO;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_3__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_3__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_3__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_3__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_3_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_3_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_3_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_3_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_3_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_3_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_3_w_GOGR;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_3_w_GOGR;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_4__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_4__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_4__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_4__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_4_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_4_delay;
thrust::host_vector<double> brian::_dynamic_array_synapses_4_delay_1;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_4_delay_1;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_4_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_4_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_4_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_4_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_4_w_GRPKJ;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_4_w_GRPKJ;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_5__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_5__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_5__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_5__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_5_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_5_delay;
thrust::host_vector<double> brian::_dynamic_array_synapses_5_delay_1;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_5_delay_1;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_5_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_5_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_5_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_5_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_5_w_GRBS;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_5_w_GRBS;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_6__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_6__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_6__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_6__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_6_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_6_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_6_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_6_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_6_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_6_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_6_w_BSPKJ;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_6_w_BSPKJ;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_delay;
thrust::host_vector<double> brian::_dynamic_array_synapses_delay_1;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_delay_1;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_w_MFGR;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_w_MFGR;

//////////////// dynamic arrays 2d /////////
thrust::device_vector<double*> brian::addresses_monitor__dynamic_array_statemonitor_1_V;
thrust::device_vector<double>* brian::_dynamic_array_statemonitor_1_V;
thrust::device_vector<double*> brian::addresses_monitor__dynamic_array_statemonitor_2_V;
thrust::device_vector<double>* brian::_dynamic_array_statemonitor_2_V;
thrust::device_vector<double*> brian::addresses_monitor__dynamic_array_statemonitor_3_V;
thrust::device_vector<double>* brian::_dynamic_array_statemonitor_3_V;
thrust::device_vector<double*> brian::addresses_monitor__dynamic_array_statemonitor_V;
thrust::device_vector<double>* brian::_dynamic_array_statemonitor_V;

/////////////// static arrays /////////////
int32_t * brian::_static_array__array_statemonitor_1__indices;
int32_t * brian::dev_static_array__array_statemonitor_1__indices;
__device__ int32_t * brian::d_static_array__array_statemonitor_1__indices;
const int brian::_num__static_array__array_statemonitor_1__indices = 10;
int32_t * brian::_static_array__array_statemonitor_2__indices;
int32_t * brian::dev_static_array__array_statemonitor_2__indices;
__device__ int32_t * brian::d_static_array__array_statemonitor_2__indices;
const int brian::_num__static_array__array_statemonitor_2__indices = 10;
int32_t * brian::_static_array__array_statemonitor_3__indices;
int32_t * brian::dev_static_array__array_statemonitor_3__indices;
__device__ int32_t * brian::d_static_array__array_statemonitor_3__indices;
const int brian::_num__static_array__array_statemonitor_3__indices = 10;
int32_t * brian::_static_array__array_statemonitor__indices;
int32_t * brian::dev_static_array__array_statemonitor__indices;
__device__ int32_t * brian::d_static_array__array_statemonitor__indices;
const int brian::_num__static_array__array_statemonitor__indices = 100;
int32_t * brian::_static_array__array_synapses_4_sources;
int32_t * brian::dev_static_array__array_synapses_4_sources;
__device__ int32_t * brian::d_static_array__array_synapses_4_sources;
const int brian::_num__static_array__array_synapses_4_sources = 10;
int32_t * brian::_static_array__array_synapses_4_sources_1;
int32_t * brian::dev_static_array__array_synapses_4_sources_1;
__device__ int32_t * brian::d_static_array__array_synapses_4_sources_1;
const int brian::_num__static_array__array_synapses_4_sources_1 = 10;
int32_t * brian::_static_array__array_synapses_4_sources_2;
int32_t * brian::dev_static_array__array_synapses_4_sources_2;
__device__ int32_t * brian::d_static_array__array_synapses_4_sources_2;
const int brian::_num__static_array__array_synapses_4_sources_2 = 10;
int32_t * brian::_static_array__array_synapses_4_sources_3;
int32_t * brian::dev_static_array__array_synapses_4_sources_3;
__device__ int32_t * brian::d_static_array__array_synapses_4_sources_3;
const int brian::_num__static_array__array_synapses_4_sources_3 = 10;
int32_t * brian::_static_array__array_synapses_4_sources_4;
int32_t * brian::dev_static_array__array_synapses_4_sources_4;
__device__ int32_t * brian::d_static_array__array_synapses_4_sources_4;
const int brian::_num__static_array__array_synapses_4_sources_4 = 10;
int32_t * brian::_static_array__array_synapses_4_sources_5;
int32_t * brian::dev_static_array__array_synapses_4_sources_5;
__device__ int32_t * brian::d_static_array__array_synapses_4_sources_5;
const int brian::_num__static_array__array_synapses_4_sources_5 = 10;
int32_t * brian::_static_array__array_synapses_4_sources_6;
int32_t * brian::dev_static_array__array_synapses_4_sources_6;
__device__ int32_t * brian::d_static_array__array_synapses_4_sources_6;
const int brian::_num__static_array__array_synapses_4_sources_6 = 10;
int32_t * brian::_static_array__array_synapses_4_sources_7;
int32_t * brian::dev_static_array__array_synapses_4_sources_7;
__device__ int32_t * brian::d_static_array__array_synapses_4_sources_7;
const int brian::_num__static_array__array_synapses_4_sources_7 = 10;
int32_t * brian::_static_array__array_synapses_4_sources_8;
int32_t * brian::dev_static_array__array_synapses_4_sources_8;
__device__ int32_t * brian::d_static_array__array_synapses_4_sources_8;
const int brian::_num__static_array__array_synapses_4_sources_8 = 10;
int32_t * brian::_static_array__array_synapses_4_sources_9;
int32_t * brian::dev_static_array__array_synapses_4_sources_9;
__device__ int32_t * brian::d_static_array__array_synapses_4_sources_9;
const int brian::_num__static_array__array_synapses_4_sources_9 = 10;
int32_t * brian::_static_array__array_synapses_4_targets;
int32_t * brian::dev_static_array__array_synapses_4_targets;
__device__ int32_t * brian::d_static_array__array_synapses_4_targets;
const int brian::_num__static_array__array_synapses_4_targets = 10;
int32_t * brian::_static_array__array_synapses_4_targets_1;
int32_t * brian::dev_static_array__array_synapses_4_targets_1;
__device__ int32_t * brian::d_static_array__array_synapses_4_targets_1;
const int brian::_num__static_array__array_synapses_4_targets_1 = 10;
int32_t * brian::_static_array__array_synapses_4_targets_2;
int32_t * brian::dev_static_array__array_synapses_4_targets_2;
__device__ int32_t * brian::d_static_array__array_synapses_4_targets_2;
const int brian::_num__static_array__array_synapses_4_targets_2 = 10;
int32_t * brian::_static_array__array_synapses_4_targets_3;
int32_t * brian::dev_static_array__array_synapses_4_targets_3;
__device__ int32_t * brian::d_static_array__array_synapses_4_targets_3;
const int brian::_num__static_array__array_synapses_4_targets_3 = 10;
int32_t * brian::_static_array__array_synapses_4_targets_4;
int32_t * brian::dev_static_array__array_synapses_4_targets_4;
__device__ int32_t * brian::d_static_array__array_synapses_4_targets_4;
const int brian::_num__static_array__array_synapses_4_targets_4 = 10;
int32_t * brian::_static_array__array_synapses_4_targets_5;
int32_t * brian::dev_static_array__array_synapses_4_targets_5;
__device__ int32_t * brian::d_static_array__array_synapses_4_targets_5;
const int brian::_num__static_array__array_synapses_4_targets_5 = 10;
int32_t * brian::_static_array__array_synapses_4_targets_6;
int32_t * brian::dev_static_array__array_synapses_4_targets_6;
__device__ int32_t * brian::d_static_array__array_synapses_4_targets_6;
const int brian::_num__static_array__array_synapses_4_targets_6 = 10;
int32_t * brian::_static_array__array_synapses_4_targets_7;
int32_t * brian::dev_static_array__array_synapses_4_targets_7;
__device__ int32_t * brian::d_static_array__array_synapses_4_targets_7;
const int brian::_num__static_array__array_synapses_4_targets_7 = 10;
int32_t * brian::_static_array__array_synapses_4_targets_8;
int32_t * brian::dev_static_array__array_synapses_4_targets_8;
__device__ int32_t * brian::d_static_array__array_synapses_4_targets_8;
const int brian::_num__static_array__array_synapses_4_targets_8 = 10;
int32_t * brian::_static_array__array_synapses_4_targets_9;
int32_t * brian::dev_static_array__array_synapses_4_targets_9;
__device__ int32_t * brian::d_static_array__array_synapses_4_targets_9;
const int brian::_num__static_array__array_synapses_4_targets_9 = 10;
int32_t * brian::_static_array__array_synapses_5_sources;
int32_t * brian::dev_static_array__array_synapses_5_sources;
__device__ int32_t * brian::d_static_array__array_synapses_5_sources;
const int brian::_num__static_array__array_synapses_5_sources = 10;
int32_t * brian::_static_array__array_synapses_5_sources_1;
int32_t * brian::dev_static_array__array_synapses_5_sources_1;
__device__ int32_t * brian::d_static_array__array_synapses_5_sources_1;
const int brian::_num__static_array__array_synapses_5_sources_1 = 10;
int32_t * brian::_static_array__array_synapses_5_sources_2;
int32_t * brian::dev_static_array__array_synapses_5_sources_2;
__device__ int32_t * brian::d_static_array__array_synapses_5_sources_2;
const int brian::_num__static_array__array_synapses_5_sources_2 = 10;
int32_t * brian::_static_array__array_synapses_5_sources_3;
int32_t * brian::dev_static_array__array_synapses_5_sources_3;
__device__ int32_t * brian::d_static_array__array_synapses_5_sources_3;
const int brian::_num__static_array__array_synapses_5_sources_3 = 10;
int32_t * brian::_static_array__array_synapses_5_sources_4;
int32_t * brian::dev_static_array__array_synapses_5_sources_4;
__device__ int32_t * brian::d_static_array__array_synapses_5_sources_4;
const int brian::_num__static_array__array_synapses_5_sources_4 = 10;
int32_t * brian::_static_array__array_synapses_5_sources_5;
int32_t * brian::dev_static_array__array_synapses_5_sources_5;
__device__ int32_t * brian::d_static_array__array_synapses_5_sources_5;
const int brian::_num__static_array__array_synapses_5_sources_5 = 10;
int32_t * brian::_static_array__array_synapses_5_sources_6;
int32_t * brian::dev_static_array__array_synapses_5_sources_6;
__device__ int32_t * brian::d_static_array__array_synapses_5_sources_6;
const int brian::_num__static_array__array_synapses_5_sources_6 = 10;
int32_t * brian::_static_array__array_synapses_5_sources_7;
int32_t * brian::dev_static_array__array_synapses_5_sources_7;
__device__ int32_t * brian::d_static_array__array_synapses_5_sources_7;
const int brian::_num__static_array__array_synapses_5_sources_7 = 10;
int32_t * brian::_static_array__array_synapses_5_sources_8;
int32_t * brian::dev_static_array__array_synapses_5_sources_8;
__device__ int32_t * brian::d_static_array__array_synapses_5_sources_8;
const int brian::_num__static_array__array_synapses_5_sources_8 = 10;
int32_t * brian::_static_array__array_synapses_5_sources_9;
int32_t * brian::dev_static_array__array_synapses_5_sources_9;
__device__ int32_t * brian::d_static_array__array_synapses_5_sources_9;
const int brian::_num__static_array__array_synapses_5_sources_9 = 10;
int32_t * brian::_static_array__array_synapses_5_targets;
int32_t * brian::dev_static_array__array_synapses_5_targets;
__device__ int32_t * brian::d_static_array__array_synapses_5_targets;
const int brian::_num__static_array__array_synapses_5_targets = 10;
int32_t * brian::_static_array__array_synapses_5_targets_1;
int32_t * brian::dev_static_array__array_synapses_5_targets_1;
__device__ int32_t * brian::d_static_array__array_synapses_5_targets_1;
const int brian::_num__static_array__array_synapses_5_targets_1 = 10;
int32_t * brian::_static_array__array_synapses_5_targets_2;
int32_t * brian::dev_static_array__array_synapses_5_targets_2;
__device__ int32_t * brian::d_static_array__array_synapses_5_targets_2;
const int brian::_num__static_array__array_synapses_5_targets_2 = 10;
int32_t * brian::_static_array__array_synapses_5_targets_3;
int32_t * brian::dev_static_array__array_synapses_5_targets_3;
__device__ int32_t * brian::d_static_array__array_synapses_5_targets_3;
const int brian::_num__static_array__array_synapses_5_targets_3 = 10;
int32_t * brian::_static_array__array_synapses_5_targets_4;
int32_t * brian::dev_static_array__array_synapses_5_targets_4;
__device__ int32_t * brian::d_static_array__array_synapses_5_targets_4;
const int brian::_num__static_array__array_synapses_5_targets_4 = 10;
int32_t * brian::_static_array__array_synapses_5_targets_5;
int32_t * brian::dev_static_array__array_synapses_5_targets_5;
__device__ int32_t * brian::d_static_array__array_synapses_5_targets_5;
const int brian::_num__static_array__array_synapses_5_targets_5 = 10;
int32_t * brian::_static_array__array_synapses_5_targets_6;
int32_t * brian::dev_static_array__array_synapses_5_targets_6;
__device__ int32_t * brian::d_static_array__array_synapses_5_targets_6;
const int brian::_num__static_array__array_synapses_5_targets_6 = 10;
int32_t * brian::_static_array__array_synapses_5_targets_7;
int32_t * brian::dev_static_array__array_synapses_5_targets_7;
__device__ int32_t * brian::d_static_array__array_synapses_5_targets_7;
const int brian::_num__static_array__array_synapses_5_targets_7 = 10;
int32_t * brian::_static_array__array_synapses_5_targets_8;
int32_t * brian::dev_static_array__array_synapses_5_targets_8;
__device__ int32_t * brian::d_static_array__array_synapses_5_targets_8;
const int brian::_num__static_array__array_synapses_5_targets_8 = 10;
int32_t * brian::_static_array__array_synapses_5_targets_9;
int32_t * brian::dev_static_array__array_synapses_5_targets_9;
__device__ int32_t * brian::d_static_array__array_synapses_5_targets_9;
const int brian::_num__static_array__array_synapses_5_targets_9 = 10;

//////////////// synapses /////////////////
// synapses
int32_t synapses_source_start_index;
int32_t synapses_source_stop_index;
bool brian::synapses_multiple_pre_post = false;
// synapses_post
__device__ int* brian::synapses_post_num_synapses_by_pre;
__device__ int* brian::synapses_post_num_synapses_by_bundle;
__device__ int* brian::synapses_post_unique_delays;
__device__ int* brian::synapses_post_synapses_offset_by_bundle;
__device__ int* brian::synapses_post_global_bundle_id_start_by_pre;
int brian::synapses_post_max_bundle_size = 0;
int brian::synapses_post_mean_bundle_size = 0;
int brian::synapses_post_max_size = 0;
__device__ int* brian::synapses_post_num_unique_delays_by_pre;
int brian::synapses_post_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_post_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_post_synapse_ids;
__device__ int* brian::synapses_post_unique_delay_start_idcs;
__device__ int* brian::synapses_post_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_post;
int brian::synapses_post_eventspace_idx = 0;
int brian::synapses_post_delay;
bool brian::synapses_post_scalar_delay;
// synapses_pre
__device__ int* brian::synapses_pre_num_synapses_by_pre;
__device__ int* brian::synapses_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_pre_unique_delays;
__device__ int* brian::synapses_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_pre_global_bundle_id_start_by_pre;
int brian::synapses_pre_max_bundle_size = 0;
int brian::synapses_pre_mean_bundle_size = 0;
int brian::synapses_pre_max_size = 0;
__device__ int* brian::synapses_pre_num_unique_delays_by_pre;
int brian::synapses_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_pre_synapse_ids;
__device__ int* brian::synapses_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_pre;
int brian::synapses_pre_eventspace_idx = 0;
int brian::synapses_pre_delay;
bool brian::synapses_pre_scalar_delay;
// synapses_1
int32_t synapses_1_source_start_index;
int32_t synapses_1_source_stop_index;
bool brian::synapses_1_multiple_pre_post = false;
// synapses_1_post
__device__ int* brian::synapses_1_post_num_synapses_by_pre;
__device__ int* brian::synapses_1_post_num_synapses_by_bundle;
__device__ int* brian::synapses_1_post_unique_delays;
__device__ int* brian::synapses_1_post_synapses_offset_by_bundle;
__device__ int* brian::synapses_1_post_global_bundle_id_start_by_pre;
int brian::synapses_1_post_max_bundle_size = 0;
int brian::synapses_1_post_mean_bundle_size = 0;
int brian::synapses_1_post_max_size = 0;
__device__ int* brian::synapses_1_post_num_unique_delays_by_pre;
int brian::synapses_1_post_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_1_post_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_1_post_synapse_ids;
__device__ int* brian::synapses_1_post_unique_delay_start_idcs;
__device__ int* brian::synapses_1_post_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_1_post;
int brian::synapses_1_post_eventspace_idx = 0;
int brian::synapses_1_post_delay;
bool brian::synapses_1_post_scalar_delay;
// synapses_1_pre
__device__ int* brian::synapses_1_pre_num_synapses_by_pre;
__device__ int* brian::synapses_1_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_1_pre_unique_delays;
__device__ int* brian::synapses_1_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_1_pre_global_bundle_id_start_by_pre;
int brian::synapses_1_pre_max_bundle_size = 0;
int brian::synapses_1_pre_mean_bundle_size = 0;
int brian::synapses_1_pre_max_size = 0;
__device__ int* brian::synapses_1_pre_num_unique_delays_by_pre;
int brian::synapses_1_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_1_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_1_pre_synapse_ids;
__device__ int* brian::synapses_1_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_1_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_1_pre;
int brian::synapses_1_pre_eventspace_idx = 0;
int brian::synapses_1_pre_delay;
bool brian::synapses_1_pre_scalar_delay;
// synapses_2
int32_t synapses_2_source_start_index;
int32_t synapses_2_source_stop_index;
bool brian::synapses_2_multiple_pre_post = false;
// synapses_2_post
__device__ int* brian::synapses_2_post_num_synapses_by_pre;
__device__ int* brian::synapses_2_post_num_synapses_by_bundle;
__device__ int* brian::synapses_2_post_unique_delays;
__device__ int* brian::synapses_2_post_synapses_offset_by_bundle;
__device__ int* brian::synapses_2_post_global_bundle_id_start_by_pre;
int brian::synapses_2_post_max_bundle_size = 0;
int brian::synapses_2_post_mean_bundle_size = 0;
int brian::synapses_2_post_max_size = 0;
__device__ int* brian::synapses_2_post_num_unique_delays_by_pre;
int brian::synapses_2_post_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_2_post_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_2_post_synapse_ids;
__device__ int* brian::synapses_2_post_unique_delay_start_idcs;
__device__ int* brian::synapses_2_post_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_2_post;
int brian::synapses_2_post_eventspace_idx = 0;
int brian::synapses_2_post_delay;
bool brian::synapses_2_post_scalar_delay;
// synapses_2_pre
__device__ int* brian::synapses_2_pre_num_synapses_by_pre;
__device__ int* brian::synapses_2_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_2_pre_unique_delays;
__device__ int* brian::synapses_2_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_2_pre_global_bundle_id_start_by_pre;
int brian::synapses_2_pre_max_bundle_size = 0;
int brian::synapses_2_pre_mean_bundle_size = 0;
int brian::synapses_2_pre_max_size = 0;
__device__ int* brian::synapses_2_pre_num_unique_delays_by_pre;
int brian::synapses_2_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_2_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_2_pre_synapse_ids;
__device__ int* brian::synapses_2_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_2_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_2_pre;
int brian::synapses_2_pre_eventspace_idx = 0;
int brian::synapses_2_pre_delay;
bool brian::synapses_2_pre_scalar_delay;
// synapses_3
int32_t synapses_3_source_start_index;
int32_t synapses_3_source_stop_index;
bool brian::synapses_3_multiple_pre_post = false;
// synapses_3_pre
__device__ int* brian::synapses_3_pre_num_synapses_by_pre;
__device__ int* brian::synapses_3_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_3_pre_unique_delays;
__device__ int* brian::synapses_3_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_3_pre_global_bundle_id_start_by_pre;
int brian::synapses_3_pre_max_bundle_size = 0;
int brian::synapses_3_pre_mean_bundle_size = 0;
int brian::synapses_3_pre_max_size = 0;
__device__ int* brian::synapses_3_pre_num_unique_delays_by_pre;
int brian::synapses_3_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_3_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_3_pre_synapse_ids;
__device__ int* brian::synapses_3_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_3_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_3_pre;
int brian::synapses_3_pre_eventspace_idx = 0;
int brian::synapses_3_pre_delay;
bool brian::synapses_3_pre_scalar_delay;
// synapses_4
int32_t synapses_4_source_start_index;
int32_t synapses_4_source_stop_index;
bool brian::synapses_4_multiple_pre_post = false;
// synapses_4_post
__device__ int* brian::synapses_4_post_num_synapses_by_pre;
__device__ int* brian::synapses_4_post_num_synapses_by_bundle;
__device__ int* brian::synapses_4_post_unique_delays;
__device__ int* brian::synapses_4_post_synapses_offset_by_bundle;
__device__ int* brian::synapses_4_post_global_bundle_id_start_by_pre;
int brian::synapses_4_post_max_bundle_size = 0;
int brian::synapses_4_post_mean_bundle_size = 0;
int brian::synapses_4_post_max_size = 0;
__device__ int* brian::synapses_4_post_num_unique_delays_by_pre;
int brian::synapses_4_post_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_4_post_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_4_post_synapse_ids;
__device__ int* brian::synapses_4_post_unique_delay_start_idcs;
__device__ int* brian::synapses_4_post_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_4_post;
int brian::synapses_4_post_eventspace_idx = 0;
int brian::synapses_4_post_delay;
bool brian::synapses_4_post_scalar_delay;
// synapses_4_pre
__device__ int* brian::synapses_4_pre_num_synapses_by_pre;
__device__ int* brian::synapses_4_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_4_pre_unique_delays;
__device__ int* brian::synapses_4_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_4_pre_global_bundle_id_start_by_pre;
int brian::synapses_4_pre_max_bundle_size = 0;
int brian::synapses_4_pre_mean_bundle_size = 0;
int brian::synapses_4_pre_max_size = 0;
__device__ int* brian::synapses_4_pre_num_unique_delays_by_pre;
int brian::synapses_4_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_4_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_4_pre_synapse_ids;
__device__ int* brian::synapses_4_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_4_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_4_pre;
int brian::synapses_4_pre_eventspace_idx = 0;
int brian::synapses_4_pre_delay;
bool brian::synapses_4_pre_scalar_delay;
// synapses_5
int32_t synapses_5_source_start_index;
int32_t synapses_5_source_stop_index;
bool brian::synapses_5_multiple_pre_post = false;
// synapses_5_post
__device__ int* brian::synapses_5_post_num_synapses_by_pre;
__device__ int* brian::synapses_5_post_num_synapses_by_bundle;
__device__ int* brian::synapses_5_post_unique_delays;
__device__ int* brian::synapses_5_post_synapses_offset_by_bundle;
__device__ int* brian::synapses_5_post_global_bundle_id_start_by_pre;
int brian::synapses_5_post_max_bundle_size = 0;
int brian::synapses_5_post_mean_bundle_size = 0;
int brian::synapses_5_post_max_size = 0;
__device__ int* brian::synapses_5_post_num_unique_delays_by_pre;
int brian::synapses_5_post_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_5_post_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_5_post_synapse_ids;
__device__ int* brian::synapses_5_post_unique_delay_start_idcs;
__device__ int* brian::synapses_5_post_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_5_post;
int brian::synapses_5_post_eventspace_idx = 0;
int brian::synapses_5_post_delay;
bool brian::synapses_5_post_scalar_delay;
// synapses_5_pre
__device__ int* brian::synapses_5_pre_num_synapses_by_pre;
__device__ int* brian::synapses_5_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_5_pre_unique_delays;
__device__ int* brian::synapses_5_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_5_pre_global_bundle_id_start_by_pre;
int brian::synapses_5_pre_max_bundle_size = 0;
int brian::synapses_5_pre_mean_bundle_size = 0;
int brian::synapses_5_pre_max_size = 0;
__device__ int* brian::synapses_5_pre_num_unique_delays_by_pre;
int brian::synapses_5_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_5_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_5_pre_synapse_ids;
__device__ int* brian::synapses_5_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_5_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_5_pre;
int brian::synapses_5_pre_eventspace_idx = 0;
int brian::synapses_5_pre_delay;
bool brian::synapses_5_pre_scalar_delay;
// synapses_6
int32_t synapses_6_source_start_index;
int32_t synapses_6_source_stop_index;
bool brian::synapses_6_multiple_pre_post = false;
// synapses_6_pre
__device__ int* brian::synapses_6_pre_num_synapses_by_pre;
__device__ int* brian::synapses_6_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_6_pre_unique_delays;
__device__ int* brian::synapses_6_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_6_pre_global_bundle_id_start_by_pre;
int brian::synapses_6_pre_max_bundle_size = 0;
int brian::synapses_6_pre_mean_bundle_size = 0;
int brian::synapses_6_pre_max_size = 0;
__device__ int* brian::synapses_6_pre_num_unique_delays_by_pre;
int brian::synapses_6_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_6_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_6_pre_synapse_ids;
__device__ int* brian::synapses_6_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_6_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_6_pre;
int brian::synapses_6_pre_eventspace_idx = 0;
int brian::synapses_6_pre_delay;
bool brian::synapses_6_pre_scalar_delay;

int brian::num_parallel_blocks;
int brian::max_threads_per_block;
int brian::max_threads_per_sm;
int brian::max_shared_mem_size;
int brian::num_threads_per_warp;

__global__ void synapses_post_init(
                int Nsource,
                int Ntarget,
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t start,
                int32_t stop
        )
{
    using namespace brian;

    synapses_post.init(Nsource, Ntarget, sources, targets, dt, start, stop);
}
__global__ void synapses_pre_init(
                int Nsource,
                int Ntarget,
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t start,
                int32_t stop
        )
{
    using namespace brian;

    synapses_pre.init(Nsource, Ntarget, sources, targets, dt, start, stop);
}
__global__ void synapses_1_post_init(
                int Nsource,
                int Ntarget,
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t start,
                int32_t stop
        )
{
    using namespace brian;

    synapses_1_post.init(Nsource, Ntarget, sources, targets, dt, start, stop);
}
__global__ void synapses_1_pre_init(
                int Nsource,
                int Ntarget,
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t start,
                int32_t stop
        )
{
    using namespace brian;

    synapses_1_pre.init(Nsource, Ntarget, sources, targets, dt, start, stop);
}
__global__ void synapses_2_post_init(
                int Nsource,
                int Ntarget,
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t start,
                int32_t stop
        )
{
    using namespace brian;

    synapses_2_post.init(Nsource, Ntarget, sources, targets, dt, start, stop);
}
__global__ void synapses_2_pre_init(
                int Nsource,
                int Ntarget,
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t start,
                int32_t stop
        )
{
    using namespace brian;

    synapses_2_pre.init(Nsource, Ntarget, sources, targets, dt, start, stop);
}
__global__ void synapses_3_pre_init(
                int Nsource,
                int Ntarget,
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t start,
                int32_t stop
        )
{
    using namespace brian;

    synapses_3_pre.init(Nsource, Ntarget, sources, targets, dt, start, stop);
}
__global__ void synapses_4_post_init(
                int Nsource,
                int Ntarget,
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t start,
                int32_t stop
        )
{
    using namespace brian;

    synapses_4_post.init(Nsource, Ntarget, sources, targets, dt, start, stop);
}
__global__ void synapses_4_pre_init(
                int Nsource,
                int Ntarget,
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t start,
                int32_t stop
        )
{
    using namespace brian;

    synapses_4_pre.init(Nsource, Ntarget, sources, targets, dt, start, stop);
}
__global__ void synapses_5_post_init(
                int Nsource,
                int Ntarget,
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t start,
                int32_t stop
        )
{
    using namespace brian;

    synapses_5_post.init(Nsource, Ntarget, sources, targets, dt, start, stop);
}
__global__ void synapses_5_pre_init(
                int Nsource,
                int Ntarget,
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t start,
                int32_t stop
        )
{
    using namespace brian;

    synapses_5_pre.init(Nsource, Ntarget, sources, targets, dt, start, stop);
}
__global__ void synapses_6_pre_init(
                int Nsource,
                int Ntarget,
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t start,
                int32_t stop
        )
{
    using namespace brian;

    synapses_6_pre.init(Nsource, Ntarget, sources, targets, dt, start, stop);
}

// Profiling information for each code object

//////////////random numbers//////////////////
curandGenerator_t brian::curand_generator;
__device__ unsigned long long* brian::d_curand_seed;
unsigned long long* brian::dev_curand_seed;
// dev_{}_rand(n)_allocator
//      pointer to start of generated random numbers array
//      at each generation cycle this array is refilled
// dev_{}_rand(n)
//      pointer moving through generated random number array
//      until it is regenerated at the next generation cycle
randomNumber_t* brian::dev_poissongroup_1_thresholder_codeobject_rand_allocator;
randomNumber_t* brian::dev_poissongroup_1_thresholder_codeobject_rand;
__device__ randomNumber_t* brian::_array_poissongroup_1_thresholder_codeobject_rand;
randomNumber_t* brian::dev_poissongroup_thresholder_codeobject_rand_allocator;
randomNumber_t* brian::dev_poissongroup_thresholder_codeobject_rand;
__device__ randomNumber_t* brian::_array_poissongroup_thresholder_codeobject_rand;
curandState* brian::dev_curand_states;
__device__ curandState* brian::d_curand_states;
RandomNumberBuffer brian::random_number_buffer;

void _init_arrays()
{
    using namespace brian;

    std::clock_t start_timer = std::clock();

    CUDA_CHECK_MEMORY();
    size_t used_device_memory_start = used_device_memory;

    cudaDeviceProp props;
    CUDA_SAFE_CALL(
            cudaGetDeviceProperties(&props, 0)
            );

    num_parallel_blocks = props.multiProcessorCount * 1;
    printf("objects cu num par blocks %d\n", num_parallel_blocks);
    max_threads_per_block = props.maxThreadsPerBlock;
    max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    max_shared_mem_size = props.sharedMemPerBlock;
    num_threads_per_warp = props.warpSize;

    // Random seeds might be overwritten in main.cu
    unsigned long long seed = time(0);

    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_curand_seed,
                sizeof(unsigned long long))
            );

    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_curand_seed, &dev_curand_seed,
                sizeof(unsigned long long*))
            );

    curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT);

    // this sets seed for host and device api RNG
    random_number_buffer.set_seed(seed);

    synapses_post_init<<<1,1>>>(
            100,
            100,
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_post[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_pre[0]),
            0,  //was dt, maybe irrelevant?
            0,
            100
            );
    CUDA_CHECK_ERROR("synapses_post_init");
    synapses_pre_init<<<1,1>>>(
            100,
            100,
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            100
            );
    CUDA_CHECK_ERROR("synapses_pre_init");
    synapses_1_post_init<<<1,1>>>(
            10,
            10,
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_post[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_pre[0]),
            0,  //was dt, maybe irrelevant?
            0,
            10
            );
    CUDA_CHECK_ERROR("synapses_1_post_init");
    synapses_1_pre_init<<<1,1>>>(
            10,
            10,
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            10
            );
    CUDA_CHECK_ERROR("synapses_1_pre_init");
    synapses_2_post_init<<<1,1>>>(
            10,
            100,
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2__synaptic_post[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2__synaptic_pre[0]),
            0,  //was dt, maybe irrelevant?
            0,
            10
            );
    CUDA_CHECK_ERROR("synapses_2_post_init");
    synapses_2_pre_init<<<1,1>>>(
            100,
            10,
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            100
            );
    CUDA_CHECK_ERROR("synapses_2_pre_init");
    synapses_3_pre_init<<<1,1>>>(
            10,
            100,
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_3__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_3__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            10
            );
    CUDA_CHECK_ERROR("synapses_3_pre_init");
    synapses_4_post_init<<<1,1>>>(
            10,
            100,
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_4__synaptic_post[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_4__synaptic_pre[0]),
            0,  //was dt, maybe irrelevant?
            0,
            10
            );
    CUDA_CHECK_ERROR("synapses_4_post_init");
    synapses_4_pre_init<<<1,1>>>(
            100,
            10,
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_4__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_4__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            100
            );
    CUDA_CHECK_ERROR("synapses_4_pre_init");
    synapses_5_post_init<<<1,1>>>(
            10,
            100,
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_5__synaptic_post[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_5__synaptic_pre[0]),
            0,  //was dt, maybe irrelevant?
            0,
            10
            );
    CUDA_CHECK_ERROR("synapses_5_post_init");
    synapses_5_pre_init<<<1,1>>>(
            100,
            10,
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_5__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_5__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            100
            );
    CUDA_CHECK_ERROR("synapses_5_pre_init");
    synapses_6_pre_init<<<1,1>>>(
            10,
            10,
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_6__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_6__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            10
            );
    CUDA_CHECK_ERROR("synapses_6_pre_init");

    // Arrays initialized to 0
            _array_defaultclock_dt = new double[1];
            for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_dt, _array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt, cudaMemcpyHostToDevice)
                    );
            _array_defaultclock_t = new double[1];
            for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_t, _array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t, cudaMemcpyHostToDevice)
                    );
            _array_defaultclock_timestep = new int64_t[1];
            for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_timestep, sizeof(int64_t)*_num__array_defaultclock_timestep)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_timestep, _array_defaultclock_timestep, sizeof(int64_t)*_num__array_defaultclock_timestep, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_i = new int32_t[10];
            for(int i=0; i<10; i++) _array_neurongroup_1_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_i, _array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_s_ahp_GO = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_1_s_ahp_GO[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_s_ahp_GO, sizeof(double)*_num__array_neurongroup_1_s_ahp_GO)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_s_ahp_GO, _array_neurongroup_1_s_ahp_GO, sizeof(double)*_num__array_neurongroup_1_s_ahp_GO, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_s_AMPA = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_1_s_AMPA[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_s_AMPA, sizeof(double)*_num__array_neurongroup_1_s_AMPA)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_s_AMPA, _array_neurongroup_1_s_AMPA, sizeof(double)*_num__array_neurongroup_1_s_AMPA, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_s_NMDA_1 = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_1_s_NMDA_1[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_s_NMDA_1, sizeof(double)*_num__array_neurongroup_1_s_NMDA_1)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_s_NMDA_1, _array_neurongroup_1_s_NMDA_1, sizeof(double)*_num__array_neurongroup_1_s_NMDA_1, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_s_NMDA_2 = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_1_s_NMDA_2[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_s_NMDA_2, sizeof(double)*_num__array_neurongroup_1_s_NMDA_2)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_s_NMDA_2, _array_neurongroup_1_s_NMDA_2, sizeof(double)*_num__array_neurongroup_1_s_NMDA_2, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_V = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_1_V[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_V, sizeof(double)*_num__array_neurongroup_1_V)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_V, _array_neurongroup_1_V, sizeof(double)*_num__array_neurongroup_1_V, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_x = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_1_x[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_x, sizeof(double)*_num__array_neurongroup_1_x)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_x, _array_neurongroup_1_x, sizeof(double)*_num__array_neurongroup_1_x, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_y = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_1_y[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_y, sizeof(double)*_num__array_neurongroup_1_y)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_y, _array_neurongroup_1_y, sizeof(double)*_num__array_neurongroup_1_y, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_i = new int32_t[10];
            for(int i=0; i<10; i++) _array_neurongroup_2_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_i, sizeof(int32_t)*_num__array_neurongroup_2_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_i, _array_neurongroup_2_i, sizeof(int32_t)*_num__array_neurongroup_2_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_s_AHP_PKJ = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_2_s_AHP_PKJ[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_s_AHP_PKJ, sizeof(double)*_num__array_neurongroup_2_s_AHP_PKJ)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_s_AHP_PKJ, _array_neurongroup_2_s_AHP_PKJ, sizeof(double)*_num__array_neurongroup_2_s_AHP_PKJ, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_s_AMPA = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_2_s_AMPA[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_s_AMPA, sizeof(double)*_num__array_neurongroup_2_s_AMPA)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_s_AMPA, _array_neurongroup_2_s_AMPA, sizeof(double)*_num__array_neurongroup_2_s_AMPA, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_s_GABA = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_2_s_GABA[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_s_GABA, sizeof(double)*_num__array_neurongroup_2_s_GABA)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_s_GABA, _array_neurongroup_2_s_GABA, sizeof(double)*_num__array_neurongroup_2_s_GABA, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_V = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_2_V[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_V, sizeof(double)*_num__array_neurongroup_2_V)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_V, _array_neurongroup_2_V, sizeof(double)*_num__array_neurongroup_2_V, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_3_i = new int32_t[10];
            for(int i=0; i<10; i++) _array_neurongroup_3_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_3_i, sizeof(int32_t)*_num__array_neurongroup_3_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_3_i, _array_neurongroup_3_i, sizeof(int32_t)*_num__array_neurongroup_3_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_3_s_AHP_BS = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_3_s_AHP_BS[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_3_s_AHP_BS, sizeof(double)*_num__array_neurongroup_3_s_AHP_BS)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_3_s_AHP_BS, _array_neurongroup_3_s_AHP_BS, sizeof(double)*_num__array_neurongroup_3_s_AHP_BS, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_3_s_AMPA = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_3_s_AMPA[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_3_s_AMPA, sizeof(double)*_num__array_neurongroup_3_s_AMPA)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_3_s_AMPA, _array_neurongroup_3_s_AMPA, sizeof(double)*_num__array_neurongroup_3_s_AMPA, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_3_V = new double[10];
            for(int i=0; i<10; i++) _array_neurongroup_3_V[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_3_V, sizeof(double)*_num__array_neurongroup_3_V)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_3_V, _array_neurongroup_3_V, sizeof(double)*_num__array_neurongroup_3_V, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_i = new int32_t[100];
            for(int i=0; i<100; i++) _array_neurongroup_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_s_ahp_GR = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_s_ahp_GR[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_s_ahp_GR, sizeof(double)*_num__array_neurongroup_s_ahp_GR)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_s_ahp_GR, _array_neurongroup_s_ahp_GR, sizeof(double)*_num__array_neurongroup_s_ahp_GR, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_s_AMPA = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_s_AMPA[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_s_AMPA, sizeof(double)*_num__array_neurongroup_s_AMPA)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_s_AMPA, _array_neurongroup_s_AMPA, sizeof(double)*_num__array_neurongroup_s_AMPA, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_s_GABA_1 = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_s_GABA_1[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_s_GABA_1, sizeof(double)*_num__array_neurongroup_s_GABA_1)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_s_GABA_1, _array_neurongroup_s_GABA_1, sizeof(double)*_num__array_neurongroup_s_GABA_1, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_s_GABA_2 = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_s_GABA_2[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_s_GABA_2, sizeof(double)*_num__array_neurongroup_s_GABA_2)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_s_GABA_2, _array_neurongroup_s_GABA_2, sizeof(double)*_num__array_neurongroup_s_GABA_2, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_s_NMDA = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_s_NMDA[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_s_NMDA, sizeof(double)*_num__array_neurongroup_s_NMDA)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_s_NMDA, _array_neurongroup_s_NMDA, sizeof(double)*_num__array_neurongroup_s_NMDA, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_V = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_V[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_V, sizeof(double)*_num__array_neurongroup_V)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_V, _array_neurongroup_V, sizeof(double)*_num__array_neurongroup_V, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_x = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_x[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_x, sizeof(double)*_num__array_neurongroup_x)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_x, _array_neurongroup_x, sizeof(double)*_num__array_neurongroup_x, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_y = new double[100];
            for(int i=0; i<100; i++) _array_neurongroup_y[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_y, sizeof(double)*_num__array_neurongroup_y)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_y, _array_neurongroup_y, sizeof(double)*_num__array_neurongroup_y, cudaMemcpyHostToDevice)
                    );
            _array_poissongroup_1_i = new int32_t[10];
            for(int i=0; i<10; i++) _array_poissongroup_1_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_poissongroup_1_i, sizeof(int32_t)*_num__array_poissongroup_1_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_poissongroup_1_i, _array_poissongroup_1_i, sizeof(int32_t)*_num__array_poissongroup_1_i, cudaMemcpyHostToDevice)
                    );
            _array_poissongroup_i = new int32_t[100];
            for(int i=0; i<100; i++) _array_poissongroup_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_poissongroup_i, sizeof(int32_t)*_num__array_poissongroup_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_poissongroup_i, _array_poissongroup_i, sizeof(int32_t)*_num__array_poissongroup_i, cudaMemcpyHostToDevice)
                    );
            _array_ratemonitor_1_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_ratemonitor_1_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_ratemonitor_1_N, sizeof(int32_t)*_num__array_ratemonitor_1_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_ratemonitor_1_N, _array_ratemonitor_1_N, sizeof(int32_t)*_num__array_ratemonitor_1_N, cudaMemcpyHostToDevice)
                    );
            _array_ratemonitor_2_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_ratemonitor_2_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_ratemonitor_2_N, sizeof(int32_t)*_num__array_ratemonitor_2_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_ratemonitor_2_N, _array_ratemonitor_2_N, sizeof(int32_t)*_num__array_ratemonitor_2_N, cudaMemcpyHostToDevice)
                    );
            _array_ratemonitor_3_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_ratemonitor_3_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_ratemonitor_3_N, sizeof(int32_t)*_num__array_ratemonitor_3_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_ratemonitor_3_N, _array_ratemonitor_3_N, sizeof(int32_t)*_num__array_ratemonitor_3_N, cudaMemcpyHostToDevice)
                    );
            _array_ratemonitor_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_ratemonitor_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_ratemonitor_N, sizeof(int32_t)*_num__array_ratemonitor_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_ratemonitor_N, _array_ratemonitor_N, sizeof(int32_t)*_num__array_ratemonitor_N, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_1__source_idx = new int32_t[10];
            for(int i=0; i<10; i++) _array_spikemonitor_1__source_idx[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_1__source_idx, sizeof(int32_t)*_num__array_spikemonitor_1__source_idx)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_1__source_idx, _array_spikemonitor_1__source_idx, sizeof(int32_t)*_num__array_spikemonitor_1__source_idx, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_1_count = new int32_t[10];
            for(int i=0; i<10; i++) _array_spikemonitor_1_count[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_1_count, sizeof(int32_t)*_num__array_spikemonitor_1_count)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_1_count, _array_spikemonitor_1_count, sizeof(int32_t)*_num__array_spikemonitor_1_count, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_1_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikemonitor_1_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_1_N, sizeof(int32_t)*_num__array_spikemonitor_1_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_1_N, _array_spikemonitor_1_N, sizeof(int32_t)*_num__array_spikemonitor_1_N, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_2__source_idx = new int32_t[10];
            for(int i=0; i<10; i++) _array_spikemonitor_2__source_idx[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_2__source_idx, sizeof(int32_t)*_num__array_spikemonitor_2__source_idx)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_2__source_idx, _array_spikemonitor_2__source_idx, sizeof(int32_t)*_num__array_spikemonitor_2__source_idx, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_2_count = new int32_t[10];
            for(int i=0; i<10; i++) _array_spikemonitor_2_count[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_2_count, sizeof(int32_t)*_num__array_spikemonitor_2_count)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_2_count, _array_spikemonitor_2_count, sizeof(int32_t)*_num__array_spikemonitor_2_count, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_2_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikemonitor_2_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_2_N, sizeof(int32_t)*_num__array_spikemonitor_2_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_2_N, _array_spikemonitor_2_N, sizeof(int32_t)*_num__array_spikemonitor_2_N, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_3__source_idx = new int32_t[10];
            for(int i=0; i<10; i++) _array_spikemonitor_3__source_idx[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_3__source_idx, sizeof(int32_t)*_num__array_spikemonitor_3__source_idx)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_3__source_idx, _array_spikemonitor_3__source_idx, sizeof(int32_t)*_num__array_spikemonitor_3__source_idx, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_3_count = new int32_t[10];
            for(int i=0; i<10; i++) _array_spikemonitor_3_count[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_3_count, sizeof(int32_t)*_num__array_spikemonitor_3_count)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_3_count, _array_spikemonitor_3_count, sizeof(int32_t)*_num__array_spikemonitor_3_count, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_3_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikemonitor_3_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_3_N, sizeof(int32_t)*_num__array_spikemonitor_3_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_3_N, _array_spikemonitor_3_N, sizeof(int32_t)*_num__array_spikemonitor_3_N, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor__source_idx = new int32_t[100];
            for(int i=0; i<100; i++) _array_spikemonitor__source_idx[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor__source_idx, _array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_count = new int32_t[100];
            for(int i=0; i<100; i++) _array_spikemonitor_count[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_count, sizeof(int32_t)*_num__array_spikemonitor_count)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_count, _array_spikemonitor_count, sizeof(int32_t)*_num__array_spikemonitor_count, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikemonitor_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_N, _array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N, cudaMemcpyHostToDevice)
                    );
            _array_statemonitor_1__indices = new int32_t[10];
            for(int i=0; i<10; i++) _array_statemonitor_1__indices[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_statemonitor_1__indices, sizeof(int32_t)*_num__array_statemonitor_1__indices)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_statemonitor_1__indices, _array_statemonitor_1__indices, sizeof(int32_t)*_num__array_statemonitor_1__indices, cudaMemcpyHostToDevice)
                    );
            _array_statemonitor_1_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_statemonitor_1_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_statemonitor_1_N, sizeof(int32_t)*_num__array_statemonitor_1_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_statemonitor_1_N, _array_statemonitor_1_N, sizeof(int32_t)*_num__array_statemonitor_1_N, cudaMemcpyHostToDevice)
                    );
            _array_statemonitor_2__indices = new int32_t[10];
            for(int i=0; i<10; i++) _array_statemonitor_2__indices[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_statemonitor_2__indices, sizeof(int32_t)*_num__array_statemonitor_2__indices)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_statemonitor_2__indices, _array_statemonitor_2__indices, sizeof(int32_t)*_num__array_statemonitor_2__indices, cudaMemcpyHostToDevice)
                    );
            _array_statemonitor_2_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_statemonitor_2_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_statemonitor_2_N, sizeof(int32_t)*_num__array_statemonitor_2_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_statemonitor_2_N, _array_statemonitor_2_N, sizeof(int32_t)*_num__array_statemonitor_2_N, cudaMemcpyHostToDevice)
                    );
            _array_statemonitor_3__indices = new int32_t[10];
            for(int i=0; i<10; i++) _array_statemonitor_3__indices[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_statemonitor_3__indices, sizeof(int32_t)*_num__array_statemonitor_3__indices)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_statemonitor_3__indices, _array_statemonitor_3__indices, sizeof(int32_t)*_num__array_statemonitor_3__indices, cudaMemcpyHostToDevice)
                    );
            _array_statemonitor_3_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_statemonitor_3_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_statemonitor_3_N, sizeof(int32_t)*_num__array_statemonitor_3_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_statemonitor_3_N, _array_statemonitor_3_N, sizeof(int32_t)*_num__array_statemonitor_3_N, cudaMemcpyHostToDevice)
                    );
            _array_statemonitor__indices = new int32_t[100];
            for(int i=0; i<100; i++) _array_statemonitor__indices[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_statemonitor__indices, sizeof(int32_t)*_num__array_statemonitor__indices)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_statemonitor__indices, _array_statemonitor__indices, sizeof(int32_t)*_num__array_statemonitor__indices, cudaMemcpyHostToDevice)
                    );
            _array_statemonitor_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_statemonitor_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_statemonitor_N, sizeof(int32_t)*_num__array_statemonitor_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_statemonitor_N, _array_statemonitor_N, sizeof(int32_t)*_num__array_statemonitor_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_1_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_1_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_1_N, _array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_2_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_2_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_2_N, sizeof(int32_t)*_num__array_synapses_2_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_2_N, _array_synapses_2_N, sizeof(int32_t)*_num__array_synapses_2_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_3_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_3_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_3_N, sizeof(int32_t)*_num__array_synapses_3_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_3_N, _array_synapses_3_N, sizeof(int32_t)*_num__array_synapses_3_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_4_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_N, sizeof(int32_t)*_num__array_synapses_4_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_N, _array_synapses_4_N, sizeof(int32_t)*_num__array_synapses_4_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_sources = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_sources[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_sources, sizeof(int32_t)*_num__array_synapses_4_sources)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_sources, _array_synapses_4_sources, sizeof(int32_t)*_num__array_synapses_4_sources, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_sources_1 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_sources_1[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_sources_1, sizeof(int32_t)*_num__array_synapses_4_sources_1)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_sources_1, _array_synapses_4_sources_1, sizeof(int32_t)*_num__array_synapses_4_sources_1, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_sources_2 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_sources_2[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_sources_2, sizeof(int32_t)*_num__array_synapses_4_sources_2)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_sources_2, _array_synapses_4_sources_2, sizeof(int32_t)*_num__array_synapses_4_sources_2, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_sources_3 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_sources_3[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_sources_3, sizeof(int32_t)*_num__array_synapses_4_sources_3)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_sources_3, _array_synapses_4_sources_3, sizeof(int32_t)*_num__array_synapses_4_sources_3, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_sources_4 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_sources_4[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_sources_4, sizeof(int32_t)*_num__array_synapses_4_sources_4)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_sources_4, _array_synapses_4_sources_4, sizeof(int32_t)*_num__array_synapses_4_sources_4, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_sources_5 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_sources_5[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_sources_5, sizeof(int32_t)*_num__array_synapses_4_sources_5)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_sources_5, _array_synapses_4_sources_5, sizeof(int32_t)*_num__array_synapses_4_sources_5, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_sources_6 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_sources_6[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_sources_6, sizeof(int32_t)*_num__array_synapses_4_sources_6)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_sources_6, _array_synapses_4_sources_6, sizeof(int32_t)*_num__array_synapses_4_sources_6, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_sources_7 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_sources_7[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_sources_7, sizeof(int32_t)*_num__array_synapses_4_sources_7)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_sources_7, _array_synapses_4_sources_7, sizeof(int32_t)*_num__array_synapses_4_sources_7, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_sources_8 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_sources_8[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_sources_8, sizeof(int32_t)*_num__array_synapses_4_sources_8)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_sources_8, _array_synapses_4_sources_8, sizeof(int32_t)*_num__array_synapses_4_sources_8, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_sources_9 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_sources_9[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_sources_9, sizeof(int32_t)*_num__array_synapses_4_sources_9)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_sources_9, _array_synapses_4_sources_9, sizeof(int32_t)*_num__array_synapses_4_sources_9, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_targets = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_targets[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_targets, sizeof(int32_t)*_num__array_synapses_4_targets)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_targets, _array_synapses_4_targets, sizeof(int32_t)*_num__array_synapses_4_targets, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_targets_1 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_targets_1[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_targets_1, sizeof(int32_t)*_num__array_synapses_4_targets_1)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_targets_1, _array_synapses_4_targets_1, sizeof(int32_t)*_num__array_synapses_4_targets_1, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_targets_2 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_targets_2[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_targets_2, sizeof(int32_t)*_num__array_synapses_4_targets_2)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_targets_2, _array_synapses_4_targets_2, sizeof(int32_t)*_num__array_synapses_4_targets_2, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_targets_3 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_targets_3[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_targets_3, sizeof(int32_t)*_num__array_synapses_4_targets_3)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_targets_3, _array_synapses_4_targets_3, sizeof(int32_t)*_num__array_synapses_4_targets_3, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_targets_4 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_targets_4[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_targets_4, sizeof(int32_t)*_num__array_synapses_4_targets_4)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_targets_4, _array_synapses_4_targets_4, sizeof(int32_t)*_num__array_synapses_4_targets_4, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_targets_5 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_targets_5[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_targets_5, sizeof(int32_t)*_num__array_synapses_4_targets_5)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_targets_5, _array_synapses_4_targets_5, sizeof(int32_t)*_num__array_synapses_4_targets_5, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_targets_6 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_targets_6[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_targets_6, sizeof(int32_t)*_num__array_synapses_4_targets_6)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_targets_6, _array_synapses_4_targets_6, sizeof(int32_t)*_num__array_synapses_4_targets_6, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_targets_7 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_targets_7[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_targets_7, sizeof(int32_t)*_num__array_synapses_4_targets_7)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_targets_7, _array_synapses_4_targets_7, sizeof(int32_t)*_num__array_synapses_4_targets_7, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_targets_8 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_targets_8[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_targets_8, sizeof(int32_t)*_num__array_synapses_4_targets_8)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_targets_8, _array_synapses_4_targets_8, sizeof(int32_t)*_num__array_synapses_4_targets_8, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_targets_9 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_4_targets_9[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_targets_9, sizeof(int32_t)*_num__array_synapses_4_targets_9)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_targets_9, _array_synapses_4_targets_9, sizeof(int32_t)*_num__array_synapses_4_targets_9, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_5_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_N, sizeof(int32_t)*_num__array_synapses_5_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_N, _array_synapses_5_N, sizeof(int32_t)*_num__array_synapses_5_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_sources = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_sources[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_sources, sizeof(int32_t)*_num__array_synapses_5_sources)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_sources, _array_synapses_5_sources, sizeof(int32_t)*_num__array_synapses_5_sources, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_sources_1 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_sources_1[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_sources_1, sizeof(int32_t)*_num__array_synapses_5_sources_1)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_sources_1, _array_synapses_5_sources_1, sizeof(int32_t)*_num__array_synapses_5_sources_1, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_sources_2 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_sources_2[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_sources_2, sizeof(int32_t)*_num__array_synapses_5_sources_2)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_sources_2, _array_synapses_5_sources_2, sizeof(int32_t)*_num__array_synapses_5_sources_2, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_sources_3 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_sources_3[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_sources_3, sizeof(int32_t)*_num__array_synapses_5_sources_3)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_sources_3, _array_synapses_5_sources_3, sizeof(int32_t)*_num__array_synapses_5_sources_3, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_sources_4 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_sources_4[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_sources_4, sizeof(int32_t)*_num__array_synapses_5_sources_4)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_sources_4, _array_synapses_5_sources_4, sizeof(int32_t)*_num__array_synapses_5_sources_4, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_sources_5 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_sources_5[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_sources_5, sizeof(int32_t)*_num__array_synapses_5_sources_5)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_sources_5, _array_synapses_5_sources_5, sizeof(int32_t)*_num__array_synapses_5_sources_5, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_sources_6 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_sources_6[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_sources_6, sizeof(int32_t)*_num__array_synapses_5_sources_6)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_sources_6, _array_synapses_5_sources_6, sizeof(int32_t)*_num__array_synapses_5_sources_6, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_sources_7 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_sources_7[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_sources_7, sizeof(int32_t)*_num__array_synapses_5_sources_7)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_sources_7, _array_synapses_5_sources_7, sizeof(int32_t)*_num__array_synapses_5_sources_7, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_sources_8 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_sources_8[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_sources_8, sizeof(int32_t)*_num__array_synapses_5_sources_8)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_sources_8, _array_synapses_5_sources_8, sizeof(int32_t)*_num__array_synapses_5_sources_8, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_sources_9 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_sources_9[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_sources_9, sizeof(int32_t)*_num__array_synapses_5_sources_9)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_sources_9, _array_synapses_5_sources_9, sizeof(int32_t)*_num__array_synapses_5_sources_9, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_targets = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_targets[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_targets, sizeof(int32_t)*_num__array_synapses_5_targets)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_targets, _array_synapses_5_targets, sizeof(int32_t)*_num__array_synapses_5_targets, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_targets_1 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_targets_1[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_targets_1, sizeof(int32_t)*_num__array_synapses_5_targets_1)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_targets_1, _array_synapses_5_targets_1, sizeof(int32_t)*_num__array_synapses_5_targets_1, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_targets_2 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_targets_2[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_targets_2, sizeof(int32_t)*_num__array_synapses_5_targets_2)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_targets_2, _array_synapses_5_targets_2, sizeof(int32_t)*_num__array_synapses_5_targets_2, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_targets_3 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_targets_3[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_targets_3, sizeof(int32_t)*_num__array_synapses_5_targets_3)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_targets_3, _array_synapses_5_targets_3, sizeof(int32_t)*_num__array_synapses_5_targets_3, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_targets_4 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_targets_4[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_targets_4, sizeof(int32_t)*_num__array_synapses_5_targets_4)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_targets_4, _array_synapses_5_targets_4, sizeof(int32_t)*_num__array_synapses_5_targets_4, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_targets_5 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_targets_5[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_targets_5, sizeof(int32_t)*_num__array_synapses_5_targets_5)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_targets_5, _array_synapses_5_targets_5, sizeof(int32_t)*_num__array_synapses_5_targets_5, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_targets_6 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_targets_6[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_targets_6, sizeof(int32_t)*_num__array_synapses_5_targets_6)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_targets_6, _array_synapses_5_targets_6, sizeof(int32_t)*_num__array_synapses_5_targets_6, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_targets_7 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_targets_7[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_targets_7, sizeof(int32_t)*_num__array_synapses_5_targets_7)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_targets_7, _array_synapses_5_targets_7, sizeof(int32_t)*_num__array_synapses_5_targets_7, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_targets_8 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_targets_8[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_targets_8, sizeof(int32_t)*_num__array_synapses_5_targets_8)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_targets_8, _array_synapses_5_targets_8, sizeof(int32_t)*_num__array_synapses_5_targets_8, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_targets_9 = new int32_t[10];
            for(int i=0; i<10; i++) _array_synapses_5_targets_9[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_targets_9, sizeof(int32_t)*_num__array_synapses_5_targets_9)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_targets_9, _array_synapses_5_targets_9, sizeof(int32_t)*_num__array_synapses_5_targets_9, cudaMemcpyHostToDevice)
                    );
            _array_synapses_6_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_6_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_6_N, sizeof(int32_t)*_num__array_synapses_6_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_6_N, _array_synapses_6_N, sizeof(int32_t)*_num__array_synapses_6_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_N, sizeof(int32_t)*_num__array_synapses_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_N, _array_synapses_N, sizeof(int32_t)*_num__array_synapses_N, cudaMemcpyHostToDevice)
                    );

    // Arrays initialized to an "arange"
    _array_neurongroup_1_i = new int32_t[10];
    for(int i=0; i<10; i++) _array_neurongroup_1_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_1_i, _array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_2_i = new int32_t[10];
    for(int i=0; i<10; i++) _array_neurongroup_2_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_2_i, sizeof(int32_t)*_num__array_neurongroup_2_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_2_i, _array_neurongroup_2_i, sizeof(int32_t)*_num__array_neurongroup_2_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_3_i = new int32_t[10];
    for(int i=0; i<10; i++) _array_neurongroup_3_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_3_i, sizeof(int32_t)*_num__array_neurongroup_3_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_3_i, _array_neurongroup_3_i, sizeof(int32_t)*_num__array_neurongroup_3_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_i = new int32_t[100];
    for(int i=0; i<100; i++) _array_neurongroup_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice)
            );
    _array_poissongroup_1_i = new int32_t[10];
    for(int i=0; i<10; i++) _array_poissongroup_1_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_poissongroup_1_i, sizeof(int32_t)*_num__array_poissongroup_1_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_poissongroup_1_i, _array_poissongroup_1_i, sizeof(int32_t)*_num__array_poissongroup_1_i, cudaMemcpyHostToDevice)
            );
    _array_poissongroup_i = new int32_t[100];
    for(int i=0; i<100; i++) _array_poissongroup_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_poissongroup_i, sizeof(int32_t)*_num__array_poissongroup_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_poissongroup_i, _array_poissongroup_i, sizeof(int32_t)*_num__array_poissongroup_i, cudaMemcpyHostToDevice)
            );
    _array_spikemonitor_1__source_idx = new int32_t[10];
    for(int i=0; i<10; i++) _array_spikemonitor_1__source_idx[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikemonitor_1__source_idx, sizeof(int32_t)*_num__array_spikemonitor_1__source_idx)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_spikemonitor_1__source_idx, _array_spikemonitor_1__source_idx, sizeof(int32_t)*_num__array_spikemonitor_1__source_idx, cudaMemcpyHostToDevice)
            );
    _array_spikemonitor_2__source_idx = new int32_t[10];
    for(int i=0; i<10; i++) _array_spikemonitor_2__source_idx[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikemonitor_2__source_idx, sizeof(int32_t)*_num__array_spikemonitor_2__source_idx)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_spikemonitor_2__source_idx, _array_spikemonitor_2__source_idx, sizeof(int32_t)*_num__array_spikemonitor_2__source_idx, cudaMemcpyHostToDevice)
            );
    _array_spikemonitor_3__source_idx = new int32_t[10];
    for(int i=0; i<10; i++) _array_spikemonitor_3__source_idx[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikemonitor_3__source_idx, sizeof(int32_t)*_num__array_spikemonitor_3__source_idx)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_spikemonitor_3__source_idx, _array_spikemonitor_3__source_idx, sizeof(int32_t)*_num__array_spikemonitor_3__source_idx, cudaMemcpyHostToDevice)
            );
    _array_spikemonitor__source_idx = new int32_t[100];
    for(int i=0; i<100; i++) _array_spikemonitor__source_idx[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_spikemonitor__source_idx, _array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyHostToDevice)
            );

    // static arrays
    _static_array__array_statemonitor_1__indices = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_statemonitor_1__indices, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_statemonitor_1__indices, &dev_static_array__array_statemonitor_1__indices, sizeof(int32_t*))
            );
    _static_array__array_statemonitor_2__indices = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_statemonitor_2__indices, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_statemonitor_2__indices, &dev_static_array__array_statemonitor_2__indices, sizeof(int32_t*))
            );
    _static_array__array_statemonitor_3__indices = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_statemonitor_3__indices, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_statemonitor_3__indices, &dev_static_array__array_statemonitor_3__indices, sizeof(int32_t*))
            );
    _static_array__array_statemonitor__indices = new int32_t[100];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_statemonitor__indices, sizeof(int32_t)*100)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_statemonitor__indices, &dev_static_array__array_statemonitor__indices, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_sources = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_sources, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_sources, &dev_static_array__array_synapses_4_sources, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_sources_1 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_sources_1, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_sources_1, &dev_static_array__array_synapses_4_sources_1, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_sources_2 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_sources_2, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_sources_2, &dev_static_array__array_synapses_4_sources_2, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_sources_3 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_sources_3, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_sources_3, &dev_static_array__array_synapses_4_sources_3, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_sources_4 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_sources_4, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_sources_4, &dev_static_array__array_synapses_4_sources_4, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_sources_5 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_sources_5, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_sources_5, &dev_static_array__array_synapses_4_sources_5, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_sources_6 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_sources_6, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_sources_6, &dev_static_array__array_synapses_4_sources_6, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_sources_7 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_sources_7, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_sources_7, &dev_static_array__array_synapses_4_sources_7, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_sources_8 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_sources_8, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_sources_8, &dev_static_array__array_synapses_4_sources_8, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_sources_9 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_sources_9, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_sources_9, &dev_static_array__array_synapses_4_sources_9, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_targets = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_targets, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_targets, &dev_static_array__array_synapses_4_targets, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_targets_1 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_targets_1, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_targets_1, &dev_static_array__array_synapses_4_targets_1, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_targets_2 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_targets_2, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_targets_2, &dev_static_array__array_synapses_4_targets_2, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_targets_3 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_targets_3, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_targets_3, &dev_static_array__array_synapses_4_targets_3, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_targets_4 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_targets_4, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_targets_4, &dev_static_array__array_synapses_4_targets_4, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_targets_5 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_targets_5, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_targets_5, &dev_static_array__array_synapses_4_targets_5, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_targets_6 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_targets_6, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_targets_6, &dev_static_array__array_synapses_4_targets_6, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_targets_7 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_targets_7, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_targets_7, &dev_static_array__array_synapses_4_targets_7, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_targets_8 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_targets_8, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_targets_8, &dev_static_array__array_synapses_4_targets_8, sizeof(int32_t*))
            );
    _static_array__array_synapses_4_targets_9 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_4_targets_9, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_4_targets_9, &dev_static_array__array_synapses_4_targets_9, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_sources = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_sources, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_sources, &dev_static_array__array_synapses_5_sources, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_sources_1 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_sources_1, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_sources_1, &dev_static_array__array_synapses_5_sources_1, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_sources_2 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_sources_2, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_sources_2, &dev_static_array__array_synapses_5_sources_2, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_sources_3 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_sources_3, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_sources_3, &dev_static_array__array_synapses_5_sources_3, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_sources_4 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_sources_4, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_sources_4, &dev_static_array__array_synapses_5_sources_4, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_sources_5 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_sources_5, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_sources_5, &dev_static_array__array_synapses_5_sources_5, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_sources_6 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_sources_6, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_sources_6, &dev_static_array__array_synapses_5_sources_6, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_sources_7 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_sources_7, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_sources_7, &dev_static_array__array_synapses_5_sources_7, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_sources_8 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_sources_8, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_sources_8, &dev_static_array__array_synapses_5_sources_8, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_sources_9 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_sources_9, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_sources_9, &dev_static_array__array_synapses_5_sources_9, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_targets = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_targets, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_targets, &dev_static_array__array_synapses_5_targets, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_targets_1 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_targets_1, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_targets_1, &dev_static_array__array_synapses_5_targets_1, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_targets_2 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_targets_2, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_targets_2, &dev_static_array__array_synapses_5_targets_2, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_targets_3 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_targets_3, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_targets_3, &dev_static_array__array_synapses_5_targets_3, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_targets_4 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_targets_4, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_targets_4, &dev_static_array__array_synapses_5_targets_4, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_targets_5 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_targets_5, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_targets_5, &dev_static_array__array_synapses_5_targets_5, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_targets_6 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_targets_6, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_targets_6, &dev_static_array__array_synapses_5_targets_6, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_targets_7 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_targets_7, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_targets_7, &dev_static_array__array_synapses_5_targets_7, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_targets_8 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_targets_8, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_targets_8, &dev_static_array__array_synapses_5_targets_8, sizeof(int32_t*))
            );
    _static_array__array_synapses_5_targets_9 = new int32_t[10];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_5_targets_9, sizeof(int32_t)*10)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_5_targets_9, &dev_static_array__array_synapses_5_targets_9, sizeof(int32_t*))
            );

    _dynamic_array_statemonitor_1_V = new thrust::device_vector<double>[_num__array_statemonitor_1__indices];
    _dynamic_array_statemonitor_2_V = new thrust::device_vector<double>[_num__array_statemonitor_2__indices];
    _dynamic_array_statemonitor_3_V = new thrust::device_vector<double>[_num__array_statemonitor_3__indices];
    _dynamic_array_statemonitor_V = new thrust::device_vector<double>[_num__array_statemonitor__indices];

    // eventspace_arrays
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_1__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_1__spikespace)
            );
    _array_neurongroup_1__spikespace = new int32_t[11];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_2__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_2__spikespace)
            );
    _array_neurongroup_2__spikespace = new int32_t[11];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_3__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_3__spikespace)
            );
    _array_neurongroup_3__spikespace = new int32_t[11];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup__spikespace[0], sizeof(int32_t)*_num__array_neurongroup__spikespace)
            );
    _array_neurongroup__spikespace = new int32_t[101];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_poissongroup_1__spikespace[0], sizeof(int32_t)*_num__array_poissongroup_1__spikespace)
            );
    _array_poissongroup_1__spikespace = new int32_t[11];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_poissongroup__spikespace[0], sizeof(int32_t)*_num__array_poissongroup__spikespace)
            );
    _array_poissongroup__spikespace = new int32_t[101];

    CUDA_CHECK_MEMORY();
    const double to_MB = 1.0 / (1024.0 * 1024.0);
    double tot_memory_MB = (used_device_memory - used_device_memory_start) * to_MB;
    double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
    std::cout << "INFO: _init_arrays() took " <<  time_passed << "s";
    if (tot_memory_MB > 0)
        std::cout << " and used " << tot_memory_MB << "MB of device memory.";
    std::cout << std::endl;
}

void _load_arrays()
{
    using namespace brian;

    ifstream f_static_array__array_statemonitor_1__indices;
    f_static_array__array_statemonitor_1__indices.open("static_arrays/_static_array__array_statemonitor_1__indices", ios::in | ios::binary);
    if(f_static_array__array_statemonitor_1__indices.is_open())
    {
        f_static_array__array_statemonitor_1__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor_1__indices), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_statemonitor_1__indices." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_statemonitor_1__indices, _static_array__array_statemonitor_1__indices, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_statemonitor_2__indices;
    f_static_array__array_statemonitor_2__indices.open("static_arrays/_static_array__array_statemonitor_2__indices", ios::in | ios::binary);
    if(f_static_array__array_statemonitor_2__indices.is_open())
    {
        f_static_array__array_statemonitor_2__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor_2__indices), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_statemonitor_2__indices." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_statemonitor_2__indices, _static_array__array_statemonitor_2__indices, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_statemonitor_3__indices;
    f_static_array__array_statemonitor_3__indices.open("static_arrays/_static_array__array_statemonitor_3__indices", ios::in | ios::binary);
    if(f_static_array__array_statemonitor_3__indices.is_open())
    {
        f_static_array__array_statemonitor_3__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor_3__indices), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_statemonitor_3__indices." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_statemonitor_3__indices, _static_array__array_statemonitor_3__indices, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_statemonitor__indices;
    f_static_array__array_statemonitor__indices.open("static_arrays/_static_array__array_statemonitor__indices", ios::in | ios::binary);
    if(f_static_array__array_statemonitor__indices.is_open())
    {
        f_static_array__array_statemonitor__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor__indices), 100*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_statemonitor__indices." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_statemonitor__indices, _static_array__array_statemonitor__indices, sizeof(int32_t)*100, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_sources;
    f_static_array__array_synapses_4_sources.open("static_arrays/_static_array__array_synapses_4_sources", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_sources.is_open())
    {
        f_static_array__array_synapses_4_sources.read(reinterpret_cast<char*>(_static_array__array_synapses_4_sources), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_sources." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_sources, _static_array__array_synapses_4_sources, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_sources_1;
    f_static_array__array_synapses_4_sources_1.open("static_arrays/_static_array__array_synapses_4_sources_1", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_sources_1.is_open())
    {
        f_static_array__array_synapses_4_sources_1.read(reinterpret_cast<char*>(_static_array__array_synapses_4_sources_1), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_sources_1." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_sources_1, _static_array__array_synapses_4_sources_1, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_sources_2;
    f_static_array__array_synapses_4_sources_2.open("static_arrays/_static_array__array_synapses_4_sources_2", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_sources_2.is_open())
    {
        f_static_array__array_synapses_4_sources_2.read(reinterpret_cast<char*>(_static_array__array_synapses_4_sources_2), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_sources_2." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_sources_2, _static_array__array_synapses_4_sources_2, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_sources_3;
    f_static_array__array_synapses_4_sources_3.open("static_arrays/_static_array__array_synapses_4_sources_3", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_sources_3.is_open())
    {
        f_static_array__array_synapses_4_sources_3.read(reinterpret_cast<char*>(_static_array__array_synapses_4_sources_3), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_sources_3." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_sources_3, _static_array__array_synapses_4_sources_3, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_sources_4;
    f_static_array__array_synapses_4_sources_4.open("static_arrays/_static_array__array_synapses_4_sources_4", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_sources_4.is_open())
    {
        f_static_array__array_synapses_4_sources_4.read(reinterpret_cast<char*>(_static_array__array_synapses_4_sources_4), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_sources_4." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_sources_4, _static_array__array_synapses_4_sources_4, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_sources_5;
    f_static_array__array_synapses_4_sources_5.open("static_arrays/_static_array__array_synapses_4_sources_5", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_sources_5.is_open())
    {
        f_static_array__array_synapses_4_sources_5.read(reinterpret_cast<char*>(_static_array__array_synapses_4_sources_5), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_sources_5." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_sources_5, _static_array__array_synapses_4_sources_5, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_sources_6;
    f_static_array__array_synapses_4_sources_6.open("static_arrays/_static_array__array_synapses_4_sources_6", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_sources_6.is_open())
    {
        f_static_array__array_synapses_4_sources_6.read(reinterpret_cast<char*>(_static_array__array_synapses_4_sources_6), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_sources_6." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_sources_6, _static_array__array_synapses_4_sources_6, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_sources_7;
    f_static_array__array_synapses_4_sources_7.open("static_arrays/_static_array__array_synapses_4_sources_7", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_sources_7.is_open())
    {
        f_static_array__array_synapses_4_sources_7.read(reinterpret_cast<char*>(_static_array__array_synapses_4_sources_7), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_sources_7." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_sources_7, _static_array__array_synapses_4_sources_7, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_sources_8;
    f_static_array__array_synapses_4_sources_8.open("static_arrays/_static_array__array_synapses_4_sources_8", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_sources_8.is_open())
    {
        f_static_array__array_synapses_4_sources_8.read(reinterpret_cast<char*>(_static_array__array_synapses_4_sources_8), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_sources_8." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_sources_8, _static_array__array_synapses_4_sources_8, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_sources_9;
    f_static_array__array_synapses_4_sources_9.open("static_arrays/_static_array__array_synapses_4_sources_9", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_sources_9.is_open())
    {
        f_static_array__array_synapses_4_sources_9.read(reinterpret_cast<char*>(_static_array__array_synapses_4_sources_9), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_sources_9." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_sources_9, _static_array__array_synapses_4_sources_9, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_targets;
    f_static_array__array_synapses_4_targets.open("static_arrays/_static_array__array_synapses_4_targets", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_targets.is_open())
    {
        f_static_array__array_synapses_4_targets.read(reinterpret_cast<char*>(_static_array__array_synapses_4_targets), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_targets." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_targets, _static_array__array_synapses_4_targets, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_targets_1;
    f_static_array__array_synapses_4_targets_1.open("static_arrays/_static_array__array_synapses_4_targets_1", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_targets_1.is_open())
    {
        f_static_array__array_synapses_4_targets_1.read(reinterpret_cast<char*>(_static_array__array_synapses_4_targets_1), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_targets_1." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_targets_1, _static_array__array_synapses_4_targets_1, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_targets_2;
    f_static_array__array_synapses_4_targets_2.open("static_arrays/_static_array__array_synapses_4_targets_2", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_targets_2.is_open())
    {
        f_static_array__array_synapses_4_targets_2.read(reinterpret_cast<char*>(_static_array__array_synapses_4_targets_2), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_targets_2." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_targets_2, _static_array__array_synapses_4_targets_2, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_targets_3;
    f_static_array__array_synapses_4_targets_3.open("static_arrays/_static_array__array_synapses_4_targets_3", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_targets_3.is_open())
    {
        f_static_array__array_synapses_4_targets_3.read(reinterpret_cast<char*>(_static_array__array_synapses_4_targets_3), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_targets_3." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_targets_3, _static_array__array_synapses_4_targets_3, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_targets_4;
    f_static_array__array_synapses_4_targets_4.open("static_arrays/_static_array__array_synapses_4_targets_4", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_targets_4.is_open())
    {
        f_static_array__array_synapses_4_targets_4.read(reinterpret_cast<char*>(_static_array__array_synapses_4_targets_4), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_targets_4." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_targets_4, _static_array__array_synapses_4_targets_4, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_targets_5;
    f_static_array__array_synapses_4_targets_5.open("static_arrays/_static_array__array_synapses_4_targets_5", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_targets_5.is_open())
    {
        f_static_array__array_synapses_4_targets_5.read(reinterpret_cast<char*>(_static_array__array_synapses_4_targets_5), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_targets_5." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_targets_5, _static_array__array_synapses_4_targets_5, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_targets_6;
    f_static_array__array_synapses_4_targets_6.open("static_arrays/_static_array__array_synapses_4_targets_6", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_targets_6.is_open())
    {
        f_static_array__array_synapses_4_targets_6.read(reinterpret_cast<char*>(_static_array__array_synapses_4_targets_6), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_targets_6." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_targets_6, _static_array__array_synapses_4_targets_6, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_targets_7;
    f_static_array__array_synapses_4_targets_7.open("static_arrays/_static_array__array_synapses_4_targets_7", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_targets_7.is_open())
    {
        f_static_array__array_synapses_4_targets_7.read(reinterpret_cast<char*>(_static_array__array_synapses_4_targets_7), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_targets_7." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_targets_7, _static_array__array_synapses_4_targets_7, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_targets_8;
    f_static_array__array_synapses_4_targets_8.open("static_arrays/_static_array__array_synapses_4_targets_8", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_targets_8.is_open())
    {
        f_static_array__array_synapses_4_targets_8.read(reinterpret_cast<char*>(_static_array__array_synapses_4_targets_8), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_targets_8." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_targets_8, _static_array__array_synapses_4_targets_8, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_4_targets_9;
    f_static_array__array_synapses_4_targets_9.open("static_arrays/_static_array__array_synapses_4_targets_9", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_targets_9.is_open())
    {
        f_static_array__array_synapses_4_targets_9.read(reinterpret_cast<char*>(_static_array__array_synapses_4_targets_9), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_targets_9." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_4_targets_9, _static_array__array_synapses_4_targets_9, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_sources;
    f_static_array__array_synapses_5_sources.open("static_arrays/_static_array__array_synapses_5_sources", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_sources.is_open())
    {
        f_static_array__array_synapses_5_sources.read(reinterpret_cast<char*>(_static_array__array_synapses_5_sources), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_sources." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_sources, _static_array__array_synapses_5_sources, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_sources_1;
    f_static_array__array_synapses_5_sources_1.open("static_arrays/_static_array__array_synapses_5_sources_1", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_sources_1.is_open())
    {
        f_static_array__array_synapses_5_sources_1.read(reinterpret_cast<char*>(_static_array__array_synapses_5_sources_1), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_sources_1." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_sources_1, _static_array__array_synapses_5_sources_1, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_sources_2;
    f_static_array__array_synapses_5_sources_2.open("static_arrays/_static_array__array_synapses_5_sources_2", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_sources_2.is_open())
    {
        f_static_array__array_synapses_5_sources_2.read(reinterpret_cast<char*>(_static_array__array_synapses_5_sources_2), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_sources_2." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_sources_2, _static_array__array_synapses_5_sources_2, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_sources_3;
    f_static_array__array_synapses_5_sources_3.open("static_arrays/_static_array__array_synapses_5_sources_3", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_sources_3.is_open())
    {
        f_static_array__array_synapses_5_sources_3.read(reinterpret_cast<char*>(_static_array__array_synapses_5_sources_3), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_sources_3." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_sources_3, _static_array__array_synapses_5_sources_3, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_sources_4;
    f_static_array__array_synapses_5_sources_4.open("static_arrays/_static_array__array_synapses_5_sources_4", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_sources_4.is_open())
    {
        f_static_array__array_synapses_5_sources_4.read(reinterpret_cast<char*>(_static_array__array_synapses_5_sources_4), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_sources_4." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_sources_4, _static_array__array_synapses_5_sources_4, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_sources_5;
    f_static_array__array_synapses_5_sources_5.open("static_arrays/_static_array__array_synapses_5_sources_5", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_sources_5.is_open())
    {
        f_static_array__array_synapses_5_sources_5.read(reinterpret_cast<char*>(_static_array__array_synapses_5_sources_5), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_sources_5." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_sources_5, _static_array__array_synapses_5_sources_5, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_sources_6;
    f_static_array__array_synapses_5_sources_6.open("static_arrays/_static_array__array_synapses_5_sources_6", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_sources_6.is_open())
    {
        f_static_array__array_synapses_5_sources_6.read(reinterpret_cast<char*>(_static_array__array_synapses_5_sources_6), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_sources_6." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_sources_6, _static_array__array_synapses_5_sources_6, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_sources_7;
    f_static_array__array_synapses_5_sources_7.open("static_arrays/_static_array__array_synapses_5_sources_7", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_sources_7.is_open())
    {
        f_static_array__array_synapses_5_sources_7.read(reinterpret_cast<char*>(_static_array__array_synapses_5_sources_7), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_sources_7." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_sources_7, _static_array__array_synapses_5_sources_7, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_sources_8;
    f_static_array__array_synapses_5_sources_8.open("static_arrays/_static_array__array_synapses_5_sources_8", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_sources_8.is_open())
    {
        f_static_array__array_synapses_5_sources_8.read(reinterpret_cast<char*>(_static_array__array_synapses_5_sources_8), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_sources_8." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_sources_8, _static_array__array_synapses_5_sources_8, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_sources_9;
    f_static_array__array_synapses_5_sources_9.open("static_arrays/_static_array__array_synapses_5_sources_9", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_sources_9.is_open())
    {
        f_static_array__array_synapses_5_sources_9.read(reinterpret_cast<char*>(_static_array__array_synapses_5_sources_9), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_sources_9." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_sources_9, _static_array__array_synapses_5_sources_9, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_targets;
    f_static_array__array_synapses_5_targets.open("static_arrays/_static_array__array_synapses_5_targets", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_targets.is_open())
    {
        f_static_array__array_synapses_5_targets.read(reinterpret_cast<char*>(_static_array__array_synapses_5_targets), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_targets." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_targets, _static_array__array_synapses_5_targets, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_targets_1;
    f_static_array__array_synapses_5_targets_1.open("static_arrays/_static_array__array_synapses_5_targets_1", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_targets_1.is_open())
    {
        f_static_array__array_synapses_5_targets_1.read(reinterpret_cast<char*>(_static_array__array_synapses_5_targets_1), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_targets_1." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_targets_1, _static_array__array_synapses_5_targets_1, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_targets_2;
    f_static_array__array_synapses_5_targets_2.open("static_arrays/_static_array__array_synapses_5_targets_2", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_targets_2.is_open())
    {
        f_static_array__array_synapses_5_targets_2.read(reinterpret_cast<char*>(_static_array__array_synapses_5_targets_2), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_targets_2." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_targets_2, _static_array__array_synapses_5_targets_2, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_targets_3;
    f_static_array__array_synapses_5_targets_3.open("static_arrays/_static_array__array_synapses_5_targets_3", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_targets_3.is_open())
    {
        f_static_array__array_synapses_5_targets_3.read(reinterpret_cast<char*>(_static_array__array_synapses_5_targets_3), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_targets_3." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_targets_3, _static_array__array_synapses_5_targets_3, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_targets_4;
    f_static_array__array_synapses_5_targets_4.open("static_arrays/_static_array__array_synapses_5_targets_4", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_targets_4.is_open())
    {
        f_static_array__array_synapses_5_targets_4.read(reinterpret_cast<char*>(_static_array__array_synapses_5_targets_4), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_targets_4." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_targets_4, _static_array__array_synapses_5_targets_4, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_targets_5;
    f_static_array__array_synapses_5_targets_5.open("static_arrays/_static_array__array_synapses_5_targets_5", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_targets_5.is_open())
    {
        f_static_array__array_synapses_5_targets_5.read(reinterpret_cast<char*>(_static_array__array_synapses_5_targets_5), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_targets_5." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_targets_5, _static_array__array_synapses_5_targets_5, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_targets_6;
    f_static_array__array_synapses_5_targets_6.open("static_arrays/_static_array__array_synapses_5_targets_6", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_targets_6.is_open())
    {
        f_static_array__array_synapses_5_targets_6.read(reinterpret_cast<char*>(_static_array__array_synapses_5_targets_6), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_targets_6." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_targets_6, _static_array__array_synapses_5_targets_6, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_targets_7;
    f_static_array__array_synapses_5_targets_7.open("static_arrays/_static_array__array_synapses_5_targets_7", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_targets_7.is_open())
    {
        f_static_array__array_synapses_5_targets_7.read(reinterpret_cast<char*>(_static_array__array_synapses_5_targets_7), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_targets_7." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_targets_7, _static_array__array_synapses_5_targets_7, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_targets_8;
    f_static_array__array_synapses_5_targets_8.open("static_arrays/_static_array__array_synapses_5_targets_8", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_targets_8.is_open())
    {
        f_static_array__array_synapses_5_targets_8.read(reinterpret_cast<char*>(_static_array__array_synapses_5_targets_8), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_targets_8." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_targets_8, _static_array__array_synapses_5_targets_8, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_5_targets_9;
    f_static_array__array_synapses_5_targets_9.open("static_arrays/_static_array__array_synapses_5_targets_9", ios::in | ios::binary);
    if(f_static_array__array_synapses_5_targets_9.is_open())
    {
        f_static_array__array_synapses_5_targets_9.read(reinterpret_cast<char*>(_static_array__array_synapses_5_targets_9), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_5_targets_9." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_5_targets_9, _static_array__array_synapses_5_targets_9, sizeof(int32_t)*10, cudaMemcpyHostToDevice)
            );
}

void _write_arrays()
{
    using namespace brian;

    CUDA_SAFE_CALL(
            cudaMemcpy(_array_defaultclock_dt, dev_array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_defaultclock_dt;
    outfile__array_defaultclock_dt.open("results/_array_defaultclock_dt_-9215759865592636245", ios::binary | ios::out);
    if(outfile__array_defaultclock_dt.is_open())
    {
        outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(double));
        outfile__array_defaultclock_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_defaultclock_t, dev_array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_defaultclock_t;
    outfile__array_defaultclock_t.open("results/_array_defaultclock_t_7263079326120112646", ios::binary | ios::out);
    if(outfile__array_defaultclock_t.is_open())
    {
        outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(double));
        outfile__array_defaultclock_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_t." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_defaultclock_timestep, dev_array_defaultclock_timestep, sizeof(int64_t)*_num__array_defaultclock_timestep, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_defaultclock_timestep;
    outfile__array_defaultclock_timestep.open("results/_array_defaultclock_timestep_-8300011050550298960", ios::binary | ios::out);
    if(outfile__array_defaultclock_timestep.is_open())
    {
        outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(int64_t));
        outfile__array_defaultclock_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_i, dev_array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_i;
    outfile__array_neurongroup_1_i.open("results/_array_neurongroup_1_i_6263110221643836299", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_i.is_open())
    {
        outfile__array_neurongroup_1_i.write(reinterpret_cast<char*>(_array_neurongroup_1_i), 10*sizeof(int32_t));
        outfile__array_neurongroup_1_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_s_ahp_GO, dev_array_neurongroup_1_s_ahp_GO, sizeof(double)*_num__array_neurongroup_1_s_ahp_GO, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_s_ahp_GO;
    outfile__array_neurongroup_1_s_ahp_GO.open("results/_array_neurongroup_1_s_ahp_GO_7688894509337558433", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_s_ahp_GO.is_open())
    {
        outfile__array_neurongroup_1_s_ahp_GO.write(reinterpret_cast<char*>(_array_neurongroup_1_s_ahp_GO), 10*sizeof(double));
        outfile__array_neurongroup_1_s_ahp_GO.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_s_ahp_GO." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_s_AMPA, dev_array_neurongroup_1_s_AMPA, sizeof(double)*_num__array_neurongroup_1_s_AMPA, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_s_AMPA;
    outfile__array_neurongroup_1_s_AMPA.open("results/_array_neurongroup_1_s_AMPA_9111164632410235258", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_s_AMPA.is_open())
    {
        outfile__array_neurongroup_1_s_AMPA.write(reinterpret_cast<char*>(_array_neurongroup_1_s_AMPA), 10*sizeof(double));
        outfile__array_neurongroup_1_s_AMPA.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_s_AMPA." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_s_NMDA_1, dev_array_neurongroup_1_s_NMDA_1, sizeof(double)*_num__array_neurongroup_1_s_NMDA_1, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_s_NMDA_1;
    outfile__array_neurongroup_1_s_NMDA_1.open("results/_array_neurongroup_1_s_NMDA_1_1881551723053382569", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_s_NMDA_1.is_open())
    {
        outfile__array_neurongroup_1_s_NMDA_1.write(reinterpret_cast<char*>(_array_neurongroup_1_s_NMDA_1), 10*sizeof(double));
        outfile__array_neurongroup_1_s_NMDA_1.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_s_NMDA_1." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_s_NMDA_2, dev_array_neurongroup_1_s_NMDA_2, sizeof(double)*_num__array_neurongroup_1_s_NMDA_2, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_s_NMDA_2;
    outfile__array_neurongroup_1_s_NMDA_2.open("results/_array_neurongroup_1_s_NMDA_2_1881551723053382570", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_s_NMDA_2.is_open())
    {
        outfile__array_neurongroup_1_s_NMDA_2.write(reinterpret_cast<char*>(_array_neurongroup_1_s_NMDA_2), 10*sizeof(double));
        outfile__array_neurongroup_1_s_NMDA_2.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_s_NMDA_2." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_V, dev_array_neurongroup_1_V, sizeof(double)*_num__array_neurongroup_1_V, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_V;
    outfile__array_neurongroup_1_V.open("results/_array_neurongroup_1_V_6263110221643836340", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_V.is_open())
    {
        outfile__array_neurongroup_1_V.write(reinterpret_cast<char*>(_array_neurongroup_1_V), 10*sizeof(double));
        outfile__array_neurongroup_1_V.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_V." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_x, dev_array_neurongroup_1_x, sizeof(double)*_num__array_neurongroup_1_x, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_x;
    outfile__array_neurongroup_1_x.open("results/_array_neurongroup_1_x_6263110221643836314", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_x.is_open())
    {
        outfile__array_neurongroup_1_x.write(reinterpret_cast<char*>(_array_neurongroup_1_x), 10*sizeof(double));
        outfile__array_neurongroup_1_x.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_x." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_y, dev_array_neurongroup_1_y, sizeof(double)*_num__array_neurongroup_1_y, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_y;
    outfile__array_neurongroup_1_y.open("results/_array_neurongroup_1_y_6263110221643836315", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_y.is_open())
    {
        outfile__array_neurongroup_1_y.write(reinterpret_cast<char*>(_array_neurongroup_1_y), 10*sizeof(double));
        outfile__array_neurongroup_1_y.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_y." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_i, dev_array_neurongroup_2_i, sizeof(int32_t)*_num__array_neurongroup_2_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_i;
    outfile__array_neurongroup_2_i.open("results/_array_neurongroup_2_i_6263111221515835924", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_i.is_open())
    {
        outfile__array_neurongroup_2_i.write(reinterpret_cast<char*>(_array_neurongroup_2_i), 10*sizeof(int32_t));
        outfile__array_neurongroup_2_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_s_AHP_PKJ, dev_array_neurongroup_2_s_AHP_PKJ, sizeof(double)*_num__array_neurongroup_2_s_AHP_PKJ, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_s_AHP_PKJ;
    outfile__array_neurongroup_2_s_AHP_PKJ.open("results/_array_neurongroup_2_s_AHP_PKJ_-5788172345637898020", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_s_AHP_PKJ.is_open())
    {
        outfile__array_neurongroup_2_s_AHP_PKJ.write(reinterpret_cast<char*>(_array_neurongroup_2_s_AHP_PKJ), 10*sizeof(double));
        outfile__array_neurongroup_2_s_AHP_PKJ.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_s_AHP_PKJ." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_s_AMPA, dev_array_neurongroup_2_s_AMPA, sizeof(double)*_num__array_neurongroup_2_s_AMPA, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_s_AMPA;
    outfile__array_neurongroup_2_s_AMPA.open("results/_array_neurongroup_2_s_AMPA_-2856132977221954013", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_s_AMPA.is_open())
    {
        outfile__array_neurongroup_2_s_AMPA.write(reinterpret_cast<char*>(_array_neurongroup_2_s_AMPA), 10*sizeof(double));
        outfile__array_neurongroup_2_s_AMPA.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_s_AMPA." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_s_GABA, dev_array_neurongroup_2_s_GABA, sizeof(double)*_num__array_neurongroup_2_s_GABA, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_s_GABA;
    outfile__array_neurongroup_2_s_GABA.open("results/_array_neurongroup_2_s_GABA_-4856138977217953825", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_s_GABA.is_open())
    {
        outfile__array_neurongroup_2_s_GABA.write(reinterpret_cast<char*>(_array_neurongroup_2_s_GABA), 10*sizeof(double));
        outfile__array_neurongroup_2_s_GABA.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_s_GABA." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_V, dev_array_neurongroup_2_V, sizeof(double)*_num__array_neurongroup_2_V, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_V;
    outfile__array_neurongroup_2_V.open("results/_array_neurongroup_2_V_6263111221515835947", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_V.is_open())
    {
        outfile__array_neurongroup_2_V.write(reinterpret_cast<char*>(_array_neurongroup_2_V), 10*sizeof(double));
        outfile__array_neurongroup_2_V.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_V." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_3_i, dev_array_neurongroup_3_i, sizeof(int32_t)*_num__array_neurongroup_3_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_3_i;
    outfile__array_neurongroup_3_i.open("results/_array_neurongroup_3_i_6263112221643836317", ios::binary | ios::out);
    if(outfile__array_neurongroup_3_i.is_open())
    {
        outfile__array_neurongroup_3_i.write(reinterpret_cast<char*>(_array_neurongroup_3_i), 10*sizeof(int32_t));
        outfile__array_neurongroup_3_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_3_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_3_s_AHP_BS, dev_array_neurongroup_3_s_AHP_BS, sizeof(double)*_num__array_neurongroup_3_s_AHP_BS, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_3_s_AHP_BS;
    outfile__array_neurongroup_3_s_AHP_BS.open("results/_array_neurongroup_3_s_AHP_BS_-3634154894893597864", ios::binary | ios::out);
    if(outfile__array_neurongroup_3_s_AHP_BS.is_open())
    {
        outfile__array_neurongroup_3_s_AHP_BS.write(reinterpret_cast<char*>(_array_neurongroup_3_s_AHP_BS), 10*sizeof(double));
        outfile__array_neurongroup_3_s_AHP_BS.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_3_s_AHP_BS." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_3_s_AMPA, dev_array_neurongroup_3_s_AMPA, sizeof(double)*_num__array_neurongroup_3_s_AMPA, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_3_s_AMPA;
    outfile__array_neurongroup_3_s_AMPA.open("results/_array_neurongroup_3_s_AMPA_-4351295136519119976", ios::binary | ios::out);
    if(outfile__array_neurongroup_3_s_AMPA.is_open())
    {
        outfile__array_neurongroup_3_s_AMPA.write(reinterpret_cast<char*>(_array_neurongroup_3_s_AMPA), 10*sizeof(double));
        outfile__array_neurongroup_3_s_AMPA.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_3_s_AMPA." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_3_V, dev_array_neurongroup_3_V, sizeof(double)*_num__array_neurongroup_3_V, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_3_V;
    outfile__array_neurongroup_3_V.open("results/_array_neurongroup_3_V_6263112221643836322", ios::binary | ios::out);
    if(outfile__array_neurongroup_3_V.is_open())
    {
        outfile__array_neurongroup_3_V.write(reinterpret_cast<char*>(_array_neurongroup_3_V), 10*sizeof(double));
        outfile__array_neurongroup_3_V.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_3_V." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_i, dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_i;
    outfile__array_neurongroup_i.open("results/_array_neurongroup_i_-2688036259655650195", ios::binary | ios::out);
    if(outfile__array_neurongroup_i.is_open())
    {
        outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 100*sizeof(int32_t));
        outfile__array_neurongroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_s_ahp_GR, dev_array_neurongroup_s_ahp_GR, sizeof(double)*_num__array_neurongroup_s_ahp_GR, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_s_ahp_GR;
    outfile__array_neurongroup_s_ahp_GR.open("results/_array_neurongroup_s_ahp_GR_-506910623088042858", ios::binary | ios::out);
    if(outfile__array_neurongroup_s_ahp_GR.is_open())
    {
        outfile__array_neurongroup_s_ahp_GR.write(reinterpret_cast<char*>(_array_neurongroup_s_ahp_GR), 100*sizeof(double));
        outfile__array_neurongroup_s_ahp_GR.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_s_ahp_GR." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_s_AMPA, dev_array_neurongroup_s_AMPA, sizeof(double)*_num__array_neurongroup_s_AMPA, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_s_AMPA;
    outfile__array_neurongroup_s_AMPA.open("results/_array_neurongroup_s_AMPA_4830339705933334508", ios::binary | ios::out);
    if(outfile__array_neurongroup_s_AMPA.is_open())
    {
        outfile__array_neurongroup_s_AMPA.write(reinterpret_cast<char*>(_array_neurongroup_s_AMPA), 100*sizeof(double));
        outfile__array_neurongroup_s_AMPA.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_s_AMPA." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_s_GABA_1, dev_array_neurongroup_s_GABA_1, sizeof(double)*_num__array_neurongroup_s_GABA_1, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_s_GABA_1;
    outfile__array_neurongroup_s_GABA_1.open("results/_array_neurongroup_s_GABA_1_2952454834921723734", ios::binary | ios::out);
    if(outfile__array_neurongroup_s_GABA_1.is_open())
    {
        outfile__array_neurongroup_s_GABA_1.write(reinterpret_cast<char*>(_array_neurongroup_s_GABA_1), 100*sizeof(double));
        outfile__array_neurongroup_s_GABA_1.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_s_GABA_1." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_s_GABA_2, dev_array_neurongroup_s_GABA_2, sizeof(double)*_num__array_neurongroup_s_GABA_2, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_s_GABA_2;
    outfile__array_neurongroup_s_GABA_2.open("results/_array_neurongroup_s_GABA_2_2952454834921723733", ios::binary | ios::out);
    if(outfile__array_neurongroup_s_GABA_2.is_open())
    {
        outfile__array_neurongroup_s_GABA_2.write(reinterpret_cast<char*>(_array_neurongroup_s_GABA_2), 100*sizeof(double));
        outfile__array_neurongroup_s_GABA_2.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_s_GABA_2." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_s_NMDA, dev_array_neurongroup_s_NMDA, sizeof(double)*_num__array_neurongroup_s_NMDA, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_s_NMDA;
    outfile__array_neurongroup_s_NMDA.open("results/_array_neurongroup_s_NMDA_-4616313367485216827", ios::binary | ios::out);
    if(outfile__array_neurongroup_s_NMDA.is_open())
    {
        outfile__array_neurongroup_s_NMDA.write(reinterpret_cast<char*>(_array_neurongroup_s_NMDA), 100*sizeof(double));
        outfile__array_neurongroup_s_NMDA.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_s_NMDA." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_V, dev_array_neurongroup_V, sizeof(double)*_num__array_neurongroup_V, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_V;
    outfile__array_neurongroup_V.open("results/_array_neurongroup_V_-2688036259655650222", ios::binary | ios::out);
    if(outfile__array_neurongroup_V.is_open())
    {
        outfile__array_neurongroup_V.write(reinterpret_cast<char*>(_array_neurongroup_V), 100*sizeof(double));
        outfile__array_neurongroup_V.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_V." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_x, dev_array_neurongroup_x, sizeof(double)*_num__array_neurongroup_x, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_x;
    outfile__array_neurongroup_x.open("results/_array_neurongroup_x_-2688036259655650180", ios::binary | ios::out);
    if(outfile__array_neurongroup_x.is_open())
    {
        outfile__array_neurongroup_x.write(reinterpret_cast<char*>(_array_neurongroup_x), 100*sizeof(double));
        outfile__array_neurongroup_x.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_x." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_y, dev_array_neurongroup_y, sizeof(double)*_num__array_neurongroup_y, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_y;
    outfile__array_neurongroup_y.open("results/_array_neurongroup_y_-2688036259655650179", ios::binary | ios::out);
    if(outfile__array_neurongroup_y.is_open())
    {
        outfile__array_neurongroup_y.write(reinterpret_cast<char*>(_array_neurongroup_y), 100*sizeof(double));
        outfile__array_neurongroup_y.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_y." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_poissongroup_1_i, dev_array_poissongroup_1_i, sizeof(int32_t)*_num__array_poissongroup_1_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_poissongroup_1_i;
    outfile__array_poissongroup_1_i.open("results/_array_poissongroup_1_i_-2781519553687852768", ios::binary | ios::out);
    if(outfile__array_poissongroup_1_i.is_open())
    {
        outfile__array_poissongroup_1_i.write(reinterpret_cast<char*>(_array_poissongroup_1_i), 10*sizeof(int32_t));
        outfile__array_poissongroup_1_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_poissongroup_1_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_poissongroup_i, dev_array_poissongroup_i, sizeof(int32_t)*_num__array_poissongroup_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_poissongroup_i;
    outfile__array_poissongroup_i.open("results/_array_poissongroup_i_2968481286923345034", ios::binary | ios::out);
    if(outfile__array_poissongroup_i.is_open())
    {
        outfile__array_poissongroup_i.write(reinterpret_cast<char*>(_array_poissongroup_i), 100*sizeof(int32_t));
        outfile__array_poissongroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_poissongroup_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_ratemonitor_1_N, dev_array_ratemonitor_1_N, sizeof(int32_t)*_num__array_ratemonitor_1_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_ratemonitor_1_N;
    outfile__array_ratemonitor_1_N.open("results/_array_ratemonitor_1_N_6368318342283323206", ios::binary | ios::out);
    if(outfile__array_ratemonitor_1_N.is_open())
    {
        outfile__array_ratemonitor_1_N.write(reinterpret_cast<char*>(_array_ratemonitor_1_N), 1*sizeof(int32_t));
        outfile__array_ratemonitor_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_ratemonitor_1_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_ratemonitor_2_N, dev_array_ratemonitor_2_N, sizeof(int32_t)*_num__array_ratemonitor_2_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_ratemonitor_2_N;
    outfile__array_ratemonitor_2_N.open("results/_array_ratemonitor_2_N_6368315342411323617", ios::binary | ios::out);
    if(outfile__array_ratemonitor_2_N.is_open())
    {
        outfile__array_ratemonitor_2_N.write(reinterpret_cast<char*>(_array_ratemonitor_2_N), 1*sizeof(int32_t));
        outfile__array_ratemonitor_2_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_ratemonitor_2_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_ratemonitor_3_N, dev_array_ratemonitor_3_N, sizeof(int32_t)*_num__array_ratemonitor_3_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_ratemonitor_3_N;
    outfile__array_ratemonitor_3_N.open("results/_array_ratemonitor_3_N_6368316342283323240", ios::binary | ios::out);
    if(outfile__array_ratemonitor_3_N.is_open())
    {
        outfile__array_ratemonitor_3_N.write(reinterpret_cast<char*>(_array_ratemonitor_3_N), 1*sizeof(int32_t));
        outfile__array_ratemonitor_3_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_ratemonitor_3_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_ratemonitor_N, dev_array_ratemonitor_N, sizeof(int32_t)*_num__array_ratemonitor_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_ratemonitor_N;
    outfile__array_ratemonitor_N.open("results/_array_ratemonitor_N_-3063889987686337940", ios::binary | ios::out);
    if(outfile__array_ratemonitor_N.is_open())
    {
        outfile__array_ratemonitor_N.write(reinterpret_cast<char*>(_array_ratemonitor_N), 1*sizeof(int32_t));
        outfile__array_ratemonitor_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_ratemonitor_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_1__source_idx, dev_array_spikemonitor_1__source_idx, sizeof(int32_t)*_num__array_spikemonitor_1__source_idx, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_1__source_idx;
    outfile__array_spikemonitor_1__source_idx.open("results/_array_spikemonitor_1__source_idx_2519895180673232517", ios::binary | ios::out);
    if(outfile__array_spikemonitor_1__source_idx.is_open())
    {
        outfile__array_spikemonitor_1__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor_1__source_idx), 10*sizeof(int32_t));
        outfile__array_spikemonitor_1__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_1__source_idx." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_1_count, dev_array_spikemonitor_1_count, sizeof(int32_t)*_num__array_spikemonitor_1_count, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_1_count;
    outfile__array_spikemonitor_1_count.open("results/_array_spikemonitor_1_count_7346364689340180690", ios::binary | ios::out);
    if(outfile__array_spikemonitor_1_count.is_open())
    {
        outfile__array_spikemonitor_1_count.write(reinterpret_cast<char*>(_array_spikemonitor_1_count), 10*sizeof(int32_t));
        outfile__array_spikemonitor_1_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_1_count." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_1_N, dev_array_spikemonitor_1_N, sizeof(int32_t)*_num__array_spikemonitor_1_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_1_N;
    outfile__array_spikemonitor_1_N.open("results/_array_spikemonitor_1_N_5308729907026091517", ios::binary | ios::out);
    if(outfile__array_spikemonitor_1_N.is_open())
    {
        outfile__array_spikemonitor_1_N.write(reinterpret_cast<char*>(_array_spikemonitor_1_N), 1*sizeof(int32_t));
        outfile__array_spikemonitor_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_1_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_2__source_idx, dev_array_spikemonitor_2__source_idx, sizeof(int32_t)*_num__array_spikemonitor_2__source_idx, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_2__source_idx;
    outfile__array_spikemonitor_2__source_idx.open("results/_array_spikemonitor_2__source_idx_-310975198662802874", ios::binary | ios::out);
    if(outfile__array_spikemonitor_2__source_idx.is_open())
    {
        outfile__array_spikemonitor_2__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor_2__source_idx), 10*sizeof(int32_t));
        outfile__array_spikemonitor_2__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_2__source_idx." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_2_count, dev_array_spikemonitor_2_count, sizeof(int32_t)*_num__array_spikemonitor_2_count, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_2_count;
    outfile__array_spikemonitor_2_count.open("results/_array_spikemonitor_2_count_-2295171584887810235", ios::binary | ios::out);
    if(outfile__array_spikemonitor_2_count.is_open())
    {
        outfile__array_spikemonitor_2_count.write(reinterpret_cast<char*>(_array_spikemonitor_2_count), 10*sizeof(int32_t));
        outfile__array_spikemonitor_2_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_2_count." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_2_N, dev_array_spikemonitor_2_N, sizeof(int32_t)*_num__array_spikemonitor_2_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_2_N;
    outfile__array_spikemonitor_2_N.open("results/_array_spikemonitor_2_N_5308730907154091842", ios::binary | ios::out);
    if(outfile__array_spikemonitor_2_N.is_open())
    {
        outfile__array_spikemonitor_2_N.write(reinterpret_cast<char*>(_array_spikemonitor_2_N), 1*sizeof(int32_t));
        outfile__array_spikemonitor_2_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_2_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_3__source_idx, dev_array_spikemonitor_3__source_idx, sizeof(int32_t)*_num__array_spikemonitor_3__source_idx, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_3__source_idx;
    outfile__array_spikemonitor_3__source_idx.open("results/_array_spikemonitor_3__source_idx_2530220051397135475", ios::binary | ios::out);
    if(outfile__array_spikemonitor_3__source_idx.is_open())
    {
        outfile__array_spikemonitor_3__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor_3__source_idx), 10*sizeof(int32_t));
        outfile__array_spikemonitor_3__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_3__source_idx." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_3_count, dev_array_spikemonitor_3_count, sizeof(int32_t)*_num__array_spikemonitor_3_count, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_3_count;
    outfile__array_spikemonitor_3_count.open("results/_array_spikemonitor_3_count_-992339061168911448", ios::binary | ios::out);
    if(outfile__array_spikemonitor_3_count.is_open())
    {
        outfile__array_spikemonitor_3_count.write(reinterpret_cast<char*>(_array_spikemonitor_3_count), 10*sizeof(int32_t));
        outfile__array_spikemonitor_3_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_3_count." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_3_N, dev_array_spikemonitor_3_N, sizeof(int32_t)*_num__array_spikemonitor_3_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_3_N;
    outfile__array_spikemonitor_3_N.open("results/_array_spikemonitor_3_N_5308731907026091467", ios::binary | ios::out);
    if(outfile__array_spikemonitor_3_N.is_open())
    {
        outfile__array_spikemonitor_3_N.write(reinterpret_cast<char*>(_array_spikemonitor_3_N), 1*sizeof(int32_t));
        outfile__array_spikemonitor_3_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_3_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor__source_idx, dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor__source_idx;
    outfile__array_spikemonitor__source_idx.open("results/_array_spikemonitor__source_idx_-4045852888739339153", ios::binary | ios::out);
    if(outfile__array_spikemonitor__source_idx.is_open())
    {
        outfile__array_spikemonitor__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor__source_idx), 100*sizeof(int32_t));
        outfile__array_spikemonitor__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor__source_idx." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_count, dev_array_spikemonitor_count, sizeof(int32_t)*_num__array_spikemonitor_count, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_count;
    outfile__array_spikemonitor_count.open("results/_array_spikemonitor_count_-3651895352503201284", ios::binary | ios::out);
    if(outfile__array_spikemonitor_count.is_open())
    {
        outfile__array_spikemonitor_count.write(reinterpret_cast<char*>(_array_spikemonitor_count), 100*sizeof(int32_t));
        outfile__array_spikemonitor_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_count." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_N, dev_array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_N;
    outfile__array_spikemonitor_N.open("results/_array_spikemonitor_N_73938390545997659", ios::binary | ios::out);
    if(outfile__array_spikemonitor_N.is_open())
    {
        outfile__array_spikemonitor_N.write(reinterpret_cast<char*>(_array_spikemonitor_N), 1*sizeof(int32_t));
        outfile__array_spikemonitor_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_statemonitor_1__indices, dev_array_statemonitor_1__indices, sizeof(int32_t)*_num__array_statemonitor_1__indices, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_statemonitor_1__indices;
    outfile__array_statemonitor_1__indices.open("results/_array_statemonitor_1__indices_5442903941946222241", ios::binary | ios::out);
    if(outfile__array_statemonitor_1__indices.is_open())
    {
        outfile__array_statemonitor_1__indices.write(reinterpret_cast<char*>(_array_statemonitor_1__indices), 10*sizeof(int32_t));
        outfile__array_statemonitor_1__indices.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_1__indices." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_statemonitor_1_N, dev_array_statemonitor_1_N, sizeof(int32_t)*_num__array_statemonitor_1_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_statemonitor_1_N;
    outfile__array_statemonitor_1_N.open("results/_array_statemonitor_1_N_2048012943032035014", ios::binary | ios::out);
    if(outfile__array_statemonitor_1_N.is_open())
    {
        outfile__array_statemonitor_1_N.write(reinterpret_cast<char*>(_array_statemonitor_1_N), 1*sizeof(int32_t));
        outfile__array_statemonitor_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_1_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_statemonitor_2__indices, dev_array_statemonitor_2__indices, sizeof(int32_t)*_num__array_statemonitor_2__indices, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_statemonitor_2__indices;
    outfile__array_statemonitor_2__indices.open("results/_array_statemonitor_2__indices_-2667931346176156168", ios::binary | ios::out);
    if(outfile__array_statemonitor_2__indices.is_open())
    {
        outfile__array_statemonitor_2__indices.write(reinterpret_cast<char*>(_array_statemonitor_2__indices), 10*sizeof(int32_t));
        outfile__array_statemonitor_2__indices.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_2__indices." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_statemonitor_2_N, dev_array_statemonitor_2_N, sizeof(int32_t)*_num__array_statemonitor_2_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_statemonitor_2_N;
    outfile__array_statemonitor_2_N.open("results/_array_statemonitor_2_N_2048011942904034673", ios::binary | ios::out);
    if(outfile__array_statemonitor_2_N.is_open())
    {
        outfile__array_statemonitor_2_N.write(reinterpret_cast<char*>(_array_statemonitor_2_N), 1*sizeof(int32_t));
        outfile__array_statemonitor_2_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_2_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_statemonitor_3__indices, dev_array_statemonitor_3__indices, sizeof(int32_t)*_num__array_statemonitor_3__indices, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_statemonitor_3__indices;
    outfile__array_statemonitor_3__indices.open("results/_array_statemonitor_3__indices_-6406892292391473301", ios::binary | ios::out);
    if(outfile__array_statemonitor_3__indices.is_open())
    {
        outfile__array_statemonitor_3__indices.write(reinterpret_cast<char*>(_array_statemonitor_3__indices), 10*sizeof(int32_t));
        outfile__array_statemonitor_3__indices.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_3__indices." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_statemonitor_3_N, dev_array_statemonitor_3_N, sizeof(int32_t)*_num__array_statemonitor_3_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_statemonitor_3_N;
    outfile__array_statemonitor_3_N.open("results/_array_statemonitor_3_N_2048010943032035048", ios::binary | ios::out);
    if(outfile__array_statemonitor_3_N.is_open())
    {
        outfile__array_statemonitor_3_N.write(reinterpret_cast<char*>(_array_statemonitor_3_N), 1*sizeof(int32_t));
        outfile__array_statemonitor_3_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_3_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_statemonitor__indices, dev_array_statemonitor__indices, sizeof(int32_t)*_num__array_statemonitor__indices, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_statemonitor__indices;
    outfile__array_statemonitor__indices.open("results/_array_statemonitor__indices_6163485638831984707", ios::binary | ios::out);
    if(outfile__array_statemonitor__indices.is_open())
    {
        outfile__array_statemonitor__indices.write(reinterpret_cast<char*>(_array_statemonitor__indices), 100*sizeof(int32_t));
        outfile__array_statemonitor__indices.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor__indices." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_statemonitor_N, dev_array_statemonitor_N, sizeof(int32_t)*_num__array_statemonitor_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_statemonitor_N;
    outfile__array_statemonitor_N.open("results/_array_statemonitor_N_1126150466128921572", ios::binary | ios::out);
    if(outfile__array_statemonitor_N.is_open())
    {
        outfile__array_statemonitor_N.write(reinterpret_cast<char*>(_array_statemonitor_N), 1*sizeof(int32_t));
        outfile__array_statemonitor_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_1_N, dev_array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_1_N;
    outfile__array_synapses_1_N.open("results/_array_synapses_1_N_-7473518110119523383", ios::binary | ios::out);
    if(outfile__array_synapses_1_N.is_open())
    {
        outfile__array_synapses_1_N.write(reinterpret_cast<char*>(_array_synapses_1_N), 1*sizeof(int32_t));
        outfile__array_synapses_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_1_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_2_N, dev_array_synapses_2_N, sizeof(int32_t)*_num__array_synapses_2_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_2_N;
    outfile__array_synapses_2_N.open("results/_array_synapses_2_N_-7473517110247523754", ios::binary | ios::out);
    if(outfile__array_synapses_2_N.is_open())
    {
        outfile__array_synapses_2_N.write(reinterpret_cast<char*>(_array_synapses_2_N), 1*sizeof(int32_t));
        outfile__array_synapses_2_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_2_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_3_N, dev_array_synapses_3_N, sizeof(int32_t)*_num__array_synapses_3_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_3_N;
    outfile__array_synapses_3_N.open("results/_array_synapses_3_N_-7473516110119523361", ios::binary | ios::out);
    if(outfile__array_synapses_3_N.is_open())
    {
        outfile__array_synapses_3_N.write(reinterpret_cast<char*>(_array_synapses_3_N), 1*sizeof(int32_t));
        outfile__array_synapses_3_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_3_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_N, dev_array_synapses_4_N, sizeof(int32_t)*_num__array_synapses_4_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_N;
    outfile__array_synapses_4_N.open("results/_array_synapses_4_N_-7473515110247523932", ios::binary | ios::out);
    if(outfile__array_synapses_4_N.is_open())
    {
        outfile__array_synapses_4_N.write(reinterpret_cast<char*>(_array_synapses_4_N), 1*sizeof(int32_t));
        outfile__array_synapses_4_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_sources, dev_array_synapses_4_sources, sizeof(int32_t)*_num__array_synapses_4_sources, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_sources;
    outfile__array_synapses_4_sources.open("results/_array_synapses_4_sources_1332494506520317198", ios::binary | ios::out);
    if(outfile__array_synapses_4_sources.is_open())
    {
        outfile__array_synapses_4_sources.write(reinterpret_cast<char*>(_array_synapses_4_sources), 10*sizeof(int32_t));
        outfile__array_synapses_4_sources.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_sources." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_sources_1, dev_array_synapses_4_sources_1, sizeof(int32_t)*_num__array_synapses_4_sources_1, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_sources_1;
    outfile__array_synapses_4_sources_1.open("results/_array_synapses_4_sources_1_7344641326406212772", ios::binary | ios::out);
    if(outfile__array_synapses_4_sources_1.is_open())
    {
        outfile__array_synapses_4_sources_1.write(reinterpret_cast<char*>(_array_synapses_4_sources_1), 10*sizeof(int32_t));
        outfile__array_synapses_4_sources_1.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_sources_1." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_sources_2, dev_array_synapses_4_sources_2, sizeof(int32_t)*_num__array_synapses_4_sources_2, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_sources_2;
    outfile__array_synapses_4_sources_2.open("results/_array_synapses_4_sources_2_7344641326406212775", ios::binary | ios::out);
    if(outfile__array_synapses_4_sources_2.is_open())
    {
        outfile__array_synapses_4_sources_2.write(reinterpret_cast<char*>(_array_synapses_4_sources_2), 10*sizeof(int32_t));
        outfile__array_synapses_4_sources_2.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_sources_2." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_sources_3, dev_array_synapses_4_sources_3, sizeof(int32_t)*_num__array_synapses_4_sources_3, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_sources_3;
    outfile__array_synapses_4_sources_3.open("results/_array_synapses_4_sources_3_7344641326406212774", ios::binary | ios::out);
    if(outfile__array_synapses_4_sources_3.is_open())
    {
        outfile__array_synapses_4_sources_3.write(reinterpret_cast<char*>(_array_synapses_4_sources_3), 10*sizeof(int32_t));
        outfile__array_synapses_4_sources_3.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_sources_3." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_sources_4, dev_array_synapses_4_sources_4, sizeof(int32_t)*_num__array_synapses_4_sources_4, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_sources_4;
    outfile__array_synapses_4_sources_4.open("results/_array_synapses_4_sources_4_7344641326406212769", ios::binary | ios::out);
    if(outfile__array_synapses_4_sources_4.is_open())
    {
        outfile__array_synapses_4_sources_4.write(reinterpret_cast<char*>(_array_synapses_4_sources_4), 10*sizeof(int32_t));
        outfile__array_synapses_4_sources_4.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_sources_4." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_sources_5, dev_array_synapses_4_sources_5, sizeof(int32_t)*_num__array_synapses_4_sources_5, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_sources_5;
    outfile__array_synapses_4_sources_5.open("results/_array_synapses_4_sources_5_7344641326406212768", ios::binary | ios::out);
    if(outfile__array_synapses_4_sources_5.is_open())
    {
        outfile__array_synapses_4_sources_5.write(reinterpret_cast<char*>(_array_synapses_4_sources_5), 10*sizeof(int32_t));
        outfile__array_synapses_4_sources_5.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_sources_5." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_sources_6, dev_array_synapses_4_sources_6, sizeof(int32_t)*_num__array_synapses_4_sources_6, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_sources_6;
    outfile__array_synapses_4_sources_6.open("results/_array_synapses_4_sources_6_7344641326406212771", ios::binary | ios::out);
    if(outfile__array_synapses_4_sources_6.is_open())
    {
        outfile__array_synapses_4_sources_6.write(reinterpret_cast<char*>(_array_synapses_4_sources_6), 10*sizeof(int32_t));
        outfile__array_synapses_4_sources_6.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_sources_6." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_sources_7, dev_array_synapses_4_sources_7, sizeof(int32_t)*_num__array_synapses_4_sources_7, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_sources_7;
    outfile__array_synapses_4_sources_7.open("results/_array_synapses_4_sources_7_7344641326406212770", ios::binary | ios::out);
    if(outfile__array_synapses_4_sources_7.is_open())
    {
        outfile__array_synapses_4_sources_7.write(reinterpret_cast<char*>(_array_synapses_4_sources_7), 10*sizeof(int32_t));
        outfile__array_synapses_4_sources_7.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_sources_7." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_sources_8, dev_array_synapses_4_sources_8, sizeof(int32_t)*_num__array_synapses_4_sources_8, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_sources_8;
    outfile__array_synapses_4_sources_8.open("results/_array_synapses_4_sources_8_7344641326406212781", ios::binary | ios::out);
    if(outfile__array_synapses_4_sources_8.is_open())
    {
        outfile__array_synapses_4_sources_8.write(reinterpret_cast<char*>(_array_synapses_4_sources_8), 10*sizeof(int32_t));
        outfile__array_synapses_4_sources_8.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_sources_8." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_sources_9, dev_array_synapses_4_sources_9, sizeof(int32_t)*_num__array_synapses_4_sources_9, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_sources_9;
    outfile__array_synapses_4_sources_9.open("results/_array_synapses_4_sources_9_7344641326406212780", ios::binary | ios::out);
    if(outfile__array_synapses_4_sources_9.is_open())
    {
        outfile__array_synapses_4_sources_9.write(reinterpret_cast<char*>(_array_synapses_4_sources_9), 10*sizeof(int32_t));
        outfile__array_synapses_4_sources_9.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_sources_9." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_targets, dev_array_synapses_4_targets, sizeof(int32_t)*_num__array_synapses_4_targets, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_targets;
    outfile__array_synapses_4_targets.open("results/_array_synapses_4_targets_6943628183544854918", ios::binary | ios::out);
    if(outfile__array_synapses_4_targets.is_open())
    {
        outfile__array_synapses_4_targets.write(reinterpret_cast<char*>(_array_synapses_4_targets), 10*sizeof(int32_t));
        outfile__array_synapses_4_targets.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_targets." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_targets_1, dev_array_synapses_4_targets_1, sizeof(int32_t)*_num__array_synapses_4_targets_1, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_targets_1;
    outfile__array_synapses_4_targets_1.open("results/_array_synapses_4_targets_1_-707878915963310868", ios::binary | ios::out);
    if(outfile__array_synapses_4_targets_1.is_open())
    {
        outfile__array_synapses_4_targets_1.write(reinterpret_cast<char*>(_array_synapses_4_targets_1), 10*sizeof(int32_t));
        outfile__array_synapses_4_targets_1.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_targets_1." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_targets_2, dev_array_synapses_4_targets_2, sizeof(int32_t)*_num__array_synapses_4_targets_2, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_targets_2;
    outfile__array_synapses_4_targets_2.open("results/_array_synapses_4_targets_2_-707878915963310865", ios::binary | ios::out);
    if(outfile__array_synapses_4_targets_2.is_open())
    {
        outfile__array_synapses_4_targets_2.write(reinterpret_cast<char*>(_array_synapses_4_targets_2), 10*sizeof(int32_t));
        outfile__array_synapses_4_targets_2.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_targets_2." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_targets_3, dev_array_synapses_4_targets_3, sizeof(int32_t)*_num__array_synapses_4_targets_3, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_targets_3;
    outfile__array_synapses_4_targets_3.open("results/_array_synapses_4_targets_3_-707878915963310866", ios::binary | ios::out);
    if(outfile__array_synapses_4_targets_3.is_open())
    {
        outfile__array_synapses_4_targets_3.write(reinterpret_cast<char*>(_array_synapses_4_targets_3), 10*sizeof(int32_t));
        outfile__array_synapses_4_targets_3.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_targets_3." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_targets_4, dev_array_synapses_4_targets_4, sizeof(int32_t)*_num__array_synapses_4_targets_4, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_targets_4;
    outfile__array_synapses_4_targets_4.open("results/_array_synapses_4_targets_4_-707878915963310871", ios::binary | ios::out);
    if(outfile__array_synapses_4_targets_4.is_open())
    {
        outfile__array_synapses_4_targets_4.write(reinterpret_cast<char*>(_array_synapses_4_targets_4), 10*sizeof(int32_t));
        outfile__array_synapses_4_targets_4.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_targets_4." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_targets_5, dev_array_synapses_4_targets_5, sizeof(int32_t)*_num__array_synapses_4_targets_5, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_targets_5;
    outfile__array_synapses_4_targets_5.open("results/_array_synapses_4_targets_5_-707878915963310872", ios::binary | ios::out);
    if(outfile__array_synapses_4_targets_5.is_open())
    {
        outfile__array_synapses_4_targets_5.write(reinterpret_cast<char*>(_array_synapses_4_targets_5), 10*sizeof(int32_t));
        outfile__array_synapses_4_targets_5.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_targets_5." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_targets_6, dev_array_synapses_4_targets_6, sizeof(int32_t)*_num__array_synapses_4_targets_6, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_targets_6;
    outfile__array_synapses_4_targets_6.open("results/_array_synapses_4_targets_6_-707878915963310869", ios::binary | ios::out);
    if(outfile__array_synapses_4_targets_6.is_open())
    {
        outfile__array_synapses_4_targets_6.write(reinterpret_cast<char*>(_array_synapses_4_targets_6), 10*sizeof(int32_t));
        outfile__array_synapses_4_targets_6.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_targets_6." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_targets_7, dev_array_synapses_4_targets_7, sizeof(int32_t)*_num__array_synapses_4_targets_7, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_targets_7;
    outfile__array_synapses_4_targets_7.open("results/_array_synapses_4_targets_7_-707878915963310870", ios::binary | ios::out);
    if(outfile__array_synapses_4_targets_7.is_open())
    {
        outfile__array_synapses_4_targets_7.write(reinterpret_cast<char*>(_array_synapses_4_targets_7), 10*sizeof(int32_t));
        outfile__array_synapses_4_targets_7.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_targets_7." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_targets_8, dev_array_synapses_4_targets_8, sizeof(int32_t)*_num__array_synapses_4_targets_8, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_targets_8;
    outfile__array_synapses_4_targets_8.open("results/_array_synapses_4_targets_8_-707878915963310875", ios::binary | ios::out);
    if(outfile__array_synapses_4_targets_8.is_open())
    {
        outfile__array_synapses_4_targets_8.write(reinterpret_cast<char*>(_array_synapses_4_targets_8), 10*sizeof(int32_t));
        outfile__array_synapses_4_targets_8.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_targets_8." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_targets_9, dev_array_synapses_4_targets_9, sizeof(int32_t)*_num__array_synapses_4_targets_9, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_targets_9;
    outfile__array_synapses_4_targets_9.open("results/_array_synapses_4_targets_9_-707878915963310876", ios::binary | ios::out);
    if(outfile__array_synapses_4_targets_9.is_open())
    {
        outfile__array_synapses_4_targets_9.write(reinterpret_cast<char*>(_array_synapses_4_targets_9), 10*sizeof(int32_t));
        outfile__array_synapses_4_targets_9.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_targets_9." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_N, dev_array_synapses_5_N, sizeof(int32_t)*_num__array_synapses_5_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_N;
    outfile__array_synapses_5_N.open("results/_array_synapses_5_N_-7473514110119523539", ios::binary | ios::out);
    if(outfile__array_synapses_5_N.is_open())
    {
        outfile__array_synapses_5_N.write(reinterpret_cast<char*>(_array_synapses_5_N), 1*sizeof(int32_t));
        outfile__array_synapses_5_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_sources, dev_array_synapses_5_sources, sizeof(int32_t)*_num__array_synapses_5_sources, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_sources;
    outfile__array_synapses_5_sources.open("results/_array_synapses_5_sources_-2410148184643561825", ios::binary | ios::out);
    if(outfile__array_synapses_5_sources.is_open())
    {
        outfile__array_synapses_5_sources.write(reinterpret_cast<char*>(_array_synapses_5_sources), 10*sizeof(int32_t));
        outfile__array_synapses_5_sources.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_sources." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_sources_1, dev_array_synapses_5_sources_1, sizeof(int32_t)*_num__array_synapses_5_sources_1, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_sources_1;
    outfile__array_synapses_5_sources_1.open("results/_array_synapses_5_sources_1_6042193950247813389", ios::binary | ios::out);
    if(outfile__array_synapses_5_sources_1.is_open())
    {
        outfile__array_synapses_5_sources_1.write(reinterpret_cast<char*>(_array_synapses_5_sources_1), 10*sizeof(int32_t));
        outfile__array_synapses_5_sources_1.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_sources_1." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_sources_2, dev_array_synapses_5_sources_2, sizeof(int32_t)*_num__array_synapses_5_sources_2, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_sources_2;
    outfile__array_synapses_5_sources_2.open("results/_array_synapses_5_sources_2_6042193950247813390", ios::binary | ios::out);
    if(outfile__array_synapses_5_sources_2.is_open())
    {
        outfile__array_synapses_5_sources_2.write(reinterpret_cast<char*>(_array_synapses_5_sources_2), 10*sizeof(int32_t));
        outfile__array_synapses_5_sources_2.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_sources_2." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_sources_3, dev_array_synapses_5_sources_3, sizeof(int32_t)*_num__array_synapses_5_sources_3, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_sources_3;
    outfile__array_synapses_5_sources_3.open("results/_array_synapses_5_sources_3_6042193950247813391", ios::binary | ios::out);
    if(outfile__array_synapses_5_sources_3.is_open())
    {
        outfile__array_synapses_5_sources_3.write(reinterpret_cast<char*>(_array_synapses_5_sources_3), 10*sizeof(int32_t));
        outfile__array_synapses_5_sources_3.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_sources_3." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_sources_4, dev_array_synapses_5_sources_4, sizeof(int32_t)*_num__array_synapses_5_sources_4, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_sources_4;
    outfile__array_synapses_5_sources_4.open("results/_array_synapses_5_sources_4_6042193950247813384", ios::binary | ios::out);
    if(outfile__array_synapses_5_sources_4.is_open())
    {
        outfile__array_synapses_5_sources_4.write(reinterpret_cast<char*>(_array_synapses_5_sources_4), 10*sizeof(int32_t));
        outfile__array_synapses_5_sources_4.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_sources_4." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_sources_5, dev_array_synapses_5_sources_5, sizeof(int32_t)*_num__array_synapses_5_sources_5, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_sources_5;
    outfile__array_synapses_5_sources_5.open("results/_array_synapses_5_sources_5_6042193950247813385", ios::binary | ios::out);
    if(outfile__array_synapses_5_sources_5.is_open())
    {
        outfile__array_synapses_5_sources_5.write(reinterpret_cast<char*>(_array_synapses_5_sources_5), 10*sizeof(int32_t));
        outfile__array_synapses_5_sources_5.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_sources_5." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_sources_6, dev_array_synapses_5_sources_6, sizeof(int32_t)*_num__array_synapses_5_sources_6, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_sources_6;
    outfile__array_synapses_5_sources_6.open("results/_array_synapses_5_sources_6_6042193950247813386", ios::binary | ios::out);
    if(outfile__array_synapses_5_sources_6.is_open())
    {
        outfile__array_synapses_5_sources_6.write(reinterpret_cast<char*>(_array_synapses_5_sources_6), 10*sizeof(int32_t));
        outfile__array_synapses_5_sources_6.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_sources_6." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_sources_7, dev_array_synapses_5_sources_7, sizeof(int32_t)*_num__array_synapses_5_sources_7, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_sources_7;
    outfile__array_synapses_5_sources_7.open("results/_array_synapses_5_sources_7_6042193950247813387", ios::binary | ios::out);
    if(outfile__array_synapses_5_sources_7.is_open())
    {
        outfile__array_synapses_5_sources_7.write(reinterpret_cast<char*>(_array_synapses_5_sources_7), 10*sizeof(int32_t));
        outfile__array_synapses_5_sources_7.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_sources_7." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_sources_8, dev_array_synapses_5_sources_8, sizeof(int32_t)*_num__array_synapses_5_sources_8, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_sources_8;
    outfile__array_synapses_5_sources_8.open("results/_array_synapses_5_sources_8_6042193950247813380", ios::binary | ios::out);
    if(outfile__array_synapses_5_sources_8.is_open())
    {
        outfile__array_synapses_5_sources_8.write(reinterpret_cast<char*>(_array_synapses_5_sources_8), 10*sizeof(int32_t));
        outfile__array_synapses_5_sources_8.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_sources_8." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_sources_9, dev_array_synapses_5_sources_9, sizeof(int32_t)*_num__array_synapses_5_sources_9, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_sources_9;
    outfile__array_synapses_5_sources_9.open("results/_array_synapses_5_sources_9_6042193950247813381", ios::binary | ios::out);
    if(outfile__array_synapses_5_sources_9.is_open())
    {
        outfile__array_synapses_5_sources_9.write(reinterpret_cast<char*>(_array_synapses_5_sources_9), 10*sizeof(int32_t));
        outfile__array_synapses_5_sources_9.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_sources_9." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_targets, dev_array_synapses_5_targets, sizeof(int32_t)*_num__array_synapses_5_targets, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_targets;
    outfile__array_synapses_5_targets.open("results/_array_synapses_5_targets_-1134252501759464813", ios::binary | ios::out);
    if(outfile__array_synapses_5_targets.is_open())
    {
        outfile__array_synapses_5_targets.write(reinterpret_cast<char*>(_array_synapses_5_targets), 10*sizeof(int32_t));
        outfile__array_synapses_5_targets.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_targets." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_targets_1, dev_array_synapses_5_targets_1, sizeof(int32_t)*_num__array_synapses_5_targets_1, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_targets_1;
    outfile__array_synapses_5_targets_1.open("results/_array_synapses_5_targets_1_39328386038150441", ios::binary | ios::out);
    if(outfile__array_synapses_5_targets_1.is_open())
    {
        outfile__array_synapses_5_targets_1.write(reinterpret_cast<char*>(_array_synapses_5_targets_1), 10*sizeof(int32_t));
        outfile__array_synapses_5_targets_1.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_targets_1." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_targets_2, dev_array_synapses_5_targets_2, sizeof(int32_t)*_num__array_synapses_5_targets_2, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_targets_2;
    outfile__array_synapses_5_targets_2.open("results/_array_synapses_5_targets_2_39328386038150442", ios::binary | ios::out);
    if(outfile__array_synapses_5_targets_2.is_open())
    {
        outfile__array_synapses_5_targets_2.write(reinterpret_cast<char*>(_array_synapses_5_targets_2), 10*sizeof(int32_t));
        outfile__array_synapses_5_targets_2.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_targets_2." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_targets_3, dev_array_synapses_5_targets_3, sizeof(int32_t)*_num__array_synapses_5_targets_3, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_targets_3;
    outfile__array_synapses_5_targets_3.open("results/_array_synapses_5_targets_3_39328386038150443", ios::binary | ios::out);
    if(outfile__array_synapses_5_targets_3.is_open())
    {
        outfile__array_synapses_5_targets_3.write(reinterpret_cast<char*>(_array_synapses_5_targets_3), 10*sizeof(int32_t));
        outfile__array_synapses_5_targets_3.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_targets_3." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_targets_4, dev_array_synapses_5_targets_4, sizeof(int32_t)*_num__array_synapses_5_targets_4, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_targets_4;
    outfile__array_synapses_5_targets_4.open("results/_array_synapses_5_targets_4_39328386038150444", ios::binary | ios::out);
    if(outfile__array_synapses_5_targets_4.is_open())
    {
        outfile__array_synapses_5_targets_4.write(reinterpret_cast<char*>(_array_synapses_5_targets_4), 10*sizeof(int32_t));
        outfile__array_synapses_5_targets_4.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_targets_4." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_targets_5, dev_array_synapses_5_targets_5, sizeof(int32_t)*_num__array_synapses_5_targets_5, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_targets_5;
    outfile__array_synapses_5_targets_5.open("results/_array_synapses_5_targets_5_39328386038150445", ios::binary | ios::out);
    if(outfile__array_synapses_5_targets_5.is_open())
    {
        outfile__array_synapses_5_targets_5.write(reinterpret_cast<char*>(_array_synapses_5_targets_5), 10*sizeof(int32_t));
        outfile__array_synapses_5_targets_5.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_targets_5." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_targets_6, dev_array_synapses_5_targets_6, sizeof(int32_t)*_num__array_synapses_5_targets_6, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_targets_6;
    outfile__array_synapses_5_targets_6.open("results/_array_synapses_5_targets_6_39328386038150446", ios::binary | ios::out);
    if(outfile__array_synapses_5_targets_6.is_open())
    {
        outfile__array_synapses_5_targets_6.write(reinterpret_cast<char*>(_array_synapses_5_targets_6), 10*sizeof(int32_t));
        outfile__array_synapses_5_targets_6.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_targets_6." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_targets_7, dev_array_synapses_5_targets_7, sizeof(int32_t)*_num__array_synapses_5_targets_7, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_targets_7;
    outfile__array_synapses_5_targets_7.open("results/_array_synapses_5_targets_7_39328386038150447", ios::binary | ios::out);
    if(outfile__array_synapses_5_targets_7.is_open())
    {
        outfile__array_synapses_5_targets_7.write(reinterpret_cast<char*>(_array_synapses_5_targets_7), 10*sizeof(int32_t));
        outfile__array_synapses_5_targets_7.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_targets_7." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_targets_8, dev_array_synapses_5_targets_8, sizeof(int32_t)*_num__array_synapses_5_targets_8, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_targets_8;
    outfile__array_synapses_5_targets_8.open("results/_array_synapses_5_targets_8_39328386038150432", ios::binary | ios::out);
    if(outfile__array_synapses_5_targets_8.is_open())
    {
        outfile__array_synapses_5_targets_8.write(reinterpret_cast<char*>(_array_synapses_5_targets_8), 10*sizeof(int32_t));
        outfile__array_synapses_5_targets_8.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_targets_8." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_targets_9, dev_array_synapses_5_targets_9, sizeof(int32_t)*_num__array_synapses_5_targets_9, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_targets_9;
    outfile__array_synapses_5_targets_9.open("results/_array_synapses_5_targets_9_39328386038150433", ios::binary | ios::out);
    if(outfile__array_synapses_5_targets_9.is_open())
    {
        outfile__array_synapses_5_targets_9.write(reinterpret_cast<char*>(_array_synapses_5_targets_9), 10*sizeof(int32_t));
        outfile__array_synapses_5_targets_9.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_targets_9." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_6_N, dev_array_synapses_6_N, sizeof(int32_t)*_num__array_synapses_6_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_6_N;
    outfile__array_synapses_6_N.open("results/_array_synapses_6_N_-7473513110247523910", ios::binary | ios::out);
    if(outfile__array_synapses_6_N.is_open())
    {
        outfile__array_synapses_6_N.write(reinterpret_cast<char*>(_array_synapses_6_N), 1*sizeof(int32_t));
        outfile__array_synapses_6_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_6_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_N, dev_array_synapses_N, sizeof(int32_t)*_num__array_synapses_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_N;
    outfile__array_synapses_N.open("results/_array_synapses_N_-7833853409752232273", ios::binary | ios::out);
    if(outfile__array_synapses_N.is_open())
    {
        outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(int32_t));
        outfile__array_synapses_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_N." << endl;
    }

    _dynamic_array_ratemonitor_1_rate = dev_dynamic_array_ratemonitor_1_rate;
    ofstream outfile__dynamic_array_ratemonitor_1_rate;
    outfile__dynamic_array_ratemonitor_1_rate.open("results/_dynamic_array_ratemonitor_1_rate_-3835024069465499341", ios::binary | ios::out);
    if(outfile__dynamic_array_ratemonitor_1_rate.is_open())
    {
        outfile__dynamic_array_ratemonitor_1_rate.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_ratemonitor_1_rate[0])), _dynamic_array_ratemonitor_1_rate.size()*sizeof(double));
        outfile__dynamic_array_ratemonitor_1_rate.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_ratemonitor_1_rate." << endl;
    }
    _dynamic_array_ratemonitor_1_t = dev_dynamic_array_ratemonitor_1_t;
    ofstream outfile__dynamic_array_ratemonitor_1_t;
    outfile__dynamic_array_ratemonitor_1_t.open("results/_dynamic_array_ratemonitor_1_t_4415938933556539630", ios::binary | ios::out);
    if(outfile__dynamic_array_ratemonitor_1_t.is_open())
    {
        outfile__dynamic_array_ratemonitor_1_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_ratemonitor_1_t[0])), _dynamic_array_ratemonitor_1_t.size()*sizeof(double));
        outfile__dynamic_array_ratemonitor_1_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_ratemonitor_1_t." << endl;
    }
    _dynamic_array_ratemonitor_2_rate = dev_dynamic_array_ratemonitor_2_rate;
    ofstream outfile__dynamic_array_ratemonitor_2_rate;
    outfile__dynamic_array_ratemonitor_2_rate.open("results/_dynamic_array_ratemonitor_2_rate_-3104250090693381814", ios::binary | ios::out);
    if(outfile__dynamic_array_ratemonitor_2_rate.is_open())
    {
        outfile__dynamic_array_ratemonitor_2_rate.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_ratemonitor_2_rate[0])), _dynamic_array_ratemonitor_2_rate.size()*sizeof(double));
        outfile__dynamic_array_ratemonitor_2_rate.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_ratemonitor_2_rate." << endl;
    }
    _dynamic_array_ratemonitor_2_t = dev_dynamic_array_ratemonitor_2_t;
    ofstream outfile__dynamic_array_ratemonitor_2_t;
    outfile__dynamic_array_ratemonitor_2_t.open("results/_dynamic_array_ratemonitor_2_t_4415939933684539793", ios::binary | ios::out);
    if(outfile__dynamic_array_ratemonitor_2_t.is_open())
    {
        outfile__dynamic_array_ratemonitor_2_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_ratemonitor_2_t[0])), _dynamic_array_ratemonitor_2_t.size()*sizeof(double));
        outfile__dynamic_array_ratemonitor_2_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_ratemonitor_2_t." << endl;
    }
    _dynamic_array_ratemonitor_3_rate = dev_dynamic_array_ratemonitor_3_rate;
    ofstream outfile__dynamic_array_ratemonitor_3_rate;
    outfile__dynamic_array_ratemonitor_3_rate.open("results/_dynamic_array_ratemonitor_3_rate_-2041395251008424639", ios::binary | ios::out);
    if(outfile__dynamic_array_ratemonitor_3_rate.is_open())
    {
        outfile__dynamic_array_ratemonitor_3_rate.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_ratemonitor_3_rate[0])), _dynamic_array_ratemonitor_3_rate.size()*sizeof(double));
        outfile__dynamic_array_ratemonitor_3_rate.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_ratemonitor_3_rate." << endl;
    }
    _dynamic_array_ratemonitor_3_t = dev_dynamic_array_ratemonitor_3_t;
    ofstream outfile__dynamic_array_ratemonitor_3_t;
    outfile__dynamic_array_ratemonitor_3_t.open("results/_dynamic_array_ratemonitor_3_t_4415940933556539416", ios::binary | ios::out);
    if(outfile__dynamic_array_ratemonitor_3_t.is_open())
    {
        outfile__dynamic_array_ratemonitor_3_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_ratemonitor_3_t[0])), _dynamic_array_ratemonitor_3_t.size()*sizeof(double));
        outfile__dynamic_array_ratemonitor_3_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_ratemonitor_3_t." << endl;
    }
    _dynamic_array_ratemonitor_rate = dev_dynamic_array_ratemonitor_rate;
    ofstream outfile__dynamic_array_ratemonitor_rate;
    outfile__dynamic_array_ratemonitor_rate.open("results/_dynamic_array_ratemonitor_rate_-4222103208777388567", ios::binary | ios::out);
    if(outfile__dynamic_array_ratemonitor_rate.is_open())
    {
        outfile__dynamic_array_ratemonitor_rate.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_ratemonitor_rate[0])), _dynamic_array_ratemonitor_rate.size()*sizeof(double));
        outfile__dynamic_array_ratemonitor_rate.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_ratemonitor_rate." << endl;
    }
    _dynamic_array_ratemonitor_t = dev_dynamic_array_ratemonitor_t;
    ofstream outfile__dynamic_array_ratemonitor_t;
    outfile__dynamic_array_ratemonitor_t.open("results/_dynamic_array_ratemonitor_t_8906058387149996872", ios::binary | ios::out);
    if(outfile__dynamic_array_ratemonitor_t.is_open())
    {
        outfile__dynamic_array_ratemonitor_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_ratemonitor_t[0])), _dynamic_array_ratemonitor_t.size()*sizeof(double));
        outfile__dynamic_array_ratemonitor_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_ratemonitor_t." << endl;
    }
    _dynamic_array_spikemonitor_1_i = dev_dynamic_array_spikemonitor_1_i;
    ofstream outfile__dynamic_array_spikemonitor_1_i;
    outfile__dynamic_array_spikemonitor_1_i.open("results/_dynamic_array_spikemonitor_1_i_3316870184202501388", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_1_i.is_open())
    {
        outfile__dynamic_array_spikemonitor_1_i.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_1_i[0])), _dynamic_array_spikemonitor_1_i.size()*sizeof(int32_t));
        outfile__dynamic_array_spikemonitor_1_i.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_1_i." << endl;
    }
    _dynamic_array_spikemonitor_1_t = dev_dynamic_array_spikemonitor_1_t;
    ofstream outfile__dynamic_array_spikemonitor_1_t;
    outfile__dynamic_array_spikemonitor_1_t.open("results/_dynamic_array_spikemonitor_1_t_3316870184202501393", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_1_t.is_open())
    {
        outfile__dynamic_array_spikemonitor_1_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_1_t[0])), _dynamic_array_spikemonitor_1_t.size()*sizeof(double));
        outfile__dynamic_array_spikemonitor_1_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_1_t." << endl;
    }
    _dynamic_array_spikemonitor_2_i = dev_dynamic_array_spikemonitor_2_i;
    ofstream outfile__dynamic_array_spikemonitor_2_i;
    outfile__dynamic_array_spikemonitor_2_i.open("results/_dynamic_array_spikemonitor_2_i_3316867184010500899", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_2_i.is_open())
    {
        outfile__dynamic_array_spikemonitor_2_i.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_2_i[0])), _dynamic_array_spikemonitor_2_i.size()*sizeof(int32_t));
        outfile__dynamic_array_spikemonitor_2_i.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_2_i." << endl;
    }
    _dynamic_array_spikemonitor_2_t = dev_dynamic_array_spikemonitor_2_t;
    ofstream outfile__dynamic_array_spikemonitor_2_t;
    outfile__dynamic_array_spikemonitor_2_t.open("results/_dynamic_array_spikemonitor_2_t_3316867184010500926", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_2_t.is_open())
    {
        outfile__dynamic_array_spikemonitor_2_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_2_t[0])), _dynamic_array_spikemonitor_2_t.size()*sizeof(double));
        outfile__dynamic_array_spikemonitor_2_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_2_t." << endl;
    }
    _dynamic_array_spikemonitor_3_i = dev_dynamic_array_spikemonitor_3_i;
    ofstream outfile__dynamic_array_spikemonitor_3_i;
    outfile__dynamic_array_spikemonitor_3_i.open("results/_dynamic_array_spikemonitor_3_i_3316868184138501306", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_3_i.is_open())
    {
        outfile__dynamic_array_spikemonitor_3_i.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_3_i[0])), _dynamic_array_spikemonitor_3_i.size()*sizeof(int32_t));
        outfile__dynamic_array_spikemonitor_3_i.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_3_i." << endl;
    }
    _dynamic_array_spikemonitor_3_t = dev_dynamic_array_spikemonitor_3_t;
    ofstream outfile__dynamic_array_spikemonitor_3_t;
    outfile__dynamic_array_spikemonitor_3_t.open("results/_dynamic_array_spikemonitor_3_t_3316868184138501287", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_3_t.is_open())
    {
        outfile__dynamic_array_spikemonitor_3_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_3_t[0])), _dynamic_array_spikemonitor_3_t.size()*sizeof(double));
        outfile__dynamic_array_spikemonitor_3_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_3_t." << endl;
    }
    _dynamic_array_spikemonitor_i = dev_dynamic_array_spikemonitor_i;
    ofstream outfile__dynamic_array_spikemonitor_i;
    outfile__dynamic_array_spikemonitor_i.open("results/_dynamic_array_spikemonitor_i_3873805716461528078", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_i.is_open())
    {
        outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_i[0])), _dynamic_array_spikemonitor_i.size()*sizeof(int32_t));
        outfile__dynamic_array_spikemonitor_i.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
    }
    _dynamic_array_spikemonitor_t = dev_dynamic_array_spikemonitor_t;
    ofstream outfile__dynamic_array_spikemonitor_t;
    outfile__dynamic_array_spikemonitor_t.open("results/_dynamic_array_spikemonitor_t_3873805716461528083", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_t.is_open())
    {
        outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_t[0])), _dynamic_array_spikemonitor_t.size()*sizeof(double));
        outfile__dynamic_array_spikemonitor_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_t." << endl;
    }
    _dynamic_array_statemonitor_1_t = dev_dynamic_array_statemonitor_1_t;
    ofstream outfile__dynamic_array_statemonitor_1_t;
    outfile__dynamic_array_statemonitor_1_t.open("results/_dynamic_array_statemonitor_1_t_-8409868653121327110", ios::binary | ios::out);
    if(outfile__dynamic_array_statemonitor_1_t.is_open())
    {
        outfile__dynamic_array_statemonitor_1_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_statemonitor_1_t[0])), _dynamic_array_statemonitor_1_t.size()*sizeof(double));
        outfile__dynamic_array_statemonitor_1_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_statemonitor_1_t." << endl;
    }
    _dynamic_array_statemonitor_2_t = dev_dynamic_array_statemonitor_2_t;
    ofstream outfile__dynamic_array_statemonitor_2_t;
    outfile__dynamic_array_statemonitor_2_t.open("results/_dynamic_array_statemonitor_2_t_-8409865652993326947", ios::binary | ios::out);
    if(outfile__dynamic_array_statemonitor_2_t.is_open())
    {
        outfile__dynamic_array_statemonitor_2_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_statemonitor_2_t[0])), _dynamic_array_statemonitor_2_t.size()*sizeof(double));
        outfile__dynamic_array_statemonitor_2_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_statemonitor_2_t." << endl;
    }
    _dynamic_array_statemonitor_3_t = dev_dynamic_array_statemonitor_3_t;
    ofstream outfile__dynamic_array_statemonitor_3_t;
    outfile__dynamic_array_statemonitor_3_t.open("results/_dynamic_array_statemonitor_3_t_-8409866653121327340", ios::binary | ios::out);
    if(outfile__dynamic_array_statemonitor_3_t.is_open())
    {
        outfile__dynamic_array_statemonitor_3_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_statemonitor_3_t[0])), _dynamic_array_statemonitor_3_t.size()*sizeof(double));
        outfile__dynamic_array_statemonitor_3_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_statemonitor_3_t." << endl;
    }
    _dynamic_array_statemonitor_t = dev_dynamic_array_statemonitor_t;
    ofstream outfile__dynamic_array_statemonitor_t;
    outfile__dynamic_array_statemonitor_t.open("results/_dynamic_array_statemonitor_t_6620044162385838772", ios::binary | ios::out);
    if(outfile__dynamic_array_statemonitor_t.is_open())
    {
        outfile__dynamic_array_statemonitor_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_statemonitor_t[0])), _dynamic_array_statemonitor_t.size()*sizeof(double));
        outfile__dynamic_array_statemonitor_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_statemonitor_t." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1__synaptic_post;
    outfile__dynamic_array_synapses_1__synaptic_post.open("results/_dynamic_array_synapses_1__synaptic_post_-4367449856540212009", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_1__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1__synaptic_post[0])), _dynamic_array_synapses_1__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1__synaptic_pre;
    outfile__dynamic_array_synapses_1__synaptic_pre.open("results/_dynamic_array_synapses_1__synaptic_pre_1368795276670783483", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_1__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1__synaptic_pre[0])), _dynamic_array_synapses_1__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_delay;
    outfile__dynamic_array_synapses_1_delay.open("results/_dynamic_array_synapses_1_delay_3502988388203646216", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_delay.is_open())
    {
        outfile__dynamic_array_synapses_1_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_delay[0])), _dynamic_array_synapses_1_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_delay_1;
    outfile__dynamic_array_synapses_1_delay_1.open("results/_dynamic_array_synapses_1_delay_1_1655075671283817630", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_delay_1.is_open())
    {
        outfile__dynamic_array_synapses_1_delay_1.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_delay_1[0])), _dynamic_array_synapses_1_delay_1.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_delay_1.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_delay_1." << endl;
    }
    _dynamic_array_synapses_1_N_incoming = dev_dynamic_array_synapses_1_N_incoming;
    ofstream outfile__dynamic_array_synapses_1_N_incoming;
    outfile__dynamic_array_synapses_1_N_incoming.open("results/_dynamic_array_synapses_1_N_incoming_-5364435978754666149", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_1_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_N_incoming[0])), _dynamic_array_synapses_1_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_N_incoming." << endl;
    }
    _dynamic_array_synapses_1_N_outgoing = dev_dynamic_array_synapses_1_N_outgoing;
    ofstream outfile__dynamic_array_synapses_1_N_outgoing;
    outfile__dynamic_array_synapses_1_N_outgoing.open("results/_dynamic_array_synapses_1_N_outgoing_7721560298971024321", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_1_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_N_outgoing[0])), _dynamic_array_synapses_1_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_N_outgoing." << endl;
    }
    _dynamic_array_synapses_1_w_CFPKJ = dev_dynamic_array_synapses_1_w_CFPKJ;
    ofstream outfile__dynamic_array_synapses_1_w_CFPKJ;
    outfile__dynamic_array_synapses_1_w_CFPKJ.open("results/_dynamic_array_synapses_1_w_CFPKJ_7591340365364892479", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_w_CFPKJ.is_open())
    {
        outfile__dynamic_array_synapses_1_w_CFPKJ.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_w_CFPKJ[0])), _dynamic_array_synapses_1_w_CFPKJ.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_w_CFPKJ.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_w_CFPKJ." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2__synaptic_post;
    outfile__dynamic_array_synapses_2__synaptic_post.open("results/_dynamic_array_synapses_2__synaptic_post_3137150149881540902", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_2__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2__synaptic_post[0])), _dynamic_array_synapses_2__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2__synaptic_pre;
    outfile__dynamic_array_synapses_2__synaptic_pre.open("results/_dynamic_array_synapses_2__synaptic_pre_405387568754284564", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_2__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2__synaptic_pre[0])), _dynamic_array_synapses_2__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_delay;
    outfile__dynamic_array_synapses_2_delay.open("results/_dynamic_array_synapses_2_delay_842331536822541471", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_delay.is_open())
    {
        outfile__dynamic_array_synapses_2_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_delay[0])), _dynamic_array_synapses_2_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_2_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_delay_1;
    outfile__dynamic_array_synapses_2_delay_1.open("results/_dynamic_array_synapses_2_delay_1_-8886276883156445363", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_delay_1.is_open())
    {
        outfile__dynamic_array_synapses_2_delay_1.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_delay_1[0])), _dynamic_array_synapses_2_delay_1.size()*sizeof(double));
        outfile__dynamic_array_synapses_2_delay_1.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_delay_1." << endl;
    }
    _dynamic_array_synapses_2_N_incoming = dev_dynamic_array_synapses_2_N_incoming;
    ofstream outfile__dynamic_array_synapses_2_N_incoming;
    outfile__dynamic_array_synapses_2_N_incoming.open("results/_dynamic_array_synapses_2_N_incoming_-3275600429616745766", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_2_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_N_incoming[0])), _dynamic_array_synapses_2_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_N_incoming." << endl;
    }
    _dynamic_array_synapses_2_N_outgoing = dev_dynamic_array_synapses_2_N_outgoing;
    ofstream outfile__dynamic_array_synapses_2_N_outgoing;
    outfile__dynamic_array_synapses_2_N_outgoing.open("results/_dynamic_array_synapses_2_N_outgoing_6717848568732218728", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_2_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_N_outgoing[0])), _dynamic_array_synapses_2_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_N_outgoing." << endl;
    }
    _dynamic_array_synapses_2_w_GRGO = dev_dynamic_array_synapses_2_w_GRGO;
    ofstream outfile__dynamic_array_synapses_2_w_GRGO;
    outfile__dynamic_array_synapses_2_w_GRGO.open("results/_dynamic_array_synapses_2_w_GRGO_2918502671259078648", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_w_GRGO.is_open())
    {
        outfile__dynamic_array_synapses_2_w_GRGO.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_w_GRGO[0])), _dynamic_array_synapses_2_w_GRGO.size()*sizeof(double));
        outfile__dynamic_array_synapses_2_w_GRGO.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_w_GRGO." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3__synaptic_post;
    outfile__dynamic_array_synapses_3__synaptic_post.open("results/_dynamic_array_synapses_3__synaptic_post_-6807719042071735663", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_3__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3__synaptic_post[0])), _dynamic_array_synapses_3__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_3__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3__synaptic_pre;
    outfile__dynamic_array_synapses_3__synaptic_pre.open("results/_dynamic_array_synapses_3__synaptic_pre_-2450464939086801735", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_3__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3__synaptic_pre[0])), _dynamic_array_synapses_3__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_3__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3_delay;
    outfile__dynamic_array_synapses_3_delay.open("results/_dynamic_array_synapses_3_delay_-2489879914270044434", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_delay.is_open())
    {
        outfile__dynamic_array_synapses_3_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3_delay[0])), _dynamic_array_synapses_3_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_3_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_delay." << endl;
    }
    _dynamic_array_synapses_3_N_incoming = dev_dynamic_array_synapses_3_N_incoming;
    ofstream outfile__dynamic_array_synapses_3_N_incoming;
    outfile__dynamic_array_synapses_3_N_incoming.open("results/_dynamic_array_synapses_3_N_incoming_-2106702523814753163", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_3_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3_N_incoming[0])), _dynamic_array_synapses_3_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_3_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_N_incoming." << endl;
    }
    _dynamic_array_synapses_3_N_outgoing = dev_dynamic_array_synapses_3_N_outgoing;
    ofstream outfile__dynamic_array_synapses_3_N_outgoing;
    outfile__dynamic_array_synapses_3_N_outgoing.open("results/_dynamic_array_synapses_3_N_outgoing_-3421667455987502645", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_3_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3_N_outgoing[0])), _dynamic_array_synapses_3_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_3_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_N_outgoing." << endl;
    }
    _dynamic_array_synapses_3_w_GOGR = dev_dynamic_array_synapses_3_w_GOGR;
    ofstream outfile__dynamic_array_synapses_3_w_GOGR;
    outfile__dynamic_array_synapses_3_w_GOGR.open("results/_dynamic_array_synapses_3_w_GOGR_-8251411578444143129", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_w_GOGR.is_open())
    {
        outfile__dynamic_array_synapses_3_w_GOGR.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3_w_GOGR[0])), _dynamic_array_synapses_3_w_GOGR.size()*sizeof(double));
        outfile__dynamic_array_synapses_3_w_GOGR.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_w_GOGR." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4__synaptic_post;
    outfile__dynamic_array_synapses_4__synaptic_post.open("results/_dynamic_array_synapses_4__synaptic_post_-4092947258271513536", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_4__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_4__synaptic_post[0])), _dynamic_array_synapses_4__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_4__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4__synaptic_pre;
    outfile__dynamic_array_synapses_4__synaptic_pre.open("results/_dynamic_array_synapses_4__synaptic_pre_5068518783656269522", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_4__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_4__synaptic_pre[0])), _dynamic_array_synapses_4__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_4__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4_delay;
    outfile__dynamic_array_synapses_4_delay.open("results/_dynamic_array_synapses_4_delay_-9070136677468453939", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_delay.is_open())
    {
        outfile__dynamic_array_synapses_4_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_4_delay[0])), _dynamic_array_synapses_4_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_4_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4_delay_1;
    outfile__dynamic_array_synapses_4_delay_1.open("results/_dynamic_array_synapses_4_delay_1_7747394344862426923", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_delay_1.is_open())
    {
        outfile__dynamic_array_synapses_4_delay_1.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_4_delay_1[0])), _dynamic_array_synapses_4_delay_1.size()*sizeof(double));
        outfile__dynamic_array_synapses_4_delay_1.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_delay_1." << endl;
    }
    _dynamic_array_synapses_4_N_incoming = dev_dynamic_array_synapses_4_N_incoming;
    ofstream outfile__dynamic_array_synapses_4_N_incoming;
    outfile__dynamic_array_synapses_4_N_incoming.open("results/_dynamic_array_synapses_4_N_incoming_-3505121638531903940", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_4_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_4_N_incoming[0])), _dynamic_array_synapses_4_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_4_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_N_incoming." << endl;
    }
    _dynamic_array_synapses_4_N_outgoing = dev_dynamic_array_synapses_4_N_outgoing;
    ofstream outfile__dynamic_array_synapses_4_N_outgoing;
    outfile__dynamic_array_synapses_4_N_outgoing.open("results/_dynamic_array_synapses_4_N_outgoing_46890498304195442", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_4_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_4_N_outgoing[0])), _dynamic_array_synapses_4_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_4_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_N_outgoing." << endl;
    }
    _dynamic_array_synapses_4_w_GRPKJ = dev_dynamic_array_synapses_4_w_GRPKJ;
    ofstream outfile__dynamic_array_synapses_4_w_GRPKJ;
    outfile__dynamic_array_synapses_4_w_GRPKJ.open("results/_dynamic_array_synapses_4_w_GRPKJ_8142285067335974478", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_w_GRPKJ.is_open())
    {
        outfile__dynamic_array_synapses_4_w_GRPKJ.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_4_w_GRPKJ[0])), _dynamic_array_synapses_4_w_GRPKJ.size()*sizeof(double));
        outfile__dynamic_array_synapses_4_w_GRPKJ.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_w_GRPKJ." << endl;
    }
    ofstream outfile__dynamic_array_synapses_5__synaptic_post;
    outfile__dynamic_array_synapses_5__synaptic_post.open("results/_dynamic_array_synapses_5__synaptic_post_9066991549005155803", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_5__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_5__synaptic_post[0])), _dynamic_array_synapses_5__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_5__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_5__synaptic_pre;
    outfile__dynamic_array_synapses_5__synaptic_pre.open("results/_dynamic_array_synapses_5__synaptic_pre_6928493121890122439", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_5__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_5__synaptic_pre[0])), _dynamic_array_synapses_5__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_5__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_5_delay;
    outfile__dynamic_array_synapses_5_delay.open("results/_dynamic_array_synapses_5_delay_7719061035252566996", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5_delay.is_open())
    {
        outfile__dynamic_array_synapses_5_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_5_delay[0])), _dynamic_array_synapses_5_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_5_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_5_delay_1;
    outfile__dynamic_array_synapses_5_delay_1.open("results/_dynamic_array_synapses_5_delay_1_760996267362132202", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5_delay_1.is_open())
    {
        outfile__dynamic_array_synapses_5_delay_1.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_5_delay_1[0])), _dynamic_array_synapses_5_delay_1.size()*sizeof(double));
        outfile__dynamic_array_synapses_5_delay_1.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5_delay_1." << endl;
    }
    _dynamic_array_synapses_5_N_incoming = dev_dynamic_array_synapses_5_N_incoming;
    ofstream outfile__dynamic_array_synapses_5_N_incoming;
    outfile__dynamic_array_synapses_5_N_incoming.open("results/_dynamic_array_synapses_5_N_incoming_-8542884590583238577", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_5_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_5_N_incoming[0])), _dynamic_array_synapses_5_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_5_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5_N_incoming." << endl;
    }
    _dynamic_array_synapses_5_N_outgoing = dev_dynamic_array_synapses_5_N_outgoing;
    ofstream outfile__dynamic_array_synapses_5_N_outgoing;
    outfile__dynamic_array_synapses_5_N_outgoing.open("results/_dynamic_array_synapses_5_N_outgoing_8541650108320020005", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_5_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_5_N_outgoing[0])), _dynamic_array_synapses_5_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_5_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5_N_outgoing." << endl;
    }
    _dynamic_array_synapses_5_w_GRBS = dev_dynamic_array_synapses_5_w_GRBS;
    ofstream outfile__dynamic_array_synapses_5_w_GRBS;
    outfile__dynamic_array_synapses_5_w_GRBS.open("results/_dynamic_array_synapses_5_w_GRBS_4034572240006583138", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5_w_GRBS.is_open())
    {
        outfile__dynamic_array_synapses_5_w_GRBS.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_5_w_GRBS[0])), _dynamic_array_synapses_5_w_GRBS.size()*sizeof(double));
        outfile__dynamic_array_synapses_5_w_GRBS.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5_w_GRBS." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6__synaptic_post;
    outfile__dynamic_array_synapses_6__synaptic_post.open("results/_dynamic_array_synapses_6__synaptic_post_8979672492784261850", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_6__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_6__synaptic_post[0])), _dynamic_array_synapses_6__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_6__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6__synaptic_pre;
    outfile__dynamic_array_synapses_6__synaptic_pre.open("results/_dynamic_array_synapses_6__synaptic_pre_2283155644187591344", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_6__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_6__synaptic_pre[0])), _dynamic_array_synapses_6__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_6__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6_delay;
    outfile__dynamic_array_synapses_6_delay.open("results/_dynamic_array_synapses_6_delay_7870373788113341035", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_delay.is_open())
    {
        outfile__dynamic_array_synapses_6_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_6_delay[0])), _dynamic_array_synapses_6_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_6_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_delay." << endl;
    }
    _dynamic_array_synapses_6_N_incoming = dev_dynamic_array_synapses_6_N_incoming;
    ofstream outfile__dynamic_array_synapses_6_N_incoming;
    outfile__dynamic_array_synapses_6_N_incoming.open("results/_dynamic_array_synapses_6_N_incoming_-942052844014408290", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_6_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_6_N_incoming[0])), _dynamic_array_synapses_6_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_6_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_N_incoming." << endl;
    }
    _dynamic_array_synapses_6_N_outgoing = dev_dynamic_array_synapses_6_N_outgoing;
    ofstream outfile__dynamic_array_synapses_6_N_outgoing;
    outfile__dynamic_array_synapses_6_N_outgoing.open("results/_dynamic_array_synapses_6_N_outgoing_9083518235633377116", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_6_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_6_N_outgoing[0])), _dynamic_array_synapses_6_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_6_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_N_outgoing." << endl;
    }
    _dynamic_array_synapses_6_w_BSPKJ = dev_dynamic_array_synapses_6_w_BSPKJ;
    ofstream outfile__dynamic_array_synapses_6_w_BSPKJ;
    outfile__dynamic_array_synapses_6_w_BSPKJ.open("results/_dynamic_array_synapses_6_w_BSPKJ_-837459483582306538", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_w_BSPKJ.is_open())
    {
        outfile__dynamic_array_synapses_6_w_BSPKJ.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_6_w_BSPKJ[0])), _dynamic_array_synapses_6_w_BSPKJ.size()*sizeof(double));
        outfile__dynamic_array_synapses_6_w_BSPKJ.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_w_BSPKJ." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_post;
    outfile__dynamic_array_synapses__synaptic_post.open("results/_dynamic_array_synapses__synaptic_post_3840486125387374025", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_post[0])), _dynamic_array_synapses__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_pre;
    outfile__dynamic_array_synapses__synaptic_pre.open("results/_dynamic_array_synapses__synaptic_pre_5162992210040840425", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_pre[0])), _dynamic_array_synapses__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_delay;
    outfile__dynamic_array_synapses_delay.open("results/_dynamic_array_synapses_delay_-1215025784993018630", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_delay.is_open())
    {
        outfile__dynamic_array_synapses_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_delay[0])), _dynamic_array_synapses_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_delay_1;
    outfile__dynamic_array_synapses_delay_1.open("results/_dynamic_array_synapses_delay_1_-8085691701396690736", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_delay_1.is_open())
    {
        outfile__dynamic_array_synapses_delay_1.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_delay_1[0])), _dynamic_array_synapses_delay_1.size()*sizeof(double));
        outfile__dynamic_array_synapses_delay_1.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_delay_1." << endl;
    }
    _dynamic_array_synapses_N_incoming = dev_dynamic_array_synapses_N_incoming;
    ofstream outfile__dynamic_array_synapses_N_incoming;
    outfile__dynamic_array_synapses_N_incoming.open("results/_dynamic_array_synapses_N_incoming_6651214916728133133", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_N_incoming[0])), _dynamic_array_synapses_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
    }
    _dynamic_array_synapses_N_outgoing = dev_dynamic_array_synapses_N_outgoing;
    ofstream outfile__dynamic_array_synapses_N_outgoing;
    outfile__dynamic_array_synapses_N_outgoing.open("results/_dynamic_array_synapses_N_outgoing_-3277140854151949897", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_N_outgoing[0])), _dynamic_array_synapses_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
    }
    _dynamic_array_synapses_w_MFGR = dev_dynamic_array_synapses_w_MFGR;
    ofstream outfile__dynamic_array_synapses_w_MFGR;
    outfile__dynamic_array_synapses_w_MFGR.open("results/_dynamic_array_synapses_w_MFGR_-4589322038306240560", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_w_MFGR.is_open())
    {
        outfile__dynamic_array_synapses_w_MFGR.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_w_MFGR[0])), _dynamic_array_synapses_w_MFGR.size()*sizeof(double));
        outfile__dynamic_array_synapses_w_MFGR.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_w_MFGR." << endl;
    }

        ofstream outfile__dynamic_array_statemonitor_1_V;
        outfile__dynamic_array_statemonitor_1_V.open("results/_dynamic_array_statemonitor_1_V_-8409868653121327144", ios::binary | ios::out);
        if(outfile__dynamic_array_statemonitor_1_V.is_open())
        {
            thrust::host_vector<double>* temp_array_dynamic_array_statemonitor_1_V = new thrust::host_vector<double>[_num__array_statemonitor_1__indices];
            for (int n=0; n<_num__array_statemonitor_1__indices; n++)
            {
                temp_array_dynamic_array_statemonitor_1_V[n] = _dynamic_array_statemonitor_1_V[n];
            }
            for(int j = 0; j < temp_array_dynamic_array_statemonitor_1_V[0].size(); j++)
            {
                for(int i = 0; i < _num__array_statemonitor_1__indices; i++)
                {
                    outfile__dynamic_array_statemonitor_1_V.write(reinterpret_cast<char*>(&temp_array_dynamic_array_statemonitor_1_V[i][j]), sizeof(double));
                }
            }
            outfile__dynamic_array_statemonitor_1_V.close();
        } else
        {
            std::cout << "Error writing output file for _dynamic_array_statemonitor_1_V." << endl;
        }
        ofstream outfile__dynamic_array_statemonitor_2_V;
        outfile__dynamic_array_statemonitor_2_V.open("results/_dynamic_array_statemonitor_2_V_-8409865652993326913", ios::binary | ios::out);
        if(outfile__dynamic_array_statemonitor_2_V.is_open())
        {
            thrust::host_vector<double>* temp_array_dynamic_array_statemonitor_2_V = new thrust::host_vector<double>[_num__array_statemonitor_2__indices];
            for (int n=0; n<_num__array_statemonitor_2__indices; n++)
            {
                temp_array_dynamic_array_statemonitor_2_V[n] = _dynamic_array_statemonitor_2_V[n];
            }
            for(int j = 0; j < temp_array_dynamic_array_statemonitor_2_V[0].size(); j++)
            {
                for(int i = 0; i < _num__array_statemonitor_2__indices; i++)
                {
                    outfile__dynamic_array_statemonitor_2_V.write(reinterpret_cast<char*>(&temp_array_dynamic_array_statemonitor_2_V[i][j]), sizeof(double));
                }
            }
            outfile__dynamic_array_statemonitor_2_V.close();
        } else
        {
            std::cout << "Error writing output file for _dynamic_array_statemonitor_2_V." << endl;
        }
        ofstream outfile__dynamic_array_statemonitor_3_V;
        outfile__dynamic_array_statemonitor_3_V.open("results/_dynamic_array_statemonitor_3_V_-8409866653121327306", ios::binary | ios::out);
        if(outfile__dynamic_array_statemonitor_3_V.is_open())
        {
            thrust::host_vector<double>* temp_array_dynamic_array_statemonitor_3_V = new thrust::host_vector<double>[_num__array_statemonitor_3__indices];
            for (int n=0; n<_num__array_statemonitor_3__indices; n++)
            {
                temp_array_dynamic_array_statemonitor_3_V[n] = _dynamic_array_statemonitor_3_V[n];
            }
            for(int j = 0; j < temp_array_dynamic_array_statemonitor_3_V[0].size(); j++)
            {
                for(int i = 0; i < _num__array_statemonitor_3__indices; i++)
                {
                    outfile__dynamic_array_statemonitor_3_V.write(reinterpret_cast<char*>(&temp_array_dynamic_array_statemonitor_3_V[i][j]), sizeof(double));
                }
            }
            outfile__dynamic_array_statemonitor_3_V.close();
        } else
        {
            std::cout << "Error writing output file for _dynamic_array_statemonitor_3_V." << endl;
        }
        ofstream outfile__dynamic_array_statemonitor_V;
        outfile__dynamic_array_statemonitor_V.open("results/_dynamic_array_statemonitor_V_6620044162385838742", ios::binary | ios::out);
        if(outfile__dynamic_array_statemonitor_V.is_open())
        {
            thrust::host_vector<double>* temp_array_dynamic_array_statemonitor_V = new thrust::host_vector<double>[_num__array_statemonitor__indices];
            for (int n=0; n<_num__array_statemonitor__indices; n++)
            {
                temp_array_dynamic_array_statemonitor_V[n] = _dynamic_array_statemonitor_V[n];
            }
            for(int j = 0; j < temp_array_dynamic_array_statemonitor_V[0].size(); j++)
            {
                for(int i = 0; i < _num__array_statemonitor__indices; i++)
                {
                    outfile__dynamic_array_statemonitor_V.write(reinterpret_cast<char*>(&temp_array_dynamic_array_statemonitor_V[i][j]), sizeof(double));
                }
            }
            outfile__dynamic_array_statemonitor_V.close();
        } else
        {
            std::cout << "Error writing output file for _dynamic_array_statemonitor_V." << endl;
        }

    // Write last run info to disk
    ofstream outfile_last_run_info;
    outfile_last_run_info.open("results/last_run_info.txt", ios::out);
    if(outfile_last_run_info.is_open())
    {
        outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
        outfile_last_run_info.close();
    } else
    {
        std::cout << "Error writing last run info to file." << std::endl;
    }
}

__global__ void synapses_post_destroy()
{
    using namespace brian;

    synapses_post.destroy();
}
__global__ void synapses_pre_destroy()
{
    using namespace brian;

    synapses_pre.destroy();
}
__global__ void synapses_1_post_destroy()
{
    using namespace brian;

    synapses_1_post.destroy();
}
__global__ void synapses_1_pre_destroy()
{
    using namespace brian;

    synapses_1_pre.destroy();
}
__global__ void synapses_2_post_destroy()
{
    using namespace brian;

    synapses_2_post.destroy();
}
__global__ void synapses_2_pre_destroy()
{
    using namespace brian;

    synapses_2_pre.destroy();
}
__global__ void synapses_3_pre_destroy()
{
    using namespace brian;

    synapses_3_pre.destroy();
}
__global__ void synapses_4_post_destroy()
{
    using namespace brian;

    synapses_4_post.destroy();
}
__global__ void synapses_4_pre_destroy()
{
    using namespace brian;

    synapses_4_pre.destroy();
}
__global__ void synapses_5_post_destroy()
{
    using namespace brian;

    synapses_5_post.destroy();
}
__global__ void synapses_5_pre_destroy()
{
    using namespace brian;

    synapses_5_pre.destroy();
}
__global__ void synapses_6_pre_destroy()
{
    using namespace brian;

    synapses_6_pre.destroy();
}

void _dealloc_arrays()
{
    using namespace brian;

    CUDA_SAFE_CALL(
            cudaFree(dev_poissongroup_1_thresholder_codeobject_rand_allocator)
            );
    CUDA_SAFE_CALL(
            cudaFree(dev_poissongroup_thresholder_codeobject_rand_allocator)
            );

    synapses_post_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_post_destroy");
    synapses_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_pre_destroy");
    synapses_1_post_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_1_post_destroy");
    synapses_1_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_1_pre_destroy");
    synapses_2_post_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_2_post_destroy");
    synapses_2_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_2_pre_destroy");
    synapses_3_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_3_pre_destroy");
    synapses_4_post_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_4_post_destroy");
    synapses_4_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_4_pre_destroy");
    synapses_5_post_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_5_post_destroy");
    synapses_5_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_5_pre_destroy");
    synapses_6_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_6_pre_destroy");

    dev_dynamic_array_ratemonitor_1_rate.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_ratemonitor_1_rate);
    _dynamic_array_ratemonitor_1_rate.clear();
    thrust::host_vector<double>().swap(_dynamic_array_ratemonitor_1_rate);
    dev_dynamic_array_ratemonitor_1_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_ratemonitor_1_t);
    _dynamic_array_ratemonitor_1_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_ratemonitor_1_t);
    dev_dynamic_array_ratemonitor_2_rate.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_ratemonitor_2_rate);
    _dynamic_array_ratemonitor_2_rate.clear();
    thrust::host_vector<double>().swap(_dynamic_array_ratemonitor_2_rate);
    dev_dynamic_array_ratemonitor_2_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_ratemonitor_2_t);
    _dynamic_array_ratemonitor_2_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_ratemonitor_2_t);
    dev_dynamic_array_ratemonitor_3_rate.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_ratemonitor_3_rate);
    _dynamic_array_ratemonitor_3_rate.clear();
    thrust::host_vector<double>().swap(_dynamic_array_ratemonitor_3_rate);
    dev_dynamic_array_ratemonitor_3_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_ratemonitor_3_t);
    _dynamic_array_ratemonitor_3_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_ratemonitor_3_t);
    dev_dynamic_array_ratemonitor_rate.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_ratemonitor_rate);
    _dynamic_array_ratemonitor_rate.clear();
    thrust::host_vector<double>().swap(_dynamic_array_ratemonitor_rate);
    dev_dynamic_array_ratemonitor_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_ratemonitor_t);
    _dynamic_array_ratemonitor_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_ratemonitor_t);
    dev_dynamic_array_spikemonitor_1_i.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikemonitor_1_i);
    _dynamic_array_spikemonitor_1_i.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikemonitor_1_i);
    dev_dynamic_array_spikemonitor_1_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_spikemonitor_1_t);
    _dynamic_array_spikemonitor_1_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_spikemonitor_1_t);
    dev_dynamic_array_spikemonitor_2_i.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikemonitor_2_i);
    _dynamic_array_spikemonitor_2_i.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikemonitor_2_i);
    dev_dynamic_array_spikemonitor_2_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_spikemonitor_2_t);
    _dynamic_array_spikemonitor_2_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_spikemonitor_2_t);
    dev_dynamic_array_spikemonitor_3_i.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikemonitor_3_i);
    _dynamic_array_spikemonitor_3_i.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikemonitor_3_i);
    dev_dynamic_array_spikemonitor_3_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_spikemonitor_3_t);
    _dynamic_array_spikemonitor_3_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_spikemonitor_3_t);
    dev_dynamic_array_spikemonitor_i.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikemonitor_i);
    _dynamic_array_spikemonitor_i.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikemonitor_i);
    dev_dynamic_array_spikemonitor_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_spikemonitor_t);
    _dynamic_array_spikemonitor_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_spikemonitor_t);
    dev_dynamic_array_statemonitor_1_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_statemonitor_1_t);
    _dynamic_array_statemonitor_1_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_statemonitor_1_t);
    dev_dynamic_array_statemonitor_2_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_statemonitor_2_t);
    _dynamic_array_statemonitor_2_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_statemonitor_2_t);
    dev_dynamic_array_statemonitor_3_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_statemonitor_3_t);
    _dynamic_array_statemonitor_3_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_statemonitor_3_t);
    dev_dynamic_array_statemonitor_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_statemonitor_t);
    _dynamic_array_statemonitor_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_statemonitor_t);
    dev_dynamic_array_synapses_1__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1__synaptic_post);
    _dynamic_array_synapses_1__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1__synaptic_post);
    dev_dynamic_array_synapses_1__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1__synaptic_pre);
    _dynamic_array_synapses_1__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1__synaptic_pre);
    dev_dynamic_array_synapses_1_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_delay);
    _dynamic_array_synapses_1_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_delay);
    dev_dynamic_array_synapses_1_delay_1.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_delay_1);
    _dynamic_array_synapses_1_delay_1.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_delay_1);
    dev_dynamic_array_synapses_1_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1_N_incoming);
    _dynamic_array_synapses_1_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1_N_incoming);
    dev_dynamic_array_synapses_1_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1_N_outgoing);
    _dynamic_array_synapses_1_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1_N_outgoing);
    dev_dynamic_array_synapses_1_w_CFPKJ.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_w_CFPKJ);
    _dynamic_array_synapses_1_w_CFPKJ.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_w_CFPKJ);
    dev_dynamic_array_synapses_2__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2__synaptic_post);
    _dynamic_array_synapses_2__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2__synaptic_post);
    dev_dynamic_array_synapses_2__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2__synaptic_pre);
    _dynamic_array_synapses_2__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2__synaptic_pre);
    dev_dynamic_array_synapses_2_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_2_delay);
    _dynamic_array_synapses_2_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_2_delay);
    dev_dynamic_array_synapses_2_delay_1.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_2_delay_1);
    _dynamic_array_synapses_2_delay_1.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_2_delay_1);
    dev_dynamic_array_synapses_2_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2_N_incoming);
    _dynamic_array_synapses_2_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2_N_incoming);
    dev_dynamic_array_synapses_2_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2_N_outgoing);
    _dynamic_array_synapses_2_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2_N_outgoing);
    dev_dynamic_array_synapses_2_w_GRGO.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_2_w_GRGO);
    _dynamic_array_synapses_2_w_GRGO.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_2_w_GRGO);
    dev_dynamic_array_synapses_3__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_3__synaptic_post);
    _dynamic_array_synapses_3__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_3__synaptic_post);
    dev_dynamic_array_synapses_3__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_3__synaptic_pre);
    _dynamic_array_synapses_3__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_3__synaptic_pre);
    dev_dynamic_array_synapses_3_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_3_delay);
    _dynamic_array_synapses_3_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_3_delay);
    dev_dynamic_array_synapses_3_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_3_N_incoming);
    _dynamic_array_synapses_3_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_3_N_incoming);
    dev_dynamic_array_synapses_3_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_3_N_outgoing);
    _dynamic_array_synapses_3_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_3_N_outgoing);
    dev_dynamic_array_synapses_3_w_GOGR.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_3_w_GOGR);
    _dynamic_array_synapses_3_w_GOGR.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_3_w_GOGR);
    dev_dynamic_array_synapses_4__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_4__synaptic_post);
    _dynamic_array_synapses_4__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_4__synaptic_post);
    dev_dynamic_array_synapses_4__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_4__synaptic_pre);
    _dynamic_array_synapses_4__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_4__synaptic_pre);
    dev_dynamic_array_synapses_4_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_4_delay);
    _dynamic_array_synapses_4_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_4_delay);
    dev_dynamic_array_synapses_4_delay_1.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_4_delay_1);
    _dynamic_array_synapses_4_delay_1.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_4_delay_1);
    dev_dynamic_array_synapses_4_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_4_N_incoming);
    _dynamic_array_synapses_4_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_4_N_incoming);
    dev_dynamic_array_synapses_4_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_4_N_outgoing);
    _dynamic_array_synapses_4_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_4_N_outgoing);
    dev_dynamic_array_synapses_4_w_GRPKJ.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_4_w_GRPKJ);
    _dynamic_array_synapses_4_w_GRPKJ.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_4_w_GRPKJ);
    dev_dynamic_array_synapses_5__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_5__synaptic_post);
    _dynamic_array_synapses_5__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_5__synaptic_post);
    dev_dynamic_array_synapses_5__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_5__synaptic_pre);
    _dynamic_array_synapses_5__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_5__synaptic_pre);
    dev_dynamic_array_synapses_5_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_5_delay);
    _dynamic_array_synapses_5_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_5_delay);
    dev_dynamic_array_synapses_5_delay_1.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_5_delay_1);
    _dynamic_array_synapses_5_delay_1.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_5_delay_1);
    dev_dynamic_array_synapses_5_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_5_N_incoming);
    _dynamic_array_synapses_5_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_5_N_incoming);
    dev_dynamic_array_synapses_5_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_5_N_outgoing);
    _dynamic_array_synapses_5_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_5_N_outgoing);
    dev_dynamic_array_synapses_5_w_GRBS.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_5_w_GRBS);
    _dynamic_array_synapses_5_w_GRBS.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_5_w_GRBS);
    dev_dynamic_array_synapses_6__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_6__synaptic_post);
    _dynamic_array_synapses_6__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_6__synaptic_post);
    dev_dynamic_array_synapses_6__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_6__synaptic_pre);
    _dynamic_array_synapses_6__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_6__synaptic_pre);
    dev_dynamic_array_synapses_6_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_6_delay);
    _dynamic_array_synapses_6_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_6_delay);
    dev_dynamic_array_synapses_6_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_6_N_incoming);
    _dynamic_array_synapses_6_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_6_N_incoming);
    dev_dynamic_array_synapses_6_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_6_N_outgoing);
    _dynamic_array_synapses_6_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_6_N_outgoing);
    dev_dynamic_array_synapses_6_w_BSPKJ.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_6_w_BSPKJ);
    _dynamic_array_synapses_6_w_BSPKJ.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_6_w_BSPKJ);
    dev_dynamic_array_synapses__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses__synaptic_post);
    _dynamic_array_synapses__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_post);
    dev_dynamic_array_synapses__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses__synaptic_pre);
    _dynamic_array_synapses__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_pre);
    dev_dynamic_array_synapses_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_delay);
    _dynamic_array_synapses_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_delay);
    dev_dynamic_array_synapses_delay_1.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_delay_1);
    _dynamic_array_synapses_delay_1.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_delay_1);
    dev_dynamic_array_synapses_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_N_incoming);
    _dynamic_array_synapses_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_N_incoming);
    dev_dynamic_array_synapses_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_N_outgoing);
    _dynamic_array_synapses_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_N_outgoing);
    dev_dynamic_array_synapses_w_MFGR.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_w_MFGR);
    _dynamic_array_synapses_w_MFGR.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_w_MFGR);

    if(_array_defaultclock_dt!=0)
    {
        delete [] _array_defaultclock_dt;
        _array_defaultclock_dt = 0;
    }
    if(dev_array_defaultclock_dt!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_dt)
                );
        dev_array_defaultclock_dt = 0;
    }
    if(_array_defaultclock_t!=0)
    {
        delete [] _array_defaultclock_t;
        _array_defaultclock_t = 0;
    }
    if(dev_array_defaultclock_t!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_t)
                );
        dev_array_defaultclock_t = 0;
    }
    if(_array_defaultclock_timestep!=0)
    {
        delete [] _array_defaultclock_timestep;
        _array_defaultclock_timestep = 0;
    }
    if(dev_array_defaultclock_timestep!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_timestep)
                );
        dev_array_defaultclock_timestep = 0;
    }
    if(_array_neurongroup_1_i!=0)
    {
        delete [] _array_neurongroup_1_i;
        _array_neurongroup_1_i = 0;
    }
    if(dev_array_neurongroup_1_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_i)
                );
        dev_array_neurongroup_1_i = 0;
    }
    if(_array_neurongroup_1_s_ahp_GO!=0)
    {
        delete [] _array_neurongroup_1_s_ahp_GO;
        _array_neurongroup_1_s_ahp_GO = 0;
    }
    if(dev_array_neurongroup_1_s_ahp_GO!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_s_ahp_GO)
                );
        dev_array_neurongroup_1_s_ahp_GO = 0;
    }
    if(_array_neurongroup_1_s_AMPA!=0)
    {
        delete [] _array_neurongroup_1_s_AMPA;
        _array_neurongroup_1_s_AMPA = 0;
    }
    if(dev_array_neurongroup_1_s_AMPA!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_s_AMPA)
                );
        dev_array_neurongroup_1_s_AMPA = 0;
    }
    if(_array_neurongroup_1_s_NMDA_1!=0)
    {
        delete [] _array_neurongroup_1_s_NMDA_1;
        _array_neurongroup_1_s_NMDA_1 = 0;
    }
    if(dev_array_neurongroup_1_s_NMDA_1!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_s_NMDA_1)
                );
        dev_array_neurongroup_1_s_NMDA_1 = 0;
    }
    if(_array_neurongroup_1_s_NMDA_2!=0)
    {
        delete [] _array_neurongroup_1_s_NMDA_2;
        _array_neurongroup_1_s_NMDA_2 = 0;
    }
    if(dev_array_neurongroup_1_s_NMDA_2!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_s_NMDA_2)
                );
        dev_array_neurongroup_1_s_NMDA_2 = 0;
    }
    if(_array_neurongroup_1_V!=0)
    {
        delete [] _array_neurongroup_1_V;
        _array_neurongroup_1_V = 0;
    }
    if(dev_array_neurongroup_1_V!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_V)
                );
        dev_array_neurongroup_1_V = 0;
    }
    if(_array_neurongroup_1_x!=0)
    {
        delete [] _array_neurongroup_1_x;
        _array_neurongroup_1_x = 0;
    }
    if(dev_array_neurongroup_1_x!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_x)
                );
        dev_array_neurongroup_1_x = 0;
    }
    if(_array_neurongroup_1_y!=0)
    {
        delete [] _array_neurongroup_1_y;
        _array_neurongroup_1_y = 0;
    }
    if(dev_array_neurongroup_1_y!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_y)
                );
        dev_array_neurongroup_1_y = 0;
    }
    if(_array_neurongroup_2_i!=0)
    {
        delete [] _array_neurongroup_2_i;
        _array_neurongroup_2_i = 0;
    }
    if(dev_array_neurongroup_2_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_i)
                );
        dev_array_neurongroup_2_i = 0;
    }
    if(_array_neurongroup_2_s_AHP_PKJ!=0)
    {
        delete [] _array_neurongroup_2_s_AHP_PKJ;
        _array_neurongroup_2_s_AHP_PKJ = 0;
    }
    if(dev_array_neurongroup_2_s_AHP_PKJ!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_s_AHP_PKJ)
                );
        dev_array_neurongroup_2_s_AHP_PKJ = 0;
    }
    if(_array_neurongroup_2_s_AMPA!=0)
    {
        delete [] _array_neurongroup_2_s_AMPA;
        _array_neurongroup_2_s_AMPA = 0;
    }
    if(dev_array_neurongroup_2_s_AMPA!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_s_AMPA)
                );
        dev_array_neurongroup_2_s_AMPA = 0;
    }
    if(_array_neurongroup_2_s_GABA!=0)
    {
        delete [] _array_neurongroup_2_s_GABA;
        _array_neurongroup_2_s_GABA = 0;
    }
    if(dev_array_neurongroup_2_s_GABA!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_s_GABA)
                );
        dev_array_neurongroup_2_s_GABA = 0;
    }
    if(_array_neurongroup_2_V!=0)
    {
        delete [] _array_neurongroup_2_V;
        _array_neurongroup_2_V = 0;
    }
    if(dev_array_neurongroup_2_V!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_V)
                );
        dev_array_neurongroup_2_V = 0;
    }
    if(_array_neurongroup_3_i!=0)
    {
        delete [] _array_neurongroup_3_i;
        _array_neurongroup_3_i = 0;
    }
    if(dev_array_neurongroup_3_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_3_i)
                );
        dev_array_neurongroup_3_i = 0;
    }
    if(_array_neurongroup_3_s_AHP_BS!=0)
    {
        delete [] _array_neurongroup_3_s_AHP_BS;
        _array_neurongroup_3_s_AHP_BS = 0;
    }
    if(dev_array_neurongroup_3_s_AHP_BS!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_3_s_AHP_BS)
                );
        dev_array_neurongroup_3_s_AHP_BS = 0;
    }
    if(_array_neurongroup_3_s_AMPA!=0)
    {
        delete [] _array_neurongroup_3_s_AMPA;
        _array_neurongroup_3_s_AMPA = 0;
    }
    if(dev_array_neurongroup_3_s_AMPA!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_3_s_AMPA)
                );
        dev_array_neurongroup_3_s_AMPA = 0;
    }
    if(_array_neurongroup_3_V!=0)
    {
        delete [] _array_neurongroup_3_V;
        _array_neurongroup_3_V = 0;
    }
    if(dev_array_neurongroup_3_V!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_3_V)
                );
        dev_array_neurongroup_3_V = 0;
    }
    if(_array_neurongroup_i!=0)
    {
        delete [] _array_neurongroup_i;
        _array_neurongroup_i = 0;
    }
    if(dev_array_neurongroup_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_i)
                );
        dev_array_neurongroup_i = 0;
    }
    if(_array_neurongroup_s_ahp_GR!=0)
    {
        delete [] _array_neurongroup_s_ahp_GR;
        _array_neurongroup_s_ahp_GR = 0;
    }
    if(dev_array_neurongroup_s_ahp_GR!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_s_ahp_GR)
                );
        dev_array_neurongroup_s_ahp_GR = 0;
    }
    if(_array_neurongroup_s_AMPA!=0)
    {
        delete [] _array_neurongroup_s_AMPA;
        _array_neurongroup_s_AMPA = 0;
    }
    if(dev_array_neurongroup_s_AMPA!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_s_AMPA)
                );
        dev_array_neurongroup_s_AMPA = 0;
    }
    if(_array_neurongroup_s_GABA_1!=0)
    {
        delete [] _array_neurongroup_s_GABA_1;
        _array_neurongroup_s_GABA_1 = 0;
    }
    if(dev_array_neurongroup_s_GABA_1!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_s_GABA_1)
                );
        dev_array_neurongroup_s_GABA_1 = 0;
    }
    if(_array_neurongroup_s_GABA_2!=0)
    {
        delete [] _array_neurongroup_s_GABA_2;
        _array_neurongroup_s_GABA_2 = 0;
    }
    if(dev_array_neurongroup_s_GABA_2!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_s_GABA_2)
                );
        dev_array_neurongroup_s_GABA_2 = 0;
    }
    if(_array_neurongroup_s_NMDA!=0)
    {
        delete [] _array_neurongroup_s_NMDA;
        _array_neurongroup_s_NMDA = 0;
    }
    if(dev_array_neurongroup_s_NMDA!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_s_NMDA)
                );
        dev_array_neurongroup_s_NMDA = 0;
    }
    if(_array_neurongroup_V!=0)
    {
        delete [] _array_neurongroup_V;
        _array_neurongroup_V = 0;
    }
    if(dev_array_neurongroup_V!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_V)
                );
        dev_array_neurongroup_V = 0;
    }
    if(_array_neurongroup_x!=0)
    {
        delete [] _array_neurongroup_x;
        _array_neurongroup_x = 0;
    }
    if(dev_array_neurongroup_x!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_x)
                );
        dev_array_neurongroup_x = 0;
    }
    if(_array_neurongroup_y!=0)
    {
        delete [] _array_neurongroup_y;
        _array_neurongroup_y = 0;
    }
    if(dev_array_neurongroup_y!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_y)
                );
        dev_array_neurongroup_y = 0;
    }
    if(_array_poissongroup_1_i!=0)
    {
        delete [] _array_poissongroup_1_i;
        _array_poissongroup_1_i = 0;
    }
    if(dev_array_poissongroup_1_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_poissongroup_1_i)
                );
        dev_array_poissongroup_1_i = 0;
    }
    if(_array_poissongroup_i!=0)
    {
        delete [] _array_poissongroup_i;
        _array_poissongroup_i = 0;
    }
    if(dev_array_poissongroup_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_poissongroup_i)
                );
        dev_array_poissongroup_i = 0;
    }
    if(_array_ratemonitor_1_N!=0)
    {
        delete [] _array_ratemonitor_1_N;
        _array_ratemonitor_1_N = 0;
    }
    if(dev_array_ratemonitor_1_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_ratemonitor_1_N)
                );
        dev_array_ratemonitor_1_N = 0;
    }
    if(_array_ratemonitor_2_N!=0)
    {
        delete [] _array_ratemonitor_2_N;
        _array_ratemonitor_2_N = 0;
    }
    if(dev_array_ratemonitor_2_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_ratemonitor_2_N)
                );
        dev_array_ratemonitor_2_N = 0;
    }
    if(_array_ratemonitor_3_N!=0)
    {
        delete [] _array_ratemonitor_3_N;
        _array_ratemonitor_3_N = 0;
    }
    if(dev_array_ratemonitor_3_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_ratemonitor_3_N)
                );
        dev_array_ratemonitor_3_N = 0;
    }
    if(_array_ratemonitor_N!=0)
    {
        delete [] _array_ratemonitor_N;
        _array_ratemonitor_N = 0;
    }
    if(dev_array_ratemonitor_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_ratemonitor_N)
                );
        dev_array_ratemonitor_N = 0;
    }
    if(_array_spikemonitor_1__source_idx!=0)
    {
        delete [] _array_spikemonitor_1__source_idx;
        _array_spikemonitor_1__source_idx = 0;
    }
    if(dev_array_spikemonitor_1__source_idx!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_1__source_idx)
                );
        dev_array_spikemonitor_1__source_idx = 0;
    }
    if(_array_spikemonitor_1_count!=0)
    {
        delete [] _array_spikemonitor_1_count;
        _array_spikemonitor_1_count = 0;
    }
    if(dev_array_spikemonitor_1_count!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_1_count)
                );
        dev_array_spikemonitor_1_count = 0;
    }
    if(_array_spikemonitor_1_N!=0)
    {
        delete [] _array_spikemonitor_1_N;
        _array_spikemonitor_1_N = 0;
    }
    if(dev_array_spikemonitor_1_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_1_N)
                );
        dev_array_spikemonitor_1_N = 0;
    }
    if(_array_spikemonitor_2__source_idx!=0)
    {
        delete [] _array_spikemonitor_2__source_idx;
        _array_spikemonitor_2__source_idx = 0;
    }
    if(dev_array_spikemonitor_2__source_idx!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_2__source_idx)
                );
        dev_array_spikemonitor_2__source_idx = 0;
    }
    if(_array_spikemonitor_2_count!=0)
    {
        delete [] _array_spikemonitor_2_count;
        _array_spikemonitor_2_count = 0;
    }
    if(dev_array_spikemonitor_2_count!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_2_count)
                );
        dev_array_spikemonitor_2_count = 0;
    }
    if(_array_spikemonitor_2_N!=0)
    {
        delete [] _array_spikemonitor_2_N;
        _array_spikemonitor_2_N = 0;
    }
    if(dev_array_spikemonitor_2_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_2_N)
                );
        dev_array_spikemonitor_2_N = 0;
    }
    if(_array_spikemonitor_3__source_idx!=0)
    {
        delete [] _array_spikemonitor_3__source_idx;
        _array_spikemonitor_3__source_idx = 0;
    }
    if(dev_array_spikemonitor_3__source_idx!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_3__source_idx)
                );
        dev_array_spikemonitor_3__source_idx = 0;
    }
    if(_array_spikemonitor_3_count!=0)
    {
        delete [] _array_spikemonitor_3_count;
        _array_spikemonitor_3_count = 0;
    }
    if(dev_array_spikemonitor_3_count!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_3_count)
                );
        dev_array_spikemonitor_3_count = 0;
    }
    if(_array_spikemonitor_3_N!=0)
    {
        delete [] _array_spikemonitor_3_N;
        _array_spikemonitor_3_N = 0;
    }
    if(dev_array_spikemonitor_3_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_3_N)
                );
        dev_array_spikemonitor_3_N = 0;
    }
    if(_array_spikemonitor__source_idx!=0)
    {
        delete [] _array_spikemonitor__source_idx;
        _array_spikemonitor__source_idx = 0;
    }
    if(dev_array_spikemonitor__source_idx!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor__source_idx)
                );
        dev_array_spikemonitor__source_idx = 0;
    }
    if(_array_spikemonitor_count!=0)
    {
        delete [] _array_spikemonitor_count;
        _array_spikemonitor_count = 0;
    }
    if(dev_array_spikemonitor_count!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_count)
                );
        dev_array_spikemonitor_count = 0;
    }
    if(_array_spikemonitor_N!=0)
    {
        delete [] _array_spikemonitor_N;
        _array_spikemonitor_N = 0;
    }
    if(dev_array_spikemonitor_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_N)
                );
        dev_array_spikemonitor_N = 0;
    }
    if(_array_statemonitor_1__indices!=0)
    {
        delete [] _array_statemonitor_1__indices;
        _array_statemonitor_1__indices = 0;
    }
    if(dev_array_statemonitor_1__indices!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor_1__indices)
                );
        dev_array_statemonitor_1__indices = 0;
    }
    if(_array_statemonitor_1_N!=0)
    {
        delete [] _array_statemonitor_1_N;
        _array_statemonitor_1_N = 0;
    }
    if(dev_array_statemonitor_1_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor_1_N)
                );
        dev_array_statemonitor_1_N = 0;
    }
    if(_array_statemonitor_1_V!=0)
    {
        delete [] _array_statemonitor_1_V;
        _array_statemonitor_1_V = 0;
    }
    if(dev_array_statemonitor_1_V!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor_1_V)
                );
        dev_array_statemonitor_1_V = 0;
    }
    if(_array_statemonitor_2__indices!=0)
    {
        delete [] _array_statemonitor_2__indices;
        _array_statemonitor_2__indices = 0;
    }
    if(dev_array_statemonitor_2__indices!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor_2__indices)
                );
        dev_array_statemonitor_2__indices = 0;
    }
    if(_array_statemonitor_2_N!=0)
    {
        delete [] _array_statemonitor_2_N;
        _array_statemonitor_2_N = 0;
    }
    if(dev_array_statemonitor_2_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor_2_N)
                );
        dev_array_statemonitor_2_N = 0;
    }
    if(_array_statemonitor_2_V!=0)
    {
        delete [] _array_statemonitor_2_V;
        _array_statemonitor_2_V = 0;
    }
    if(dev_array_statemonitor_2_V!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor_2_V)
                );
        dev_array_statemonitor_2_V = 0;
    }
    if(_array_statemonitor_3__indices!=0)
    {
        delete [] _array_statemonitor_3__indices;
        _array_statemonitor_3__indices = 0;
    }
    if(dev_array_statemonitor_3__indices!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor_3__indices)
                );
        dev_array_statemonitor_3__indices = 0;
    }
    if(_array_statemonitor_3_N!=0)
    {
        delete [] _array_statemonitor_3_N;
        _array_statemonitor_3_N = 0;
    }
    if(dev_array_statemonitor_3_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor_3_N)
                );
        dev_array_statemonitor_3_N = 0;
    }
    if(_array_statemonitor_3_V!=0)
    {
        delete [] _array_statemonitor_3_V;
        _array_statemonitor_3_V = 0;
    }
    if(dev_array_statemonitor_3_V!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor_3_V)
                );
        dev_array_statemonitor_3_V = 0;
    }
    if(_array_statemonitor__indices!=0)
    {
        delete [] _array_statemonitor__indices;
        _array_statemonitor__indices = 0;
    }
    if(dev_array_statemonitor__indices!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor__indices)
                );
        dev_array_statemonitor__indices = 0;
    }
    if(_array_statemonitor_N!=0)
    {
        delete [] _array_statemonitor_N;
        _array_statemonitor_N = 0;
    }
    if(dev_array_statemonitor_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor_N)
                );
        dev_array_statemonitor_N = 0;
    }
    if(_array_statemonitor_V!=0)
    {
        delete [] _array_statemonitor_V;
        _array_statemonitor_V = 0;
    }
    if(dev_array_statemonitor_V!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor_V)
                );
        dev_array_statemonitor_V = 0;
    }
    if(_array_synapses_1_N!=0)
    {
        delete [] _array_synapses_1_N;
        _array_synapses_1_N = 0;
    }
    if(dev_array_synapses_1_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_1_N)
                );
        dev_array_synapses_1_N = 0;
    }
    if(_array_synapses_2_N!=0)
    {
        delete [] _array_synapses_2_N;
        _array_synapses_2_N = 0;
    }
    if(dev_array_synapses_2_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_2_N)
                );
        dev_array_synapses_2_N = 0;
    }
    if(_array_synapses_3_N!=0)
    {
        delete [] _array_synapses_3_N;
        _array_synapses_3_N = 0;
    }
    if(dev_array_synapses_3_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_3_N)
                );
        dev_array_synapses_3_N = 0;
    }
    if(_array_synapses_4_N!=0)
    {
        delete [] _array_synapses_4_N;
        _array_synapses_4_N = 0;
    }
    if(dev_array_synapses_4_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_N)
                );
        dev_array_synapses_4_N = 0;
    }
    if(_array_synapses_4_sources!=0)
    {
        delete [] _array_synapses_4_sources;
        _array_synapses_4_sources = 0;
    }
    if(dev_array_synapses_4_sources!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_sources)
                );
        dev_array_synapses_4_sources = 0;
    }
    if(_array_synapses_4_sources_1!=0)
    {
        delete [] _array_synapses_4_sources_1;
        _array_synapses_4_sources_1 = 0;
    }
    if(dev_array_synapses_4_sources_1!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_sources_1)
                );
        dev_array_synapses_4_sources_1 = 0;
    }
    if(_array_synapses_4_sources_2!=0)
    {
        delete [] _array_synapses_4_sources_2;
        _array_synapses_4_sources_2 = 0;
    }
    if(dev_array_synapses_4_sources_2!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_sources_2)
                );
        dev_array_synapses_4_sources_2 = 0;
    }
    if(_array_synapses_4_sources_3!=0)
    {
        delete [] _array_synapses_4_sources_3;
        _array_synapses_4_sources_3 = 0;
    }
    if(dev_array_synapses_4_sources_3!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_sources_3)
                );
        dev_array_synapses_4_sources_3 = 0;
    }
    if(_array_synapses_4_sources_4!=0)
    {
        delete [] _array_synapses_4_sources_4;
        _array_synapses_4_sources_4 = 0;
    }
    if(dev_array_synapses_4_sources_4!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_sources_4)
                );
        dev_array_synapses_4_sources_4 = 0;
    }
    if(_array_synapses_4_sources_5!=0)
    {
        delete [] _array_synapses_4_sources_5;
        _array_synapses_4_sources_5 = 0;
    }
    if(dev_array_synapses_4_sources_5!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_sources_5)
                );
        dev_array_synapses_4_sources_5 = 0;
    }
    if(_array_synapses_4_sources_6!=0)
    {
        delete [] _array_synapses_4_sources_6;
        _array_synapses_4_sources_6 = 0;
    }
    if(dev_array_synapses_4_sources_6!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_sources_6)
                );
        dev_array_synapses_4_sources_6 = 0;
    }
    if(_array_synapses_4_sources_7!=0)
    {
        delete [] _array_synapses_4_sources_7;
        _array_synapses_4_sources_7 = 0;
    }
    if(dev_array_synapses_4_sources_7!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_sources_7)
                );
        dev_array_synapses_4_sources_7 = 0;
    }
    if(_array_synapses_4_sources_8!=0)
    {
        delete [] _array_synapses_4_sources_8;
        _array_synapses_4_sources_8 = 0;
    }
    if(dev_array_synapses_4_sources_8!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_sources_8)
                );
        dev_array_synapses_4_sources_8 = 0;
    }
    if(_array_synapses_4_sources_9!=0)
    {
        delete [] _array_synapses_4_sources_9;
        _array_synapses_4_sources_9 = 0;
    }
    if(dev_array_synapses_4_sources_9!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_sources_9)
                );
        dev_array_synapses_4_sources_9 = 0;
    }
    if(_array_synapses_4_targets!=0)
    {
        delete [] _array_synapses_4_targets;
        _array_synapses_4_targets = 0;
    }
    if(dev_array_synapses_4_targets!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_targets)
                );
        dev_array_synapses_4_targets = 0;
    }
    if(_array_synapses_4_targets_1!=0)
    {
        delete [] _array_synapses_4_targets_1;
        _array_synapses_4_targets_1 = 0;
    }
    if(dev_array_synapses_4_targets_1!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_targets_1)
                );
        dev_array_synapses_4_targets_1 = 0;
    }
    if(_array_synapses_4_targets_2!=0)
    {
        delete [] _array_synapses_4_targets_2;
        _array_synapses_4_targets_2 = 0;
    }
    if(dev_array_synapses_4_targets_2!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_targets_2)
                );
        dev_array_synapses_4_targets_2 = 0;
    }
    if(_array_synapses_4_targets_3!=0)
    {
        delete [] _array_synapses_4_targets_3;
        _array_synapses_4_targets_3 = 0;
    }
    if(dev_array_synapses_4_targets_3!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_targets_3)
                );
        dev_array_synapses_4_targets_3 = 0;
    }
    if(_array_synapses_4_targets_4!=0)
    {
        delete [] _array_synapses_4_targets_4;
        _array_synapses_4_targets_4 = 0;
    }
    if(dev_array_synapses_4_targets_4!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_targets_4)
                );
        dev_array_synapses_4_targets_4 = 0;
    }
    if(_array_synapses_4_targets_5!=0)
    {
        delete [] _array_synapses_4_targets_5;
        _array_synapses_4_targets_5 = 0;
    }
    if(dev_array_synapses_4_targets_5!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_targets_5)
                );
        dev_array_synapses_4_targets_5 = 0;
    }
    if(_array_synapses_4_targets_6!=0)
    {
        delete [] _array_synapses_4_targets_6;
        _array_synapses_4_targets_6 = 0;
    }
    if(dev_array_synapses_4_targets_6!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_targets_6)
                );
        dev_array_synapses_4_targets_6 = 0;
    }
    if(_array_synapses_4_targets_7!=0)
    {
        delete [] _array_synapses_4_targets_7;
        _array_synapses_4_targets_7 = 0;
    }
    if(dev_array_synapses_4_targets_7!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_targets_7)
                );
        dev_array_synapses_4_targets_7 = 0;
    }
    if(_array_synapses_4_targets_8!=0)
    {
        delete [] _array_synapses_4_targets_8;
        _array_synapses_4_targets_8 = 0;
    }
    if(dev_array_synapses_4_targets_8!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_targets_8)
                );
        dev_array_synapses_4_targets_8 = 0;
    }
    if(_array_synapses_4_targets_9!=0)
    {
        delete [] _array_synapses_4_targets_9;
        _array_synapses_4_targets_9 = 0;
    }
    if(dev_array_synapses_4_targets_9!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_targets_9)
                );
        dev_array_synapses_4_targets_9 = 0;
    }
    if(_array_synapses_5_N!=0)
    {
        delete [] _array_synapses_5_N;
        _array_synapses_5_N = 0;
    }
    if(dev_array_synapses_5_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_N)
                );
        dev_array_synapses_5_N = 0;
    }
    if(_array_synapses_5_sources!=0)
    {
        delete [] _array_synapses_5_sources;
        _array_synapses_5_sources = 0;
    }
    if(dev_array_synapses_5_sources!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_sources)
                );
        dev_array_synapses_5_sources = 0;
    }
    if(_array_synapses_5_sources_1!=0)
    {
        delete [] _array_synapses_5_sources_1;
        _array_synapses_5_sources_1 = 0;
    }
    if(dev_array_synapses_5_sources_1!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_sources_1)
                );
        dev_array_synapses_5_sources_1 = 0;
    }
    if(_array_synapses_5_sources_2!=0)
    {
        delete [] _array_synapses_5_sources_2;
        _array_synapses_5_sources_2 = 0;
    }
    if(dev_array_synapses_5_sources_2!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_sources_2)
                );
        dev_array_synapses_5_sources_2 = 0;
    }
    if(_array_synapses_5_sources_3!=0)
    {
        delete [] _array_synapses_5_sources_3;
        _array_synapses_5_sources_3 = 0;
    }
    if(dev_array_synapses_5_sources_3!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_sources_3)
                );
        dev_array_synapses_5_sources_3 = 0;
    }
    if(_array_synapses_5_sources_4!=0)
    {
        delete [] _array_synapses_5_sources_4;
        _array_synapses_5_sources_4 = 0;
    }
    if(dev_array_synapses_5_sources_4!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_sources_4)
                );
        dev_array_synapses_5_sources_4 = 0;
    }
    if(_array_synapses_5_sources_5!=0)
    {
        delete [] _array_synapses_5_sources_5;
        _array_synapses_5_sources_5 = 0;
    }
    if(dev_array_synapses_5_sources_5!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_sources_5)
                );
        dev_array_synapses_5_sources_5 = 0;
    }
    if(_array_synapses_5_sources_6!=0)
    {
        delete [] _array_synapses_5_sources_6;
        _array_synapses_5_sources_6 = 0;
    }
    if(dev_array_synapses_5_sources_6!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_sources_6)
                );
        dev_array_synapses_5_sources_6 = 0;
    }
    if(_array_synapses_5_sources_7!=0)
    {
        delete [] _array_synapses_5_sources_7;
        _array_synapses_5_sources_7 = 0;
    }
    if(dev_array_synapses_5_sources_7!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_sources_7)
                );
        dev_array_synapses_5_sources_7 = 0;
    }
    if(_array_synapses_5_sources_8!=0)
    {
        delete [] _array_synapses_5_sources_8;
        _array_synapses_5_sources_8 = 0;
    }
    if(dev_array_synapses_5_sources_8!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_sources_8)
                );
        dev_array_synapses_5_sources_8 = 0;
    }
    if(_array_synapses_5_sources_9!=0)
    {
        delete [] _array_synapses_5_sources_9;
        _array_synapses_5_sources_9 = 0;
    }
    if(dev_array_synapses_5_sources_9!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_sources_9)
                );
        dev_array_synapses_5_sources_9 = 0;
    }
    if(_array_synapses_5_targets!=0)
    {
        delete [] _array_synapses_5_targets;
        _array_synapses_5_targets = 0;
    }
    if(dev_array_synapses_5_targets!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_targets)
                );
        dev_array_synapses_5_targets = 0;
    }
    if(_array_synapses_5_targets_1!=0)
    {
        delete [] _array_synapses_5_targets_1;
        _array_synapses_5_targets_1 = 0;
    }
    if(dev_array_synapses_5_targets_1!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_targets_1)
                );
        dev_array_synapses_5_targets_1 = 0;
    }
    if(_array_synapses_5_targets_2!=0)
    {
        delete [] _array_synapses_5_targets_2;
        _array_synapses_5_targets_2 = 0;
    }
    if(dev_array_synapses_5_targets_2!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_targets_2)
                );
        dev_array_synapses_5_targets_2 = 0;
    }
    if(_array_synapses_5_targets_3!=0)
    {
        delete [] _array_synapses_5_targets_3;
        _array_synapses_5_targets_3 = 0;
    }
    if(dev_array_synapses_5_targets_3!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_targets_3)
                );
        dev_array_synapses_5_targets_3 = 0;
    }
    if(_array_synapses_5_targets_4!=0)
    {
        delete [] _array_synapses_5_targets_4;
        _array_synapses_5_targets_4 = 0;
    }
    if(dev_array_synapses_5_targets_4!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_targets_4)
                );
        dev_array_synapses_5_targets_4 = 0;
    }
    if(_array_synapses_5_targets_5!=0)
    {
        delete [] _array_synapses_5_targets_5;
        _array_synapses_5_targets_5 = 0;
    }
    if(dev_array_synapses_5_targets_5!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_targets_5)
                );
        dev_array_synapses_5_targets_5 = 0;
    }
    if(_array_synapses_5_targets_6!=0)
    {
        delete [] _array_synapses_5_targets_6;
        _array_synapses_5_targets_6 = 0;
    }
    if(dev_array_synapses_5_targets_6!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_targets_6)
                );
        dev_array_synapses_5_targets_6 = 0;
    }
    if(_array_synapses_5_targets_7!=0)
    {
        delete [] _array_synapses_5_targets_7;
        _array_synapses_5_targets_7 = 0;
    }
    if(dev_array_synapses_5_targets_7!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_targets_7)
                );
        dev_array_synapses_5_targets_7 = 0;
    }
    if(_array_synapses_5_targets_8!=0)
    {
        delete [] _array_synapses_5_targets_8;
        _array_synapses_5_targets_8 = 0;
    }
    if(dev_array_synapses_5_targets_8!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_targets_8)
                );
        dev_array_synapses_5_targets_8 = 0;
    }
    if(_array_synapses_5_targets_9!=0)
    {
        delete [] _array_synapses_5_targets_9;
        _array_synapses_5_targets_9 = 0;
    }
    if(dev_array_synapses_5_targets_9!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_targets_9)
                );
        dev_array_synapses_5_targets_9 = 0;
    }
    if(_array_synapses_6_N!=0)
    {
        delete [] _array_synapses_6_N;
        _array_synapses_6_N = 0;
    }
    if(dev_array_synapses_6_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_6_N)
                );
        dev_array_synapses_6_N = 0;
    }
    if(_array_synapses_N!=0)
    {
        delete [] _array_synapses_N;
        _array_synapses_N = 0;
    }
    if(dev_array_synapses_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_N)
                );
        dev_array_synapses_N = 0;
    }

    for(int i = 0; i < _num__array_statemonitor_1__indices; i++)
    {
        _dynamic_array_statemonitor_1_V[i].clear();
        thrust::device_vector<double>().swap(_dynamic_array_statemonitor_1_V[i]);
    }
    addresses_monitor__dynamic_array_statemonitor_1_V.clear();
    thrust::device_vector<double*>().swap(addresses_monitor__dynamic_array_statemonitor_1_V);
    for(int i = 0; i < _num__array_statemonitor_2__indices; i++)
    {
        _dynamic_array_statemonitor_2_V[i].clear();
        thrust::device_vector<double>().swap(_dynamic_array_statemonitor_2_V[i]);
    }
    addresses_monitor__dynamic_array_statemonitor_2_V.clear();
    thrust::device_vector<double*>().swap(addresses_monitor__dynamic_array_statemonitor_2_V);
    for(int i = 0; i < _num__array_statemonitor_3__indices; i++)
    {
        _dynamic_array_statemonitor_3_V[i].clear();
        thrust::device_vector<double>().swap(_dynamic_array_statemonitor_3_V[i]);
    }
    addresses_monitor__dynamic_array_statemonitor_3_V.clear();
    thrust::device_vector<double*>().swap(addresses_monitor__dynamic_array_statemonitor_3_V);
    for(int i = 0; i < _num__array_statemonitor__indices; i++)
    {
        _dynamic_array_statemonitor_V[i].clear();
        thrust::device_vector<double>().swap(_dynamic_array_statemonitor_V[i]);
    }
    addresses_monitor__dynamic_array_statemonitor_V.clear();
    thrust::device_vector<double*>().swap(addresses_monitor__dynamic_array_statemonitor_V);

    // static arrays
    if(_static_array__array_statemonitor_1__indices!=0)
    {
        delete [] _static_array__array_statemonitor_1__indices;
        _static_array__array_statemonitor_1__indices = 0;
    }
    if(_static_array__array_statemonitor_2__indices!=0)
    {
        delete [] _static_array__array_statemonitor_2__indices;
        _static_array__array_statemonitor_2__indices = 0;
    }
    if(_static_array__array_statemonitor_3__indices!=0)
    {
        delete [] _static_array__array_statemonitor_3__indices;
        _static_array__array_statemonitor_3__indices = 0;
    }
    if(_static_array__array_statemonitor__indices!=0)
    {
        delete [] _static_array__array_statemonitor__indices;
        _static_array__array_statemonitor__indices = 0;
    }
    if(_static_array__array_synapses_4_sources!=0)
    {
        delete [] _static_array__array_synapses_4_sources;
        _static_array__array_synapses_4_sources = 0;
    }
    if(_static_array__array_synapses_4_sources_1!=0)
    {
        delete [] _static_array__array_synapses_4_sources_1;
        _static_array__array_synapses_4_sources_1 = 0;
    }
    if(_static_array__array_synapses_4_sources_2!=0)
    {
        delete [] _static_array__array_synapses_4_sources_2;
        _static_array__array_synapses_4_sources_2 = 0;
    }
    if(_static_array__array_synapses_4_sources_3!=0)
    {
        delete [] _static_array__array_synapses_4_sources_3;
        _static_array__array_synapses_4_sources_3 = 0;
    }
    if(_static_array__array_synapses_4_sources_4!=0)
    {
        delete [] _static_array__array_synapses_4_sources_4;
        _static_array__array_synapses_4_sources_4 = 0;
    }
    if(_static_array__array_synapses_4_sources_5!=0)
    {
        delete [] _static_array__array_synapses_4_sources_5;
        _static_array__array_synapses_4_sources_5 = 0;
    }
    if(_static_array__array_synapses_4_sources_6!=0)
    {
        delete [] _static_array__array_synapses_4_sources_6;
        _static_array__array_synapses_4_sources_6 = 0;
    }
    if(_static_array__array_synapses_4_sources_7!=0)
    {
        delete [] _static_array__array_synapses_4_sources_7;
        _static_array__array_synapses_4_sources_7 = 0;
    }
    if(_static_array__array_synapses_4_sources_8!=0)
    {
        delete [] _static_array__array_synapses_4_sources_8;
        _static_array__array_synapses_4_sources_8 = 0;
    }
    if(_static_array__array_synapses_4_sources_9!=0)
    {
        delete [] _static_array__array_synapses_4_sources_9;
        _static_array__array_synapses_4_sources_9 = 0;
    }
    if(_static_array__array_synapses_4_targets!=0)
    {
        delete [] _static_array__array_synapses_4_targets;
        _static_array__array_synapses_4_targets = 0;
    }
    if(_static_array__array_synapses_4_targets_1!=0)
    {
        delete [] _static_array__array_synapses_4_targets_1;
        _static_array__array_synapses_4_targets_1 = 0;
    }
    if(_static_array__array_synapses_4_targets_2!=0)
    {
        delete [] _static_array__array_synapses_4_targets_2;
        _static_array__array_synapses_4_targets_2 = 0;
    }
    if(_static_array__array_synapses_4_targets_3!=0)
    {
        delete [] _static_array__array_synapses_4_targets_3;
        _static_array__array_synapses_4_targets_3 = 0;
    }
    if(_static_array__array_synapses_4_targets_4!=0)
    {
        delete [] _static_array__array_synapses_4_targets_4;
        _static_array__array_synapses_4_targets_4 = 0;
    }
    if(_static_array__array_synapses_4_targets_5!=0)
    {
        delete [] _static_array__array_synapses_4_targets_5;
        _static_array__array_synapses_4_targets_5 = 0;
    }
    if(_static_array__array_synapses_4_targets_6!=0)
    {
        delete [] _static_array__array_synapses_4_targets_6;
        _static_array__array_synapses_4_targets_6 = 0;
    }
    if(_static_array__array_synapses_4_targets_7!=0)
    {
        delete [] _static_array__array_synapses_4_targets_7;
        _static_array__array_synapses_4_targets_7 = 0;
    }
    if(_static_array__array_synapses_4_targets_8!=0)
    {
        delete [] _static_array__array_synapses_4_targets_8;
        _static_array__array_synapses_4_targets_8 = 0;
    }
    if(_static_array__array_synapses_4_targets_9!=0)
    {
        delete [] _static_array__array_synapses_4_targets_9;
        _static_array__array_synapses_4_targets_9 = 0;
    }
    if(_static_array__array_synapses_5_sources!=0)
    {
        delete [] _static_array__array_synapses_5_sources;
        _static_array__array_synapses_5_sources = 0;
    }
    if(_static_array__array_synapses_5_sources_1!=0)
    {
        delete [] _static_array__array_synapses_5_sources_1;
        _static_array__array_synapses_5_sources_1 = 0;
    }
    if(_static_array__array_synapses_5_sources_2!=0)
    {
        delete [] _static_array__array_synapses_5_sources_2;
        _static_array__array_synapses_5_sources_2 = 0;
    }
    if(_static_array__array_synapses_5_sources_3!=0)
    {
        delete [] _static_array__array_synapses_5_sources_3;
        _static_array__array_synapses_5_sources_3 = 0;
    }
    if(_static_array__array_synapses_5_sources_4!=0)
    {
        delete [] _static_array__array_synapses_5_sources_4;
        _static_array__array_synapses_5_sources_4 = 0;
    }
    if(_static_array__array_synapses_5_sources_5!=0)
    {
        delete [] _static_array__array_synapses_5_sources_5;
        _static_array__array_synapses_5_sources_5 = 0;
    }
    if(_static_array__array_synapses_5_sources_6!=0)
    {
        delete [] _static_array__array_synapses_5_sources_6;
        _static_array__array_synapses_5_sources_6 = 0;
    }
    if(_static_array__array_synapses_5_sources_7!=0)
    {
        delete [] _static_array__array_synapses_5_sources_7;
        _static_array__array_synapses_5_sources_7 = 0;
    }
    if(_static_array__array_synapses_5_sources_8!=0)
    {
        delete [] _static_array__array_synapses_5_sources_8;
        _static_array__array_synapses_5_sources_8 = 0;
    }
    if(_static_array__array_synapses_5_sources_9!=0)
    {
        delete [] _static_array__array_synapses_5_sources_9;
        _static_array__array_synapses_5_sources_9 = 0;
    }
    if(_static_array__array_synapses_5_targets!=0)
    {
        delete [] _static_array__array_synapses_5_targets;
        _static_array__array_synapses_5_targets = 0;
    }
    if(_static_array__array_synapses_5_targets_1!=0)
    {
        delete [] _static_array__array_synapses_5_targets_1;
        _static_array__array_synapses_5_targets_1 = 0;
    }
    if(_static_array__array_synapses_5_targets_2!=0)
    {
        delete [] _static_array__array_synapses_5_targets_2;
        _static_array__array_synapses_5_targets_2 = 0;
    }
    if(_static_array__array_synapses_5_targets_3!=0)
    {
        delete [] _static_array__array_synapses_5_targets_3;
        _static_array__array_synapses_5_targets_3 = 0;
    }
    if(_static_array__array_synapses_5_targets_4!=0)
    {
        delete [] _static_array__array_synapses_5_targets_4;
        _static_array__array_synapses_5_targets_4 = 0;
    }
    if(_static_array__array_synapses_5_targets_5!=0)
    {
        delete [] _static_array__array_synapses_5_targets_5;
        _static_array__array_synapses_5_targets_5 = 0;
    }
    if(_static_array__array_synapses_5_targets_6!=0)
    {
        delete [] _static_array__array_synapses_5_targets_6;
        _static_array__array_synapses_5_targets_6 = 0;
    }
    if(_static_array__array_synapses_5_targets_7!=0)
    {
        delete [] _static_array__array_synapses_5_targets_7;
        _static_array__array_synapses_5_targets_7 = 0;
    }
    if(_static_array__array_synapses_5_targets_8!=0)
    {
        delete [] _static_array__array_synapses_5_targets_8;
        _static_array__array_synapses_5_targets_8 = 0;
    }
    if(_static_array__array_synapses_5_targets_9!=0)
    {
        delete [] _static_array__array_synapses_5_targets_9;
        _static_array__array_synapses_5_targets_9 = 0;
    }

}

