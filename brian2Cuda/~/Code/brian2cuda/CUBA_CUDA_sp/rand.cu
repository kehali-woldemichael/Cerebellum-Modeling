
#include "objects.h"
#include "rand.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/cuda_utils.h"
#include "network.h"
#include <curand.h>
#include <ctime>
#include <curand_kernel.h>

// XXX: for some documentation on random number generation, check out our wiki:
//      https://github.com/brian-team/brian2cuda/wiki/Random-number-generation

using namespace brian;

// TODO make this a class member function
// TODO don't call one kernel per codeobject but instead on kernel which takes
//      care of all codeobjects, preferably called with as many threads/blocks
//      as necessary for all states and initializing in parallel with warp
//      level divergence [needs changing set_curand_device_api_states()]
namespace {

    __global__ void init_curand_states(int N, int sequence_offset)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < N)
        {
            // Each thread gets the same seed, a different sequence number and
            // no offset
            // TODO: different seed and 0 sequence number is much faster, with
            // less security for independent sequences, add option as
            // preference!
            //curand_init(curand_seed + idx, 0, 0,
            curand_init(
                    *d_curand_seed,          // seed
                    sequence_offset + idx,   // sequence number
                    0,                       // offset
                    &d_curand_states[idx]);
        }
    }
}


// need a function pointer for Network::add(), can't pass a pointer to a class
// method, which is of different type
void _run_random_number_buffer()
{
    // random_number_buffer is a RandomNumberBuffer instance, declared in objects.cu
    random_number_buffer.next_time_step();
}


void RandomNumberBuffer::init()
{
    // check that we have enough memory available
    size_t free_byte;
    size_t total_byte;
    CUDA_SAFE_CALL(
            cudaMemGetInfo(&free_byte, &total_byte)
            );
    size_t num_free_floats = free_byte / sizeof(randomNumber_t);

    if (run_counter == 0)
    {

        // number of time steps each codeobject is executed during current Network::run() call
        // XXX: we are assuming here that this function is only run in the first time step of a Network::run()
        int64_t num_steps_this_run_poissongroup_1_thresholder_codeobject = defaultclock.i_end - *(defaultclock.timestep);
        int64_t num_steps_this_run_poissongroup_thresholder_codeobject = defaultclock.i_end - *(defaultclock.timestep);

        // Get the number of needed random numbers per clock cycle, the generation interval, and the number generated per curand call.
        num_per_cycle_rand_poissongroup_1_thresholder_codeobject = 10 * 1;
        rand_floats_per_obj_poissongroup_1_thresholder_codeobject = floats_per_obj;
        if (floats_per_obj < num_per_cycle_rand_poissongroup_1_thresholder_codeobject)
            rand_floats_per_obj_poissongroup_1_thresholder_codeobject = num_per_cycle_rand_poissongroup_1_thresholder_codeobject;
        rand_interval_poissongroup_1_thresholder_codeobject = (int)(rand_floats_per_obj_poissongroup_1_thresholder_codeobject / num_per_cycle_rand_poissongroup_1_thresholder_codeobject);
        num_per_gen_rand_poissongroup_1_thresholder_codeobject = num_per_cycle_rand_poissongroup_1_thresholder_codeobject * rand_interval_poissongroup_1_thresholder_codeobject;
        idx_rand_poissongroup_1_thresholder_codeobject = rand_interval_poissongroup_1_thresholder_codeobject;

        // create max as many random numbers as will be needed during the current Network.run() call
        if ((int64_t)rand_interval_poissongroup_1_thresholder_codeobject > num_steps_this_run_poissongroup_1_thresholder_codeobject)
        {
            // NOTE: if the conditional is true, we can savely cast num_steps_this_run_poissongroup_1_thresholder_codeobject to int
            num_per_gen_rand_poissongroup_1_thresholder_codeobject = num_per_cycle_rand_poissongroup_1_thresholder_codeobject * (int)num_steps_this_run_poissongroup_1_thresholder_codeobject;
            assert((int64_t)num_per_cycle_rand_poissongroup_1_thresholder_codeobject * num_steps_this_run_poissongroup_1_thresholder_codeobject == num_per_gen_rand_poissongroup_1_thresholder_codeobject);
            rand_interval_poissongroup_1_thresholder_codeobject = (int)num_steps_this_run_poissongroup_1_thresholder_codeobject;
            // set this for buffer to be refilled at first next_time_step() call
            idx_rand_poissongroup_1_thresholder_codeobject = rand_interval_poissongroup_1_thresholder_codeobject;
        }

        // curandGenerateNormal requires an even number for pseudorandom generators
        if (num_per_gen_rand_poissongroup_1_thresholder_codeobject % 2 != 0)
        {
            num_per_gen_rand_poissongroup_1_thresholder_codeobject = num_per_gen_rand_poissongroup_1_thresholder_codeobject + 1;
        }

        // make sure that we don't use more memory then available
        // this checks per codeobject the number of generated floats against total available floats
        while (num_free_floats < num_per_gen_rand_poissongroup_1_thresholder_codeobject)
        {
            printf("INFO not enough memory available to generate %i random numbers for poissongroup_1_thresholder_codeobject, reducing the buffer size\n", num_free_floats);
            if (num_per_gen_rand_poissongroup_1_thresholder_codeobject < num_per_cycle_rand_poissongroup_1_thresholder_codeobject)
            {
                if (num_free_floats < num_per_cycle_rand_poissongroup_1_thresholder_codeobject)
                {
                    printf("ERROR not enough memory to generate random numbers for poissongroup_1_thresholder_codeobject %s:%d\n", __FILE__, __LINE__);
                    _dealloc_arrays();
                    exit(1);
                }
                else
                {
                    num_per_gen_rand_poissongroup_1_thresholder_codeobject = num_per_cycle_rand_poissongroup_1_thresholder_codeobject;
                    break;
                }
            }
            num_per_gen_rand_poissongroup_1_thresholder_codeobject /= 2;
        }
        printf("INFO generating %i rand every %i clock cycles for poissongroup_1_thresholder_codeobject\n", num_per_gen_rand_poissongroup_1_thresholder_codeobject, rand_interval_poissongroup_1_thresholder_codeobject);

        CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_poissongroup_1_thresholder_codeobject_rand_allocator, sizeof(randomNumber_t)*num_per_gen_rand_poissongroup_1_thresholder_codeobject)
                );
        // Get the number of needed random numbers per clock cycle, the generation interval, and the number generated per curand call.
        num_per_cycle_rand_poissongroup_thresholder_codeobject = 100 * 1;
        rand_floats_per_obj_poissongroup_thresholder_codeobject = floats_per_obj;
        if (floats_per_obj < num_per_cycle_rand_poissongroup_thresholder_codeobject)
            rand_floats_per_obj_poissongroup_thresholder_codeobject = num_per_cycle_rand_poissongroup_thresholder_codeobject;
        rand_interval_poissongroup_thresholder_codeobject = (int)(rand_floats_per_obj_poissongroup_thresholder_codeobject / num_per_cycle_rand_poissongroup_thresholder_codeobject);
        num_per_gen_rand_poissongroup_thresholder_codeobject = num_per_cycle_rand_poissongroup_thresholder_codeobject * rand_interval_poissongroup_thresholder_codeobject;
        idx_rand_poissongroup_thresholder_codeobject = rand_interval_poissongroup_thresholder_codeobject;

        // create max as many random numbers as will be needed during the current Network.run() call
        if ((int64_t)rand_interval_poissongroup_thresholder_codeobject > num_steps_this_run_poissongroup_thresholder_codeobject)
        {
            // NOTE: if the conditional is true, we can savely cast num_steps_this_run_poissongroup_thresholder_codeobject to int
            num_per_gen_rand_poissongroup_thresholder_codeobject = num_per_cycle_rand_poissongroup_thresholder_codeobject * (int)num_steps_this_run_poissongroup_thresholder_codeobject;
            assert((int64_t)num_per_cycle_rand_poissongroup_thresholder_codeobject * num_steps_this_run_poissongroup_thresholder_codeobject == num_per_gen_rand_poissongroup_thresholder_codeobject);
            rand_interval_poissongroup_thresholder_codeobject = (int)num_steps_this_run_poissongroup_thresholder_codeobject;
            // set this for buffer to be refilled at first next_time_step() call
            idx_rand_poissongroup_thresholder_codeobject = rand_interval_poissongroup_thresholder_codeobject;
        }

        // curandGenerateNormal requires an even number for pseudorandom generators
        if (num_per_gen_rand_poissongroup_thresholder_codeobject % 2 != 0)
        {
            num_per_gen_rand_poissongroup_thresholder_codeobject = num_per_gen_rand_poissongroup_thresholder_codeobject + 1;
        }

        // make sure that we don't use more memory then available
        // this checks per codeobject the number of generated floats against total available floats
        while (num_free_floats < num_per_gen_rand_poissongroup_thresholder_codeobject)
        {
            printf("INFO not enough memory available to generate %i random numbers for poissongroup_thresholder_codeobject, reducing the buffer size\n", num_free_floats);
            if (num_per_gen_rand_poissongroup_thresholder_codeobject < num_per_cycle_rand_poissongroup_thresholder_codeobject)
            {
                if (num_free_floats < num_per_cycle_rand_poissongroup_thresholder_codeobject)
                {
                    printf("ERROR not enough memory to generate random numbers for poissongroup_thresholder_codeobject %s:%d\n", __FILE__, __LINE__);
                    _dealloc_arrays();
                    exit(1);
                }
                else
                {
                    num_per_gen_rand_poissongroup_thresholder_codeobject = num_per_cycle_rand_poissongroup_thresholder_codeobject;
                    break;
                }
            }
            num_per_gen_rand_poissongroup_thresholder_codeobject /= 2;
        }
        printf("INFO generating %i rand every %i clock cycles for poissongroup_thresholder_codeobject\n", num_per_gen_rand_poissongroup_thresholder_codeobject, rand_interval_poissongroup_thresholder_codeobject);

        CUDA_SAFE_CALL(
                cudaMalloc((void**)&dev_poissongroup_thresholder_codeobject_rand_allocator, sizeof(randomNumber_t)*num_per_gen_rand_poissongroup_thresholder_codeobject)
                );



        // now check if the total number of generated floats fit into available memory
        int total_num_generated_floats = 0;
        total_num_generated_floats += num_per_gen_rand_poissongroup_1_thresholder_codeobject;
        total_num_generated_floats += num_per_gen_rand_poissongroup_thresholder_codeobject;
        if (num_free_floats < total_num_generated_floats)
        {
            // TODO: find a way to deal with this? E.g. looping over buffers sorted
            // by buffer size and reducing them until it fits.
            printf("MEMORY ERROR: Trying to generate more random numbers than fit "
                   "into available memory. Please report this as an issue on "
                   "GitHub: https://github.com/brian-team/brian2cuda/issues/new");
            _dealloc_arrays();
            exit(1);
        }

    } // if (run_counter == 0)

    // init curand states only in first run
    if (run_counter == 0)
    {

        // Update curand device api states once before anything is run. At this
        // point all N's (also from probabilistically generated synapses) are
        // known. This might update the number of needed curand states.
        ensure_enough_curand_states();
    }

}


void RandomNumberBuffer::allocate_device_curand_states()
{
    // allocate globabl memory for curand device api states
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_curand_states,
                sizeof(curandState) * num_curand_states)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_curand_states,
                &dev_curand_states, sizeof(curandState*))
            );
}



void RandomNumberBuffer::update_needed_number_curand_states()
{
    // Find the maximum number of threads generating random numbers in parallel
    // using the cuRAND device API. For synapses objects, the number of
    // synapses might not be known yet. This is the case when the first random
    // seed is set and for any seed() call before the synapses creation.
    num_threads_curand_init = max_threads_per_block;
    num_blocks_curand_init = num_curand_states / max_threads_per_block + 1;
    if (num_curand_states < num_threads_curand_init)
        num_threads_curand_init = num_curand_states;
}


void RandomNumberBuffer::set_curand_device_api_states(bool reset_seed)
{
    int sequence_offset = 0;
    int num_curand_states_old = num_curand_states;
    // Whenever curand states are set, check if enough states where
    // initialized. This will generate states the first time the seed is set.
    // But it can be that the seed is set before all network objects' N are
    // available (e.g. synapses not created yet) and before the network is
    // run. In such a case, once the network is run, missing curand states are
    // generated here. If the seed was not reset inbetween, the pervious states
    // should not be reinitialized (achieved by the `sequence_offset`
    // parameter). If the seed was reset, then all states should be
    // reinitialized.
    update_needed_number_curand_states();

    // number of curand states that need to be initialized
    int num_curand_states_to_init;

    if (reset_seed)
    {
        // initialize all curand states
        num_curand_states_to_init = num_curand_states;
        sequence_offset = 0;
    }
    else
    {
        // don't initialize existing curand states, only the new ones
        num_curand_states_to_init = num_curand_states - num_curand_states_old;
        sequence_offset = num_curand_states_old;
    }

    if (num_curand_states_old < num_curand_states)
    {
        // copy curand states to new array of updated size
        curandState* dev_curand_states_old = dev_curand_states;
        // allocate memory for new number of curand states
        allocate_device_curand_states();

        if ((!reset_seed) && (num_curand_states_old > 0))
        {
            // copy old states to new memory address on device
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_curand_states, dev_curand_states_old,
                        sizeof(curandState) * num_curand_states_old,
                        cudaMemcpyDeviceToDevice)
                    );
        }
    }

    if (num_curand_states_to_init > 0)
    {
        init_curand_states<<<num_blocks_curand_init, num_threads_curand_init>>>(
                num_curand_states_to_init,
                sequence_offset);
    }
}


void RandomNumberBuffer::ensure_enough_curand_states()
{
    // Separate public function needed for synapses codeobjects that are run
    // only once before the network
    // The N of synapses will not be known when setting the seed and needs to
    // be updated before using random numbers per synapse. This occurs e.g.
    // when initializing synaptic variables (synapses_group_conditional_....)
    bool reset_seed = false;
    set_curand_device_api_states(reset_seed);
}


void RandomNumberBuffer::run_finished()
{
    needs_init = true;
    run_counter += 1;
}


void RandomNumberBuffer::set_seed(unsigned long long seed)
{
    CUDA_SAFE_CALL(
            curandSetPseudoRandomGeneratorSeed(curand_generator, seed)
            );

    // generator offset needs to be reset to its default (=0)
    CUDA_SAFE_CALL(
            curandSetGeneratorOffset(curand_generator, 0ULL)
            );

    // set seed for curand device api calls
    // don't set the same seed for host api and device api random states, just in case
    unsigned long long curand_seed = seed + 1;
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_curand_seed, &curand_seed,
                sizeof(unsigned long long), cudaMemcpyHostToDevice)
            );

    bool reset_seed = true;
    set_curand_device_api_states(reset_seed);
    // We set all device api states for codeobjects run outside the network
    // since we don't know when they will be used.
    //set_curand_device_api_states_for_separate_calls();
    // Curand device api states for binomials during network runs will be set
    // only for the current run in init(), once the network starts.
}


void RandomNumberBuffer::refill_uniform_numbers(
        randomNumber_t* dev_rand_allocator,
        randomNumber_t* &dev_rand,
        int num_per_gen_rand,
        int &idx_rand)
{
    // generate uniform distributed random numbers and reset buffer index

    curandGenerateUniformDouble(curand_generator, dev_rand_allocator, num_per_gen_rand);
    // before: XXX dev_rand = &dev_rand_allocator[0];
    dev_rand = dev_rand_allocator;
    idx_rand = 1;
}


void RandomNumberBuffer::refill_normal_numbers(
        randomNumber_t* dev_randn_allocator,
        randomNumber_t* &dev_randn,
        int num_per_gen_randn,
        int &idx_randn)
{
    // generate normal distributed random numbers and reset buffer index

    curandGenerateNormalDouble(curand_generator, dev_randn_allocator, num_per_gen_randn, 0, 1);
    // before: XXX dev_randn = &dev_randn_allocator[0];
    dev_randn = dev_randn_allocator;
    idx_randn = 1;
}


void RandomNumberBuffer::next_time_step()
{
    // init buffers at fist time step of each run call
    if (needs_init)
    {
        // free device memory for random numbers used during last run call
        if (run_counter > 0)
        {
            if (run_counter == 0)
            {

                CUDA_SAFE_CALL(
                        cudaFree(dev_poissongroup_1_thresholder_codeobject_rand_allocator)
                        );
                CUDA_SAFE_CALL(
                        cudaFree(dev_poissongroup_thresholder_codeobject_rand_allocator)
                        );

            } // run_counter == 0
        }

        // init random number buffers
        init();
        needs_init = false;
    }

    if (run_counter == 0)
    {

        // uniform numbers for poissongroup_1_thresholder_codeobject
        if (idx_rand_poissongroup_1_thresholder_codeobject == rand_interval_poissongroup_1_thresholder_codeobject)
        {
            refill_uniform_numbers(
                    dev_poissongroup_1_thresholder_codeobject_rand_allocator,
                    dev_poissongroup_1_thresholder_codeobject_rand,
                    num_per_gen_rand_poissongroup_1_thresholder_codeobject,
                    idx_rand_poissongroup_1_thresholder_codeobject);
        }
        else
        {
            // move device pointer to next numbers
            dev_poissongroup_1_thresholder_codeobject_rand += num_per_cycle_rand_poissongroup_1_thresholder_codeobject;
            idx_rand_poissongroup_1_thresholder_codeobject += 1;
        }
        assert(dev_poissongroup_1_thresholder_codeobject_rand < dev_poissongroup_1_thresholder_codeobject_rand_allocator + num_per_gen_rand_poissongroup_1_thresholder_codeobject);
        // uniform numbers for poissongroup_thresholder_codeobject
        if (idx_rand_poissongroup_thresholder_codeobject == rand_interval_poissongroup_thresholder_codeobject)
        {
            refill_uniform_numbers(
                    dev_poissongroup_thresholder_codeobject_rand_allocator,
                    dev_poissongroup_thresholder_codeobject_rand,
                    num_per_gen_rand_poissongroup_thresholder_codeobject,
                    idx_rand_poissongroup_thresholder_codeobject);
        }
        else
        {
            // move device pointer to next numbers
            dev_poissongroup_thresholder_codeobject_rand += num_per_cycle_rand_poissongroup_thresholder_codeobject;
            idx_rand_poissongroup_thresholder_codeobject += 1;
        }
        assert(dev_poissongroup_thresholder_codeobject_rand < dev_poissongroup_thresholder_codeobject_rand_allocator + num_per_gen_rand_poissongroup_thresholder_codeobject);

    }// run_counter == 0
}
