#!/usr/bin/env python
"""
Run the ``cuba.py`` example with OpenMP threads.
"""
import matplotlib
matplotlib.use('Agg')

from brian2 import *

import brian2cuda
set_device('cuda_standalone', directory='/work/kbw29/brian2cuda/CUBA_CUDA_py',compile=True, run=True, debug=True, clean = True)

#prefs['core.default_float_dtype'] = float32

taum = 20*ms
taue = 5*ms
taui = 10*ms
Vt = -50*mV
Vr = -60*mV
El = -49*mV

eqs = '''
dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
dge/dt = -ge/taue : volt (unless refractory)
dgi/dt = -gi/taui : volt (unless refractory)
'''

P = NeuronGroup(4000, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms)
P.v = 'Vr + rand() * (Vt - Vr)'
P.ge = 0*mV
P.gi = 0*mV

we = (60*0.27/10)*mV # excitatory synaptic weight (voltage)
wi = (-20*4.5/10)*mV # inhibitory synaptic weight
Ce = Synapses(P, P, on_pre='ge += we')
Ci = Synapses(P, P, on_pre='gi += wi')
Ce.connect('i<3200', p=0.02)
Ci.connect('i>=3200', p=0.02)

s_mon = SpikeMonitor(P)

run(1 * second, report='text')

plot(s_mon.t/ms, s_mon.i, '.k', ms=1)
title('CUBA_CUDA')
xlabel('Time (ms)')
ylabel('Neuron index')
savefig('/work/kbw29/brian2cuda/CUBA_CUDA/CUBA_CUDA_rasterplot.png')
#show()
