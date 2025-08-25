#### BASAL GANGLIA MODEL ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sspspace
import scipy
from scipy.stats import beta, norm

import nengo
import nengo_dft
from nengo.config import Config
from nengo.ensemble import Ensemble
from nengo.network import Network
from nengo import Node, Connection
from nengo.neurons import LIFRate, Direct
from nengo.synapses import Lowpass, SynapseParam


class DNF(object):
    @staticmethod
    def make_kernel(self, shape, exc, inh, exc_width=5, inh_width=10, epsilon=0.001):
        assert len(shape) in [1,2]
        
        max_width = np.max(shape)
        x = np.arange(0, max_width)
        k_exc = np.exp(-0.5*((x)/exc_width)**2)
        k_inh = np.exp(-0.5*((x)/inh_width)**2)
        width = np.min([np.searchsorted(k_exc[::-1], epsilon), np.searchsorted(k_inh[::-1], epsilon)])
        k_exc = k_exc[:-width]
        k_inh = k_inh[:-width]
        
        if len(shape)==2:
            xx, yy = np.meshgrid(np.arange(len(k_exc)), np.arange(len(k_exc)))
            dist = xx**2 + yy**2
            k_exc = np.exp(-0.5*(dist/exc_width**2))
            k_inh = np.exp(-0.5*(dist/inh_width**2))            
        
        
        k_exc = np.concatenate([k_exc[1:][::-1], k_exc])
        k_inh = np.concatenate([k_inh[1:][::-1], k_inh])
        
        if len(shape)==2:
            k_exc = np.hstack([k_exc[:,1:][:,::-1], k_exc])
            k_inh = np.hstack([k_inh[:,1:][:,::-1], k_inh])
        
        k_exc = k_exc * exc / np.sum(k_exc)
        k_inh = k_inh * inh / np.sum(k_inh)
    
        k = k_exc - k_inh
        
        return k
    
    @staticmethod
    def make_dnf(self, shape, tau, c_noise, beta, global_inh, h, exc, inh, exc_w, inh_w, dt):
        net = nengo.Network()
        with net:
            N = np.prod(shape)
            net.g = nengo.Ensemble(n_neurons=N, dimensions=1,
                                gain=np.ones(N)*beta,
                                bias=np.zeros(N),
                                neuron_type=nengo.Sigmoid(tau_ref=1),
                                )
            inh_node = nengo.Node(None, size_in=1)
            nengo.Connection(net.g.neurons, inh_node, transform=-global_inh*np.ones((1, N)), synapse=tau)
            nengo.Connection(inh_node, net.g.neurons, transform=np.ones((N, 1)), synapse=None)
            h_inh = nengo.Node(h)
            nengo.Connection(h_inh, inh_node, synapse=None)

            if c_noise != 0:
                noise = nengo.Node(nengo.processes.WhiteNoise(scale=False), size_out=N)
                nengo.Connection(noise, net.g.neurons, transform=c_noise, synapse=tau)

            net.u = nengo.Node(None, size_in=N)
            nengo.Connection(net.u, net.g.neurons, synapse=tau, transform=tau/dt)
        

            k = self.make_kernel(shape, exc, inh, exc_w, inh_w)
            conv = nengo.Convolution(n_filters=1, input_shape=(N, 1), kernel_size=k.shape, strides=[1], 
                                    padding='same', init=k[...,None,None])
            nengo.Connection(net.g.neurons, net.g.neurons, synapse=tau, transform=conv)
        return net



class BasalGanglia(Network):
    """
    Basal ganglia model with a single 2D DNF that accepts an (N x ssp_dim) input.
    """
    def __init__(self, n_actions, dnf_parameters,
                 encoders, d1_weight=1.0, d2_weight=1.0,
                 neuron_type=LIFRate(), seed=None, dnf_neurons = 400):
        super().__init__(seed=seed)
        self.n_actions = n_actions
        self.ssp_dim   = 512
        self.encoders  = encoders
        self.d1_weight = d1_weight
        self.d2_weight = d2_weight
        self.total_dim = self.n_actions * self.ssp_dim
        self.dnf_neurons = dnf_neurons

        # Synapse constants
        self.gaba = None
        self.ampa = 0.0

        cfg = Config(Ensemble)
        cfg[Ensemble].neuron_type = neuron_type

        with self, cfg:
            # Cortical input nodes - scaled bundles
            self.cortex_inputs = []
            for i in range(self.n_actions):
                cortical_node = Node(label=f'cortex_in_{i}', size_in = self.ssp_dim)
                self.cortex_inputs.append(cortical_node)
            
            # Dopamine
            self.dopamine = Node(label='dopamine', size_in=1)

            # Concentration layer
            self.concentration_layer = Ensemble(n_neurons=1000 * self.n_actions, 
                                                dimensions=self.total_dim + 1, 
                                                neuron_type = Direct(), 
                                                label=f'concentration_layer_{i}')
            

            # D1 DNF 
            self.d1_dnf = DNF.make_dnf(**dnf_parameters)

            # D2 DNF
            self.d2_dnf = DNF.make_dnf(**dnf_parameters)

            # -----Downstream: per-action STN, GPe, GPi -------------
            self.stn = []
            self.gpe = []
            self.gpi = []
            for i in range(self.n_actions):
                stn_i = Ensemble(n_neurons = 1000, dimensions = self.ssp_dim, label = f'STN_{i}')
                gpe_i = Ensemble(n_neurons = 1000, dimensions = self.ssp_dim, label = f'GPe_{i}')
                gpi_i = Ensemble(n_neurons = 1000, dimensions = self.ssp_dim, label = f'GPi_{i}')
                self.stn.append(stn_i)
                self.gpe.append(gpe_i)
                self.gpi.append(gpi_i)
            
            # Output node
            self.bg_out = Node(size_in= self.total_dim, label=f'Output Node')

            # ------- CONNECTIONS ----------------
            # Connect dopamine to concentration layer
            Connection(self.dopamine, self.concentration_layer[-1], synapse=None)
            # Connect cortical ndoes to concentration layer 
            for i in range(self.n_actions):
                Connection(self.cortex_inputs[i], self.concentration_layer[i * self.ssp_dim: (i + 1) * self.ssp_dim], synapse=None)

                def _d1_fn(x, d1_weight=self.d1_weight, dim = self.ssp_dim, index=i):
                    dopamine = x[-1]
                    segment = x[index * dim : (index+ 1) * dim]
                    # print('d1', index * dim, (index + 1) * dim)
                    return (segment * (d1_weight + dopamine))
                
                def _d2_fn(x, d2_weight=self.d2_weight, dim = self.ssp_dim, index=i):
                    dopamine = x[-1]
                    segment = x[index * dim : (index + 1) * dim]
                    # print('d2', index * dim, (index + 1) * dim)
                    # print(i * self.ssp_dim , (i + 1) * self.ssp_dim)
                    return (segment * (d2_weight - dopamine))
                
                # Concentration layer >> D1
                # print('DNF', i * self.dnf_neurons, (i+1) * self.dnf_neurons)
                Connection(self.concentration_layer, self.d1_dnf.g.neurons[i * self.dnf_neurons: (i+1) * self.dnf_neurons], 
                            function=_d1_fn, transform=self.encoders, synapse=None)

                # Concentration layer >> D2
                Connection(self.concentration_layer, self.d2_dnf.g.neurons[i * self.dnf_neurons: (i+1) * self.dnf_neurons], 
                            function=_d2_fn, transform=self.encoders, synapse = None)
                
                # print('--------------')

                # Cortex --> STN 
                Connection(self.cortex_inputs[i], self.stn[i], synapse=None)
                # STN --> GPi
                Connection(self.stn[i], self.gpi[i], transform=1.0, synapse=self.ampa)
                # STN --> GPe
                Connection(self.stn[i], self.gpe[i], transform=1.0, synapse=self.ampa)
                # GPe --> STN 
                Connection(self.gpe[i], self.stn[i], transform=-1.0, synapse=self.gaba)
                # D1 --> GPi
                Connection(self.d1_dnf.g.neurons[i * self.dnf_neurons: (i+1) * self.dnf_neurons], self.gpi[i], transform=-1.0 * self.encoders.T, synapse=self.gaba)
                # D2 --> GPe
                Connection(self.d2_dnf.g.neurons[i * self.dnf_neurons: (i+1) * self.dnf_neurons], self.gpe[i], transform=-1.0 * self.encoders.T, synapse=self.gaba)
                # GPe --> GPi
                Connection(self.gpe[i], self.gpi[i], transform=-1.0, synapse = self.gaba)
                # GPi --> Output
                Connection(self.gpi[i], self.bg_out[i * self.ssp_dim : (i + 1) * self.ssp_dim], transform=-3.0, synapse=None)
                # print(i)