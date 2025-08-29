#### BASAL GANGLIA MODEL ####
### WHERE WE NICKED STUFF

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
from sspspace.sspspace.encoders import RandomSSPSpace
from utility import Utility


class DNF: 
    """
    Dynamic Neural Field (DNF) helper class.

    Provides methods to construct neural field kernels and corresponding Nengo
    networks for simulating localized excitatory and inhibitory dynamics.

    References/Citations: TODO
    """
    @staticmethod
    def make_kernel(shape, exc, inh, exc_width=5, inh_width=10, epsilon=0.001):
        """
        Construct a 1D or 2D lateral interaction kernel for a dynamic neural field.
        Parameters
        ----------
            shape : tuple or list
                Shape of the DNF (1D or 2D). Determines whether a line or grid kernel is built.
            exc : float
                Total excitatory scaling factor.
            inh : float
                Total inhibitory scaling factor.
            exc_width : float, optional (default=5)
                Spread (σ) of the Gaussian excitation profile.
            inh_width : float, optional (default=10)
                Spread (σ) of the Gaussian inhibition profile.
            epsilon : float, optional (default=0.001)
                Cutoff threshold for kernel tails.
        Returns
        -------
            k : ndarray
                Final excitatory-inhibitory difference kernel for the given shape.
        """
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
    def make_dnf(shape, tau, c_noise, beta, global_inh, h, exc, inh, exc_w, inh_w, dt):
        """
        Build a Nengo network implementing a Dynamic Neural Field (DNF).
        Parameters
        ----------
            shape : tuple
                Dimensions of the neural field (e.g., (100,) for 1D or (32, 32) for 2D).
            tau : float
                Synaptic time constant for recurrent dynamics.
            c_noise : float
                Amplitude of background noise.
            beta : float
                Gain parameter for neural activation.
            global_inh : float
                Global inhibitory weight.
            h : float
                Resting potential / bias term.
            exc : float
                Excitatory scaling factor.
            inh : float
                Inhibitory scaling factor.
            exc_w : float
                Excitatory kernel width.
            inh_w : float
                Inhibitory kernel width.
            dt : float
                Simulation time step.
        Returns
        -------
            net : nengo.Network
                A Nengo network implementing the specified DNF with recurrent dynamics,
                lateral interactions, and global inhibition.
        """
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
        

            k = DNF.make_kernel(shape, exc, inh, exc_w, inh_w)
            conv = nengo.Convolution(n_filters=1, input_shape=(N, 1), kernel_size=k.shape, strides=[1], 
                                    padding='same', init=k[...,None,None])
            nengo.Connection(net.g.neurons, net.g.neurons, synapse=tau, transform=conv)
        return net


# Basal Ganglia model
class BasalGanglia(Network):
    """
    Biologically inspired Basal Ganglia (BG) model implemented as a Nengo network.

    This model integrates cortical input SSPs, dopaminergic modulation, and
    dynamic neural fields (DNFs) to simulate action selection and specification.

    Features
    --------
    - Accepts multiple actions (n_actions), each represented as a high-dimensional SSP (512D).
    - Uses D1 and D2 striatal DNFs for parallel processing of action salience.
    - Includes subcortical structures (STN, GPe, GPi) for BG loops.
    - Supports dopaminergic input for reinforcement learning tasks.

    Parameters
    ----------
        n_actions : int
            Number of discrete action channels (each with its own SSP input).
        dnf_parameters : dict
            Dictionary of parameters for constructing the DNF networks
            (passed to `DNF.make_dnf`).
        encoders : np.ndarray or list
            Encoders for representing cortical input SSPs.
        d1_weight : float, optional (default=1.0)
            Scaling factor for D1 striatal pathway.
        d2_weight : float, optional (default=1.0)
            Scaling factor for D2 striatal pathway.
        neuron_type : nengo.NeuronType, optional (default=LIFRate())
            Neuron model for Nengo ensembles.
        seed : int, optional
            Random seed for reproducibility.
        dnf_neurons : int, optional (default=400)
            Number of neurons per DNF ensemble.

    Attributes
    ----------
        cortex_inputs : list[Node]
            List of cortical input nodes, one per action channel.
        dopamine : Node
            Node representing dopaminergic modulation.
        concentration_layer : Ensemble
            Layer aggregating action representations before striatal DNFs.
        d1_dnf, d2_dnf : nengo.Network
            Dynamic neural fields for D1 and D2 pathways.
        stn, gpe, gpi : list[Ensemble]
            Subthalamic nucleus (STN), external globus pallidus (GPe),
            and internal globus pallidus (GPi) ensembles per action channel.
    """

    def __init__(self, n_actions: int, dnf_parameters: dict,
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
        self.dnf_parameters = dnf_parameters
        # Storage for deferred probe requests
        self._probe_requests = [] # list of dicts describing requested probes

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
            self.d1_dnf = DNF.make_dnf(**self.dnf_parameters)

            # D2 DNF
            self.d2_dnf = DNF.make_dnf(**self.dnf_parameters)

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
                # D1 --> GPi: connecting the domain.shape[0] neurons activities for each action channel as input to the GPi ensemble
                Connection(self.d1_dnf.g.neurons[i * self.dnf_neurons: (i+1) * self.dnf_neurons], self.gpi[i], transform=-1.0 * self.encoders.T, synapse=self.gaba)
                # D2 --> GPe: connecting the domain.shape[0] neurons activities for each action channel as input to the GPi ensemble
                Connection(self.d2_dnf.g.neurons[i * self.dnf_neurons: (i+1) * self.dnf_neurons], self.gpe[i], transform=-1.0 * self.encoders.T, synapse=self.gaba)
                # GPe --> GPi
                Connection(self.gpe[i], self.gpi[i], transform=-1.0, synapse = self.gaba)
                # GPi --> Output
                Connection(self.gpi[i], self.bg_out[i * self.ssp_dim : (i + 1) * self.ssp_dim], transform=-3.0, synapse=None)
                # print(i)

    def add_probe(self, part: str, index: int|None = None, what: str = "output", field: str | None = None, name: str | None = None):
        """
        Queue a probe to be created when simulate() is called, 

        Parameters
        ----------
        part: str
            What to probe. Accepted (case-insensitive) values:
            - Per-action lists: 'cortex_inputs', 'stn', 'gpe', 'gpi'
            - Whole objects: 'concentration_layer', 'dopamine', 'bg_out'
            - DNFs (neurons): 'd1_dnf', 'd2_dnf' (supports neuron fields)
        index: int | None, optional 
            For per-action lists, choose which channel (0 .... n_actions -1)
            If None, all channels will be probed (one probe per channel)
            Ignored for whole objects and DNFs unless you want per-action DNF slice. 
        what: {'input', 'output', 'neurons'}, optional
            'input' probes what is being encoded into Ensemble/None
            'output' probes decoded output of an Ensemble/Node
            'neurons' probes the underlying neuron signals of the Ensemble. 
            For DNFs you likely want 'neurons'
        field: {'input', 'spikes', 'voltage'}| None, optional
            Only used when what='neurons'. Selects the neuron signal to probe. 
            - None -> default (firing rates for LIFRate, spikes for LIF)
            - 'input' -> input current to neurons
            - 'spikes' -> spiking trains (for spiking neuron types)
            - 'voltage' -> membrane voltage (where supported)
        name: str | None, optional
            Optional key name for returned data. If ommitted, 
            a sensibl e default is generated, eg, 'gpi_0_output' 

        Notes:
        ------
        • This does not create the Nengo.Probe immediately. It records an instruction that 
          simulate() will realize inside the built model.
        • For DNFs, you can target a per-action slice of neurons by providing index; 
          the slice is [index * dnf_neurons: (index + 1) * dnf_neurons]
        """
        key = part.strip().lower()
        valid = {
            "cortex_inputs", "stn", "gpe", "gpi",
            "concentration_layer", "dopamine", "bg_out",
            "d1_dnf", "d2_dnf"
        }
        # Error checking 
        if key not in valid:
            raise ValueError(f"Unknown part '{part}'. Must be one of {sorted(valid)}")
        
        if what not in {"input", "output", "neurons"}:
            raise ValueError(f"what must be 'input', 'output', or 'neurons'")
        
        if what == 'neurons' and field not in {None, 'input', 'spikes', 'voltage'}:
            raise ValueError("field must be None, 'input', 'spikes', 'voltage' when what='neurons'.")
        
        # Default name if not provided 
        if name is None:
            if key is {'stn', 'gpe', 'gpi', 'cortex_inputs', 'd1_dnf', 'd2_dnf'} and index is not None:
                base = f"{key}_{index}"
            else:
                base = key
            suffix = field if (what == 'neurons' and field is not None) else what
            name = f'{base}_{suffix}'
        
        # Record the request
        self._probe_requests.append({
            "part": key,
            "index": index,
            "what": what,
            "field": field,
            "name": name
        })
        
        return name
    
    def  __add_custom_probes(self):
        custom_probes = {}
        # Helper function to create neuron-level probes with field selection 
        def _probe_neurons(ens, field):
            if field is None:
                # for what = 'neurons'
                return nengo.Probe(ens.neurons, synapse=None)
            else:
                # for what = 'input'
                return nengo.Probe(ens.neurons, field, synapse=None)
        
        for req in getattr(self, "_probe_requests", []):
            part = req["part"]
            idx = req["index"]
            what = req["what"]
            fld = req["field"]
            name = req["name"]

            # Map 'part' to objects in the built network
            if part == 'cortex_inputs':
                targets = self.cortex_inputs if idx is None else [self.cortex_inputs[idx]]
                for k, obj in enumerate(targets):
                    nm = name if idx is not None else f"{name.rstrip('_output')}_{k}_{what}"
                    if what == "output":
                        custom_probes[nm] = nengo.Probe(obj, synapse=None)
                    else:
                        raise ValueError("cortex_inputs supports 'output' only")
            elif part in {'stn', 'gpe', 'gpi'}:
                lst = getattr(self, part)
                targets = lst if idx is None else [lst[idx]]
                for k, ens in enumerate(targets):
                    nm = name if idx is not None else f"{name.rstrip('_output')}_{k}_{what if fld is None else fld}"
                    if what == "output":
                        custom_probes[nm] = nengo.Probe(ens, synapse=None)
                    else:
                        custom_probes[nm] = _probe_neurons(ens, fld)
            elif part in {'d1_dnf', 'd2_dnf'}:
                dnf = getattr(self, part)
                # DNFs: typically probe neuron arrays on internal ensemble 'g'
                if what == 'output':
                    # Probe decoded output of DNF's main ensemble
                    custom_probes[name] = nengo.Probe(dnf.g, synapse=None)
                else: 
                    # Optionally slice per action 
                    if idx is None:
                        custom_probes[name] = _probe_neurons(dnf.g, fld)
                    else:
                        start = idx * self.dnf_neurons
                        stop = (idx + 1) * self.dnf_neurons
                        custom_probes[name] = nengo.Probe(dnf.g.neurons[start:stop], fld if fld else None, synapse=None)
            elif part in {'concentration_layer', 'dopamine', 'bg_out'}:
                obj = getattr(self, part)
                if what == 'output':
                    custom_probes[name] = nengo.Probe(obj, synapse=None)
                else:
                    if part == 'concentration_layer':
                        custom_probes[name] = _probe_neurons(obj, fld)
                    else:
                        raise ValueError(f"{part} does not support neuron-level probing.")
            else:
                raise RuntimeError(f"Unhandled probe part: {part}")
            
        return custom_probes
        


    def simulate(self, input_bundles, dopamine_level = 0.0, presentation_time = 1.5, duration = 1.0):
        '''
        Run a simulatiokn of this Basal Ganglia network with the given inputs. 

        Parameters
        ----------
            input_bundles : List
                List of input bundles for N discrete actions, each is 512D SSP vector
            dopamine_level : dict
                Amount of dopamine to present to the Basal Ganglia model
            presentation_time : np.ndarray or list
                Length of time to present the input bundles and dopamine to the model
            duration : float, optional (default=1.0)
                Length of the simulation

        Returns
        -------
            dict: 
                Dictionary containing the network's probes data
                    • bundle_ins: Probes for inputs to Basal Ganglia model
                    • d1_input: Probes D1 Straitum Input
                    • d1_neuron: Probes for D1 Straitum Ensemble neurons
                    • d2_input: Probes for D2 Straitum Input
                    • d2_input: Probes for D2 Straitum Ensemble neurons
                    • bundle_out: Probes for output bundles from Basal Ganglia model
        '''

        model = nengo.Network(label="BG Network")
        with model:
            model.add(self)
            input_bundle_nodes = []
            for i in range(self.n_actions):
                cortical_bundle = nengo.Node(nengo.processes.PresentInput(inputs=[input_bundles[i]], presentation_time=presentation_time), label=f'ssp_cortical_bundle_node_{i}')
                input_bundle_nodes.append(cortical_bundle)
                
            # Dopamine
            dopamine_node = nengo.Node(nengo.processes.PresentInput(inputs = [dopamine_level], presentation_time=presentation_time), label='dopamine')

            # Output Node
            out_node = Node(size_in = self.ssp_dim * self.n_actions)

            # Connections with BG model
            # 1. Dopamine
            nengo.Connection(dopamine_node, self.dopamine, synapse=None)
            # 2. Connect bundles
            for i in range(self.n_actions):
                nengo.Connection(input_bundle_nodes[i], self.cortex_inputs[i], synapse=None)
            # 3. Output node
            nengo.Connection(self.bg_out, out_node, synapse=None)

            # ------------- Custom probes requested via add_probe() ----------# 
            custom_probes = self.__add_custom_probes()
            
            # Default Probes 
            p_ins = []
            for i in range(self.n_actions):
                p_in = nengo.Probe(input_bundle_nodes[i], synapse=None)
                p_ins.append(p_in)
            
            p_dnf_in_d1 = nengo.Probe(self.d1_dnf.g.neurons, 'input', synapse = None)
            p_dnf_neurons_d1= nengo.Probe(self.d1_dnf.g.neurons, synapse=None)

            p_dnf_in_d2 = nengo.Probe(self.d2_dnf.g.neurons, 'input', synapse = None)
            p_dnf_neurons_d2= nengo.Probe(self.d2_dnf.g.neurons, synapse=None)

            p_out = nengo.Probe(out_node, synapse = None)

        # Run the network
        with nengo.Simulator(model) as sim:
            sim.run(duration)

        # Add custom data for probes
        custom_data = {nm: sim.data[p] for nm, p in custom_probes.items()}
        
        return {
            'bundle_ins': [sim.data[p] for p in p_ins],
            'd1_input': sim.data[p_dnf_in_d1],
            'd1_neuron': sim.data[p_dnf_neurons_d1],
            'd2_input': sim.data[p_dnf_in_d2],
            'd2_neuron': sim.data[p_dnf_neurons_d2],
            'bundle_out': sim.data[p_out],
            'custom_probes': custom_data # < -- requested custom probes live here 
        }

def main():
    # Domain 
    domain = np.arange(0, 4, 0.01).reshape(-1, 1) # Shape: (400, 1)
    # SSP encoder
    ssp_encoder = RandomSSPSpace(
        domain_dim = 1, # scalar action values
        ssp_dim = 512, # high dimensional SSPs
        rng=np.random.RandomState(), 
        length_scale=0.5
    )
    # Convert scalar action values into high dimensional SSPs
    domain_phis = ssp_encoder.encode(domain) # Shape: (400, 512)

    # Place cell encoders for Nengo ensembles
    low = 0
    high = 4
    width = high - low
    places_ = np.arange(low, high, width/(domain.shape[0]))
    encoders = np.asarray(ssp_encoder.encode(places_.reshape(-1, 1))).squeeze()

    # Make a single channel basal ganglia
    n_actions = 1
    a = np.random.uniform(low=1, high=10, size=n_actions)
    b = np.random.uniform(low=1, high=10, size=n_actions)
    salience = np.random.uniform(low = 0, high = 1, size = n_actions)
    beta_distribution = beta.pdf(domain, a, b, scale=domain[-1])

    # Create bundle 
    bundle = Utility.to_bundle(Utility.to_scale([salience], Utility.to_normalize(beta_distribution)), domain_phis)
    print("bundle:", bundle.shape)

    # DNF params
    dnf_params = {'h'       : -3.009416816439706,
            'global_inh'      : 8.641108231311897,
            'tau'             : 0.04706404390267922,
            'exc'             : 9.421285790349613,
            'inh'             : 1.4841093480380807,
            'exc_w'           : 8.534993467227224,
            'inh_w'           : 4.748154014869723,
            'dt'              : 0.001, 
            'c_noise'         : 1.0,
            'shape'           : [(n_actions, 400)], 
            'beta'            : 4,
    }
    # Create the Basal Ganglia model 
    bg_model = BasalGanglia(n_actions=n_actions, dnf_parameters=dnf_params, encoders=encoders)
    # Simulate the model 
    data = bg_model.simulate(input_bundles=[bundle], dopamine_level=0.2, presentation_time=1.5, duration=1.5)
    # Display the probes data
    print(data)

if __name__ == 'main':
    main()