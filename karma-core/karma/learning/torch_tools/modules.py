from abc import ABCMeta, abstractmethod, abstractproperty
import torch
from cytoolz.dicttoolz import keyfilter

from karma.core.utils.utils import coerce_to_tuple, coerce_to_tuple_and_check_all_strings
from karma.core.utils.ordereddict import OrderedDict
from karma.core.karmacode import KarmaSequence, KarmaCode
from karma.core.instructions.vectorial import TorchModule, Multiply, Dot, Add, ScalarProduct, DirectProduct, DirectSum
from karma.core.instructions.scalar import Logit
from karma.learning.torch_tools.functions import SparseDPDotFunction, SparseProjectorFunction
from karma.core.utils.utils import use_seed, flatten


CELL_AVAILABLE_INTERACTIONS = ['multiply', 'direct_product', 'scalar_product']


def _projector_naming(side, name, identifier=0):
    return 'PR{}_{}_{}'.format(side.capitalize(), name, identifier)


class AbstractTorchKarmaIntegrator(torch.nn.Module):
    # TODO Docs

    __metaclass__ = ABCMeta

    @abstractproperty
    def name(self):
        pass

    def _set_requires_grad_on_parameters(self, parameters, value_to_set):
        if parameters is None:
            parameters = self.params_dict.keys()
        else:
            parameters = coerce_to_tuple_and_check_all_strings(parameters, 'parameters')
        parameters = set(parameters).intersection(self.params_dict.keys())
        for key in parameters:
            self.params_dict[key].requires_grad = value_to_set

    def freeze(self, parameters=None):
        """
        Sets requires_grad to False on a given list or parameters or for the whole cell
        :param parameters: list of parameters' names, default None = all parameters
        >>> cell = ConcatBiInteractionCell((('input1', 'input2'), 'input3'), {'input1': 10, 'input2': 5, 'input3': 7})

        One can freeze one parameter or list of parameters
        >>> cell.params_dict['PRLeft_input1_0'].requires_grad
        True
        >>> cell.freeze('PRLeft_input1_0')
        >>> cell.params_dict['PRLeft_input1_0'].requires_grad
        False

        or freeze the whole cell when it is a part of larger model
        >>> cell.params_dict['PRRight_input3_0'].requires_grad
        True
        >>> cell.freeze()
        >>> cell.params_dict['PRRight_input3_0'].requires_grad
        False
        """
        self._set_requires_grad_on_parameters(parameters, value_to_set=False)

    def unfreeze(self, parameters=None):
        """
        inverse operation for freeze
        :param parameters: list of parameters' names, default None = all parameters
        """
        self._set_requires_grad_on_parameters(parameters, value_to_set=True)

    @use_seed()
    def reset_parameters(self):
        for m in self.children():
            m.reset_parameters()

    def to_karmacode(self, inp='', out=None):
        out = 'out_{}'.format(self.name) if out is None else out
        return KarmaCode([TorchModule(inp, out, self)], out)

    @property
    def children_dict(self):
        return dict(self.named_children())

    @property
    def params_dict(self):
        return dict(self.named_parameters())

    @abstractmethod
    def forward(self, *input):
        pass


class KarmaLinearModule(AbstractTorchKarmaIntegrator):

    name = 'karma_linear_module'

    def __init__(self, in_dim, out_dim=None, bias=True, sigma=0.1, seed=None):
        super(KarmaLinearModule, self).__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.bias = bias
        if self.out_dim is None:
            self.projector = torch.nn.Parameter(torch.Tensor(in_dim))
        else:
            self.projector = torch.nn.Parameter(torch.Tensor(in_dim, out_dim))
        if self.bias:
            if self.out_dim is None:
                self.bias = torch.nn.Parameter(torch.Tensor(1))
            else:
                self.bias = torch.nn.Parameter(torch.Tensor(out_dim))

        self.sigma = sigma
        self.reset_parameters(seed=seed)

    @use_seed()
    def reset_parameters(self):
        self.projector.data.normal_(0, self.sigma)
        if isinstance(self.bias, torch.nn.Parameter):
            self.bias.data.zero_()

    def forward(self, inp):
        res = SparseProjectorFunction.apply(inp, self.projector)
        if isinstance(self.bias, torch.nn.Parameter):
            res += self.bias
        return res

    def to_karmacode(self, inp='', out=None):
        assert len(coerce_to_tuple_and_check_all_strings(inp)) == 1
        ks = KarmaSequence()
        out = 'linear_{}'.format(inp) if out is None else out
        ks.append(Dot(inp, out, rightfactor=self.projector.detach().numpy()))
        if isinstance(self.bias, torch.nn.Parameter):
            ks.append(Add('linear_{}'.format(inp), out, constant=self.bias.detach().numpy()))
        return KarmaCode(instructions=ks, outputs=out)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}, bias={}'.format(self.in_dim, self.out_dim, self.bias)


class LinearCell(AbstractTorchKarmaIntegrator):

    name = 'linear_cell'

    def __init__(self, size_dict, bias=True, sigma=0.1, seed=None):
        super(LinearCell, self).__init__()
        self.size_dict = OrderedDict(size_dict)
        for name, size in self.size_dict.items()[:-1]:
            self.add_module('linear_{}'.format(name), KarmaLinearModule(size, bias=False, sigma=sigma,
                                                                        seed=seed))
        self.add_module('linear_{}'.format(self.size_dict.keys()[-1]),
                        KarmaLinearModule(self.size_dict.values()[-1], bias=bias, sigma=sigma, seed=seed))

    @property
    def features(self):
        return tuple(self.size_dict.keys())

    def forward(self, ds):
        res = 0
        for name in self.size_dict.keys():
            res += self.children_dict['linear_{}'.format(name)](ds[name][:])
        return res

    def to_karmacode(self, inp='', out=None):
        out = 'linear' if out is None else out
        kc = KarmaCode()
        for name in self.size_dict.keys():
            kc += self.children_dict['linear_{}'.format(name)].to_karmacode(name)
        kc += KarmaCode([Add(kc.outputs, out)], outputs=out)
        return kc.with_outputs(out)

    def extra_repr(self):
        return '{}'.format(self.size_dict)


class ConcatBiInteractionCell(AbstractTorchKarmaIntegrator):

    name = 'concat_bi_interaction_cell'

    def __init__(self, config, size_dict, interaction='multiply', embed_dim=25, sigma=0.1, identifier=0, seed=None):
        """config should look like
        (('a', 'b', 'c'), ('f', 'e'))
         or   ('d', 'k')
        """
        super(ConcatBiInteractionCell, self).__init__()
        if interaction not in CELL_AVAILABLE_INTERACTIONS:
            raise ValueError('{} not in available interactions: {}'.format(self.interaction,
                                                                           CELL_AVAILABLE_INTERACTIONS))
        assert(isinstance(identifier, int))
        self.config = config
        self.left, self.right = map(coerce_to_tuple_and_check_all_strings, config)
        self.embed_dim = embed_dim
        self.interaction = interaction
        self.size_dict = OrderedDict(size_dict)
        self.id = identifier

        for side in ['left', 'right']:
            for name in getattr(self, side):
                self.register_parameter(_projector_naming(side, name, self.id),
                                        torch.nn.Parameter(torch.Tensor(self.size_dict[name], self.embed_dim)))

        self.sigma = sigma
        self.reset_parameters(seed=seed)

    @use_seed()
    def reset_parameters(self):
        for p in self.parameters():
            p.data.normal_(0, self.sigma)

    @property
    def features(self):
        return tuple(flatten(self.config))

    @property
    def out_dim(self):
        if self.interaction == 'multiply':
            return self.embed_dim
        elif self.interaction == 'scalar_product':
            return 1
        elif self.interaction == 'direct_product':
            return self.embed_dim**2

    def forward(self, ds):
        """
        ds should be a dict : column -> array
        """
        left_pr = 0
        for name in self.left:
            left_pr += SparseProjectorFunction.apply(ds[name][:], self.params_dict[_projector_naming('left',
                                                                                                     name, self.id)])
        right_pr = 0
        for name in self.right:
            right_pr += SparseProjectorFunction.apply(ds[name][:], self.params_dict[_projector_naming('right',
                                                                                                      name, self.id)])

        if self.interaction == 'multiply':
            left_pr *= right_pr
            return left_pr
        elif self.interaction == 'scalar_product':
            return torch.einsum('ij,ij->i', left_pr, right_pr).view(-1, 1)
        elif self.interaction == 'direct_product':
            return torch.einsum('bi,bj->bij', left_pr, right_pr).view(left_pr.size(0), -1)

    def to_karmacode(self, inp='', out=None):
        out = 'out_bi_interaction_{}'.format(self.id) if out is None else '{}_{}'.format(out, self.id)
        ks = KarmaSequence()

        for side in ['left', 'right']:
            names = getattr(self, side)
            for name in names:
                ks.append(Dot((name,), _projector_naming(side, name, self.id),
                              rightfactor=self.params_dict[_projector_naming(side, name, self.id)].detach().numpy()))
            ks.append(Add([_projector_naming(side, n, self.id) for n in names], 'out_{}_{}'.format(side, self.id)))

        instr_args = [('out_left_{}'.format(self.id), 'out_right_{}'.format(self.id)), out]
        if self.interaction == 'multiply':
            ks.append(Multiply(*instr_args))
        elif self.interaction == 'scalar_product':
            ks.append(ScalarProduct(*instr_args))
        elif self.interaction == 'direct_product':
            ks.append(DirectProduct(*instr_args))

        return KarmaCode(instructions=ks, outputs=out)

    def extra_repr(self):
        return 'left_{}, right_{}, embed_dim_{}'.format(self.left, self.right, self.embed_dim)


class DeepLayersCell(AbstractTorchKarmaIntegrator):

    name = 'deep_layers_module'

    def __init__(self, sizes, activation='relu', seed=None):
        super(DeepLayersCell, self).__init__()
        self.sizes = sizes
        self.activation = activation

        for i in range(len(self.sizes) - 1):
            self.add_module('deep_layer_{}'.format(i), torch.nn.Linear(self.sizes[i], self.sizes[i + 1], bias=False))
        self.add_module('deep_layer_{}'.format(len(self.sizes) - 1), torch.nn.Linear(self.sizes[-1], 1))

        self.reset_parameters(seed=seed)

    def forward(self, res):
        for i in range(len(self.sizes) - 1):
            res = getattr(torch, self.activation)(self.children_dict['deep_layer_{}'.format(i)](res))
        res = self.children_dict['deep_layer_{}'.format(len(self.sizes) - 1)](res)
        return res

    def extra_repr(self):
        return 'activation:{}, sizes:{}'.format(self.activation, self.sizes)


module_mapping = {'concat_bi_interaction_cell': ConcatBiInteractionCell}


class DeepFM(AbstractTorchKarmaIntegrator):

    name = 'model_compiler'

    def __init__(self, model_kwargs, seed=None):

        super(DeepFM, self).__init__()
        self.output = model_kwargs.get('output')
        self.size_dict = model_kwargs['size_dict']
        self.features = model_kwargs['features']
        assert(set(self.features).issubset(self.size_dict.keys()))
        self.base_cells = DeepFM._create_cell_objects(DeepFM._get_base_cells_dict(model_kwargs), self.size_dict, seed)
        for name, cell in self.base_cells.items():
            self.add_module(name, cell)
        base_out_dim = sum(map(lambda base_cell: base_cell.out_dim, self.base_cells.values()))

        deep_layers = model_kwargs.get('deep_layer')
        if deep_layers is not None:
            if isinstance(deep_layers, torch.nn.Module):
                self.deep_layers = deep_layers
            else:
                self.deep_layers = DeepLayersCell((base_out_dim,) + coerce_to_tuple(deep_layers['hidden_sizes']),
                                                  deep_layers['activation'], seed=seed)
        else:
            self.deep_layers = DeepLayersCell((base_out_dim,), None, seed=seed)
        self.fit_linear = model_kwargs.get('fit_linear', False)
        if self.fit_linear:
            self.linear = LinearCell(size_dict=keyfilter(lambda x: x in self.features, self.size_dict), seed=seed)

    @staticmethod
    def _get_base_cells_dict(model_kwargs):
        if isinstance(model_kwargs['base_cells'], dict):
            return OrderedDict(model_kwargs['base_cells'].items())
        else:
            base_cells_dict = OrderedDict()
            for idx, cell in enumerate(coerce_to_tuple(model_kwargs['base_cells'])):
                type_ = cell._get_name() if isinstance(cell, torch.nn.Module) else cell['type']
                base_cells_dict['{}_{}'.format(type_, idx)] = cell
            return base_cells_dict

    @staticmethod
    def _create_cell_objects(base_cells_dict, size_dict, seed=None):
        for idx, name in enumerate(base_cells_dict.keys()):
            base_cell = base_cells_dict[name]
            if isinstance(base_cell, torch.nn.Module):
                # this is needed since we give ds into forward pass of base_cells
                assert isinstance(base_cell, tuple(module_mapping.values()))
            else:
                base_cell = module_mapping[base_cell['type']](seed=seed, identifier=idx, size_dict=size_dict,
                                                              **base_cell['cell_kwargs'])
                base_cells_dict[name] = base_cell
        return base_cells_dict

    def forward(self, ds):
        res = torch.cat(map(lambda cell: cell(ds), self.base_cells.values()), dim=1)
        res = self.deep_layers(res).reshape(-1)
        if self.fit_linear:
            res += self.linear(ds)

        return torch.sigmoid(res)

    def to_karmacode(self, inp='', out=None):
        out = self.output if out is None else out
        kc = KarmaCode()
        for base_cell in self.base_cells.values():
            kc += base_cell.to_karmacode()
        kc += KarmaCode([DirectSum(kc.outputs, 'base_cells_agg_out')], outputs='base_cells_agg_out')
        if hasattr(self.deep_layers, 'to_karmacode'):
            kc_deep = self.deep_layers.to_karmacode('base_cells_agg_out')
        else:
            kc_deep = TorchModule('base_cells_agg_out', 'out_deep', self.deep_layers).to_karmacode()
        kc += kc_deep
        if self.fit_linear:
            kc += self.linear.to_karmacode()
            kc += KarmaCode([Add(kc_deep.outputs + self.linear.to_karmacode().outputs, 'add_lin_deep')],
                            outputs='add_lin_deep')
            kc += KarmaCode([Logit(('add_lin_deep',), out)], outputs=out)
            return kc.with_outputs(out)
        else:
            kc += KarmaCode([Logit(kc_deep.outputs, out)], outputs=out)
            return kc.with_outputs(out)
