import torch
from torch import nn
from src.models.act import select_act
from datetime import datetime


class MLP_HiddenBlock(nn.Module):
    """

    """
    def __init__(self, dim_in, dim_out, bias=True, act=nn.ReLU(True)):
        super(MLP_HiddenBlock, self).__init__()
        m = [nn.Linear(dim_in, dim_out, bias=bias), act]
        self.hidden_block = nn.Sequential(*m)

    def forward(self, x):
        block_output = self.hidden_block(x)
        return block_output


class DNN(nn.Module):
    def __init__(self, name='DNN', network_hpara=None):
        """

        :param name:
        :param network_hpara: (dict)
            'dim_in': (int) dimension of input.
            'dim_out': (int) dimension of output.
            'depth': (int) total depth.
                if depth == 1: Linear model.
                if depth == 2: Two-layer model.
                if depth >= 3: vc
        """
        super(DNN, self).__init__()
        if network_hpara is None:
            network_hpara = {'dim_in': 10,
                             'dim_out': 2,
                             'depth': 3,
                             'width': 5,
                             'act': 'relu'}
        self.name = name
        self.dim_in = network_hpara['dim_in']
        self.dim_out = network_hpara['dim_out']
        self.depth = network_hpara['depth']
        self.width = network_hpara['width']
        self.act = network_hpara['act']
        self.act_obj = select_act(self.act)

        self.repre_dict = {}
        for i in range(self.depth):
            self.repre_dict[f'block_{i}'] = None
        self.output = None

        if isinstance(self.width, int):  # Rectangular network
            if self.depth >= 3:  # Deep model
                self.modules_head = nn.ModuleList([nn.Flatten(),
                                                   nn.Linear(self.dim_in, self.width),
                                                   self.act_obj])
                # for i in range(self.depth-2):
                #     self.modules_body.append(MLP_HiddenBlock(self.width, self.width, act=self.act_obj))
                self.modules_body = nn.ModuleList([MLP_HiddenBlock(self.width, self.width, act=self.act_obj)
                                                   for _ in range(self.depth-2)])
                self.modules_tail = nn.ModuleList([nn.Linear(self.width, self.dim_out)])

            elif self.depth == 2:  # Two-layer model
                self.modules_head = nn.ModuleList([nn.Flatten(),
                                                   nn.Linear(self.dim_in, self.width),
                                                   self.act_obj])
                self.modules_body = nn.ModuleList()
                self.modules_tail = nn.ModuleList([nn.Linear(self.width, self.dim_out)])

            elif self.depth == 1:  # Linear model
                self.modules_head = nn.ModuleList([nn.Flatten(),
                                                   nn.Linear(self.dim_in, self.dim_out)])
                self.modules_body = nn.ModuleList()
                self.modules_tail = nn.ModuleList()

        elif isinstance(self.width, list):
            if self.depth >= 3:  # Deep model
                self.modules_head = nn.ModuleList([nn.Flatten(),
                                                   nn.Linear(self.dim_in, self.width[0]),
                                                   self.act_obj])
                # for i in range(self.depth-2):
                #     self.modules_body.append(MLP_HiddenBlock(self.width[i], self.width[i+1], act=self.act_obj))
                self.modules_body = nn.ModuleList([MLP_HiddenBlock(self.width[i], self.width[i+1], act=self.act_obj)
                                                   for i in range(self.depth-2)])
                self.modules_tail = nn.ModuleList([nn.Linear(self.width[-1], self.dim_out)])
            elif self.depth == 2:  # Two-layer model
                self.modules_head = nn.ModuleList([nn.Flatten(),
                                                   nn.Linear(self.dim_in, self.width[0]),
                                                   self.act_obj])
                self.modules_body = nn.ModuleList()
                self.modules_tail = nn.ModuleList([nn.Linear(self.width[0], self.dim_out)])
            elif self.depth == 1:  # Linear model
                self.modules_head = nn.ModuleList([nn.Flatten(),
                                                   nn.Linear(self.dim_in, self.dim_out)])
                self.modules_body = nn.ModuleList()
                self.modules_tail = nn.ModuleList()
        else:
            exit(f'{datetime.now()} E dnn.DNN.__init__(): Unsupported width argument: {self.width}')

    def forward(self, x):
        self.repre_dict['block_0'] = x
        for i_module in self.modules_head:
            self.repre_dict['block_0'] = i_module(self.repre_dict['block_0'])

        for i, i_module in enumerate(self.modules_body):
            self.repre_dict[f'block_{i+1}'] = i_module(self.repre_dict[f'block_{i}'])

        if self.depth >= 2:
            self.repre_dict[f'block_{self.depth-1}'] = self.repre_dict[f'block_{self.depth-2}']
            for i_module in self.modules_tail:
                self.repre_dict[f'block_{self.depth-1}'] = i_module(self.repre_dict[f'block_{self.depth-1}'])

        self.output = self.repre_dict[f'block_{self.depth-1}']
        return self.output


class TishbyNet(nn.Module):
    def __init__(self, name='TishbyNet', network_hpara=None):
        """
        The network for synthetic dataset presented in Tishby's work:
        https://zhaoyanlyu.notion.site/Tishby-IP-e39353fce75e43758c8de300b7463e73
        12-10-7-5-4-3-2
        :param name:
        :param network_hpara:
        """
        super(TishbyNet, self).__init__()
        if network_hpara is None:
            network_hpara = {'act': 'relu'}
        self.name = name
        self.dim_in = 12
        self.dim_out = 2
        self.depth = 6
        self.act = network_hpara['act']
        self.act_obj = select_act(self.act)

        self.repre_dict = {}
        for i in range(self.depth):
            self.repre_dict[f'block_{i}'] = None
        self.output = None

        self.modules_list = nn.ModuleList([
            nn.Sequential(nn.Linear(self.dim_in, 10, bias=True), self.act_obj),
            nn.Sequential(nn.Linear(10, 7, bias=True), self.act_obj),
            nn.Sequential(nn.Linear(7, 5, bias=True), self.act_obj),
            nn.Sequential(nn.Linear(5, 4, bias=True), self.act_obj),
            nn.Sequential(nn.Linear(4, 3, bias=True), self.act_obj),
            nn.Sequential(nn.Linear(3, self.dim_out, bias=True)),
        ])

    def forward(self, x: torch.tensor) -> torch.tensor:
        for i, i_module in enumerate(self.modules_list):
            if i == 0:
                self.repre_dict['block_0'] = i_module(x)
            else:
                self.repre_dict[f'block_{i}'] = i_module(self.repre_dict[f'block_{i-1}'])

        self.output = self.repre_dict[f'block_{self.depth-1}']
        return self.output
