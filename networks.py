from torch import nn
import torch
import math 

class GRFNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GRFNet, self).__init__()
        hidden = 512  
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)  
        self.fc_conF = nn.Linear(hidden, 4 * 3) 
        self.LReLU1 = nn.LeakyReLU(0.1)
        self.LReLU2 = nn.LeakyReLU(0.1)
        self.LReLU3 = nn.LeakyReLU(0.1)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.LReLU1(self.fc1(x))
        x = self.LReLU2(self.fc2(x))
        x = self.LReLU3(self.fc3(x))
        conF = 10 * self.fc_conF(x)
        return conF
class DynamicNetwork(nn.Module):
    def __init__(self, input_dim,output_dim,offset_coef):
        super(DynamicNetwork, self).__init__()
        hidden = 1024 
        self.bn1 = nn.BatchNorm1d(input_dim, affine=True)
        self.bn2 = nn.BatchNorm1d(hidden, affine=True)
        self.bn3 = nn.BatchNorm1d(hidden, affine=True)

        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_offset = nn.Linear(hidden, output_dim)
        self.fc_gains = nn.Linear(hidden, output_dim) 
        self.sig=nn.Sigmoid() 
        self.LReLU = nn.LeakyReLU(0.1)
        self.offset_coef=offset_coef
    def forward(self, x):
        x = self.LReLU( self.fc1(x))#)
        x = self.LReLU( self.fc2(x))#) 
        gains = 2 *self.sig(self.fc_gains(x))
        offset =self.offset_coef * self.tanh(self.fc_offset(x))
       
        return gains,offset

class TransCan3Dkeys2(nn.Module):
    def __init__(self, in_channels=74, num_features=256, out_channels=44, time_window=10, num_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.time_window = time_window
        self.conv1 = nn.Sequential(
            nn.ReplicationPad1d(1),
            nn.Conv1d(self.in_channels, self.num_features, kernel_size=3, bias=False ),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25)
        )
        self._make_blocks()
        self.pad = nn.ReplicationPad1d(4)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.25)
        self.reduce = nn.Conv1d(self.num_features, self.num_features, kernel_size= self.time_window  )
        self.embedding_3d_1 = nn.Linear(3*16,500)
        self.embedding_3d_2 = nn.Linear(500,500)
        self.LReLU1=nn.LeakyReLU()
        self.LReLU2=nn.LeakyReLU()
        self.LReLU3=nn.LeakyReLU()
        self.LReLU4=nn.LeakyReLU()
        self.out1 = nn.Linear(self.num_features+500, self.num_features)
        self.out2 = nn.Linear(self.num_features, self.num_features)
        self.out3 = nn.Linear(self.num_features, self.out_channels)
        
    def _make_blocks(self):
        layers_conv = []
        layers_bn = []
        for i in range(self.num_blocks):
            layers_conv.append(nn.Conv1d(self.num_features, self.num_features, kernel_size=5, bias=False,dilation=2))
            layers_bn.append(nn.BatchNorm1d(self.num_features))
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def forward(self,  p2ds,p3d): 
        """
        Args:
        x - (B x T x J x C)
        """

        B,T,C = p2ds.shape
        x = p2ds.permute((0, 2, 1))
        x = self.conv1(x)
        for i in range(self.num_blocks):
            pre = x
            x = self.pad(x)
            x = self.layers_conv[i](x)
            x = self.layers_bn[i](x)
            x = self.drop(self.relu(x))
            x = pre + x
        x_2d = self.relu(self.reduce(x))
        x_2d = x_2d.view(B, -1)
        x_3d = self.LReLU1(self.embedding_3d_1(p3d))
        x = torch.cat((x_2d,x_3d),1)
        x = self.LReLU3(self.out1(x))
        x = self.LReLU4(self.out2(x))
        x = self.out3(x)
        return x

class TransCan3Dkeys(nn.Module):
    def __init__(self, in_channels=74, num_features=256, out_channels=44, time_window=10, num_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.time_window = time_window
        self.conv1 = nn.Sequential(
            nn.ReplicationPad1d(1),
            nn.Conv1d(self.in_channels, self.num_features, kernel_size=3, bias=False ),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25)
        )
        self._make_blocks()
        self.pad = nn.ReplicationPad1d(4)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.25)
        self.reduce = nn.Conv1d(self.num_features, self.num_features, kernel_size= self.time_window  )
        self.embedding_3d_1 = nn.Linear(3*16,500)
        self.embedding_3d_2 = nn.Linear(500,500)
        self.LReLU1=nn.LeakyReLU()
        self.LReLU2=nn.LeakyReLU()
        self.LReLU3=nn.LeakyReLU()
        self.out1 = nn.Linear(self.num_features+500, self.num_features)
        self.out2 = nn.Linear(self.num_features, self.out_channels)

    def _make_blocks(self):
        layers_conv = []
        layers_bn = []
        for i in range(self.num_blocks):
            layers_conv.append(nn.Conv1d(self.num_features, self.num_features, kernel_size=5, bias=False,dilation=2))
            layers_bn.append(nn.BatchNorm1d(self.num_features))
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def forward(self,  p2ds,p3d): 
        """
        Args:
        x - (B x T x J x C)
        """

        B,T,C = p2ds.shape
        x = p2ds.permute((0, 2, 1))
        x = self.conv1(x)
        for i in range(self.num_blocks):
            pre = x
            x = self.pad(x)
            x = self.layers_conv[i](x)
            x = self.layers_bn[i](x)
            x = self.drop(self.relu(x))
            x = pre + x
        x_2d = self.relu(self.reduce(x))
        x_2d = x_2d.view(B, -1)
        x_3d = self.LReLU1(self.embedding_3d_1(p3d))
        x = torch.cat((x_2d,x_3d),1)
        x = self.LReLU3(self.out1(x))
        x = self.out2(x)
        return x

class ContactEstimationNetwork(nn.Module):
    def __init__(self, in_channels=74, num_features=256, out_channels=44, time_window=10, num_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.time_window = time_window
        self.conv1 = nn.Sequential(
            nn.ReplicationPad1d(1),
            nn.Conv1d(self.in_channels, self.num_features, kernel_size=3, bias=False ),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25)
        )
        self._make_blocks()
        self.pad = nn.ReplicationPad1d(4)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.25)
        self.reduce = nn.Conv1d(self.num_features, self.num_features, kernel_size= self.time_window  )

        self.out = nn.Linear(self.num_features, self.out_channels)
        self.sig = nn.Sigmoid()
    def _make_blocks(self):
        layers_conv = []
        layers_bn = []
        for i in range(self.num_blocks):
            layers_conv.append(nn.Conv1d(self.num_features, self.num_features, kernel_size=5, bias=False,dilation=2))
            layers_bn.append(nn.BatchNorm1d(self.num_features))
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def forward(self,x): 
        """
        Args:
        x - (B x T x J x C)
        """

        B,T,C = x.shape

        x = x.permute((0, 2, 1))
        x = self.conv1(x)

        for i in range(self.num_blocks):
            pre = x
            x = self.pad(x)
            x = self.layers_conv[i](x)
            x = self.layers_bn[i](x)
            x = self.drop(self.relu(x))
            x = pre + x
        x = self.relu(self.reduce(x))
        x = x.view(B, -1)
        x = self.out(x)
        pred = self.sig(x)
        return pred

class TargetPoseNetOriCon(nn.Module):
    def __init__(self, in_channels=74, num_features=256, out_channels=44, time_window=10, num_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.time_window = time_window
        self.conv1 = nn.Sequential(
            nn.ReplicationPad1d(1),
            nn.Conv1d(self.in_channels, self.num_features, kernel_size=3, bias=False ),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25)
        )
        self._make_blocks()
        self.pad = nn.ReplicationPad1d(4)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.25)
        self.reduce = nn.Conv1d(self.num_features, self.num_features, kernel_size= self.time_window  )
        self.tanh=nn.Tanh()
        self.sig = nn.Sigmoid()
        self.outCon = nn.Linear(self.num_features, 4)
        self.outOri = nn.Linear(self.num_features, 4)
    def _make_blocks(self):
        layers_conv = []
        layers_bn = []
        for i in range(self.num_blocks):
            layers_conv.append(nn.Conv1d(self.num_features, self.num_features, kernel_size=5, bias=False,dilation=2))
            layers_bn.append(nn.BatchNorm1d(self.num_features))
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def forward(self,x): 
        """
        Args:
        x - (B x T x J x C)
        """
        B,T,C = x.shape
        x = x.permute((0, 2, 1))
        x = self.conv1(x)
        for i in range(self.num_blocks):
            pre = x
            x = self.pad(x)
            x = self.layers_conv[i](x)
            x = self.layers_bn[i](x)
            x = self.drop(self.relu(x))
            x = pre + x
        x = self.relu(self.reduce(x))
        x = x.view(B, -1)
        x_con = self.sig(self.outCon(x))
        x_ori =  self.outOri(x)
        return x_con,x_ori



class TargetPoseNetArtOri(nn.Module):
    def __init__(self, in_channels=74, num_features=256, out_channels=44, time_window=10, num_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.time_window = time_window
        self.conv1 = nn.Sequential(
            nn.ReplicationPad1d(1),
            nn.Conv1d(self.in_channels, self.num_features, kernel_size=3, bias=False ),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25)
        )
        self._make_blocks()
        self.pad = nn.ReplicationPad1d(4)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.25)
        self.reduce = nn.Conv1d(self.num_features, self.num_features, kernel_size= self.time_window  )
        self.tanh=nn.Tanh()

        self.out = nn.Linear(self.num_features, self.out_channels)
        self.outOri = nn.Linear(self.num_features, 4)
    def _make_blocks(self):
        layers_conv = []
        layers_bn = []
        for i in range(self.num_blocks):
            layers_conv.append(nn.Conv1d(self.num_features, self.num_features, kernel_size=5, bias=False,dilation=2))
            layers_bn.append(nn.BatchNorm1d(self.num_features))
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def forward(self,x): 
        """
        Args:
        x - (B x T x J x C)
        """
        B,T,C = x.shape
        x = x.permute((0, 2, 1))
        x = self.conv1(x)
        for i in range(self.num_blocks):
            pre = x
            x = self.pad(x)
            x = self.layers_conv[i](x)
            x = self.layers_bn[i](x)
            x = self.drop(self.relu(x))
            x = pre + x
        x = self.relu(self.reduce(x))
        x = x.view(B, -1)
        x_art = math.pi*self.tanh(self.out(x))
        x_ori =  self.outOri(x)
        return x_art,x_ori

class TargetPoseNetArt(nn.Module):
    def __init__(self, in_channels=74, num_features=256, out_channels=44, time_window=10, num_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.time_window = time_window
        self.conv1 = nn.Sequential(
            nn.ReplicationPad1d(1),
            nn.Conv1d(self.in_channels, self.num_features, kernel_size=3, bias=False),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25)
        )
        self._make_blocks()
        self.pad = nn.ReplicationPad1d(4)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.25)
        self.reduce = nn.Conv1d(self.num_features, self.num_features, kernel_size=self.time_window)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(self.num_features, self.out_channels)

    def _make_blocks(self):
        layers_conv = []
        layers_bn = []
        for i in range(self.num_blocks):
            layers_conv.append(
                nn.Conv1d(self.num_features, self.num_features, kernel_size=5, bias=False, dilation=2))
            layers_bn.append(nn.BatchNorm1d(self.num_features))
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def forward(self, x): 
        """
        Args:
        x - (B x T x J x C)
        """
        B, T, C = x.shape
        x = x.permute((0, 2, 1))
        x = self.conv1(x)
        for i in range(self.num_blocks):
            pre = x
            x = self.pad(x)
            x = self.layers_conv[i](x)
            x = self.layers_bn[i](x)
            x = self.drop(self.relu(x))
            x = pre + x
        x = self.relu(self.reduce(x))
        x = x.view(B, -1)
        x = math.pi * self.tanh(self.out(x))
        return x


class TargetPoseNetOri(nn.Module):
    def __init__(self, in_channels=74, num_features=256, out_channels=44, time_window=10, num_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.time_window = time_window
        self.conv1 = nn.Sequential(
            nn.ReplicationPad1d(1),
            nn.Conv1d(self.in_channels, self.num_features, kernel_size=3, bias=False ),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25)
        )
        self._make_blocks()
        self.pad = nn.ReplicationPad1d(4)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.25)
        self.reduce = nn.Conv1d(self.num_features, self.num_features, kernel_size= self.time_window  )

        self.out = nn.Linear(self.num_features, self.out_channels)

    def _make_blocks(self):
        layers_conv = []
        layers_bn = []
        for i in range(self.num_blocks):
            layers_conv.append(nn.Conv1d(self.num_features, self.num_features, kernel_size=5, bias=False,dilation=2))
            layers_bn.append(nn.BatchNorm1d(self.num_features))
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def forward(self,x):
        """
        Args:
        x - (B x T x J x C)
        """

        B,T,C = x.shape

        x = x.permute((0, 2, 1))
        x = self.conv1(x)
        for i in range(self.num_blocks):
            pre = x
            x = self.pad(x)
            x = self.layers_conv[i](x)
            x = self.layers_bn[i](x)
            x = self.drop(self.relu(x))
            x = pre + x
        x = self.relu(self.reduce(x))
        x = x.view(B, -1)
        x = self.out(x)
        return x