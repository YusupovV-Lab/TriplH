import numpy as np
import torch
import torch.nn.functional as F

def arcosh(x, eps=1e-5):
    x = x.clamp(-1 + eps, 1 - eps)
    return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))


class FeaturesLinear(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
         """
         Linear Feature layer  
         """
        super().__init__()
        print(sum(field_dims))
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]))

    def forward(self, x, items = False):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if not items:
            x = x + x.new_tensor(self.offsets).unsqueeze(0)
            return torch.sum(self.fc(x), dim=1) + self.bias
        else:
            offset = x.new_tensor(self.offsets)
            offset[0] = offset[1] 
            offset = offset.unsqueeze(0)
            x = x + offset
            return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        """
        Embedding Layer
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]))
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x, items = False):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if not items:
            x = x + x.new_tensor(self.offsets).unsqueeze(0)
            return self.embedding(x)
        else:
            offset = x.new_tensor(self.offsets)
            offset[0] = offset[1] 
            offset = offset.unsqueeze(0)
            x = x + offset
            return self.embedding(x)





class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        """
        Factorization Machine layer (Rendle et. al. 2010)
        """
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix
    
class PoincareDistance(torch.nn.Module):
    
    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x[:, 1] = -x[:, 1] 
        top_value = torch.sum(torch.sum(x, dim=1) ** 2, dim=1, keepdim=True)
        bottom_value = (1 - torch.sum(x[:, 0] ** 2, dim=1, keepdim=True)) * (1 - torch.sum(x[:, 1] ** 2, dim=1, keepdim=True))
        val = 1 + 2 * torch.div(top_value, bottom_value)
        d = arcosh(val)
        return d


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        """
        MLP (Multi Layer Perceptron) model
        """
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class InnerProductNetwork(torch.nn.Module):

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:, row] * x[:, col], dim=2)


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)


    def forward(self, x, items = False):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        #print("Input FM:", x.shape)
        x = self.linear(x, items) + self.fm(self.embedding(x, items))
        return torch.sigmoid(x.squeeze(1))

class LorentzFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Lorentz Factorization Machine.

    Reference:
        Wu et al, Lorentz Factorization Machines, 2019.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        #self.inner_product = InnerProductNetwork() 
        self.inner_product = FactorizationMachine(reduce_sum=True)
        self.a = torch.nn.Parameter(torch.tensor(1.0))  # Изначальное значение 1
        self.b = torch.nn.Parameter(torch.tensor(0.0))  # Изначальное значение 0
        
        #self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x, items = False):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        feature_embedding = self.embedding(x, items)
        #print("Feature_emb:", feature_embedding.shape)
        inner_product = self.inner_product(feature_embedding)
        #print("Inner_product:", inner_product.shape)
        zeroth_components = self.get_zeroth_components(feature_embedding)
        x = self.triangle_pooling(inner_product, zeroth_components)
        #print("X_shape:", x.shape)
        #print(x.shape)
        return torch.sigmoid(x.squeeze(1) - 0.5)
    
    def get_zeroth_components(self, feature_emb):
        '''
        compute the 0th component
        '''
        sum_of_square = torch.sum(feature_emb ** 2, dim=-1) # batch * field
        zeroth_components = torch.sqrt(sum_of_square + 1) # beta = 1
        return zeroth_components # batch * field
    
    def triangle_pooling(self, inner_product, zeroth_components):
        '''
        T(u,v) = (1 - <u, v>L - u0 - v0) / (u0 * v0)
               = (1 + u0 * v0 - inner_product - u0 - v0) / (u0 * v0)
               = 1 + (1 - inner_product - u0 - v0) / (u0 * v0)
        '''
        num_fields = zeroth_components.size(1)
        p, q = zip(*list(combinations(range(num_fields), 2)))
        u0, v0 = zeroth_components[:, p], zeroth_components[:, q]  # batch * (f(f-1)/2)
        score_tensor = 1 + torch.div(1 - inner_product - u0 - v0, u0 * v0) # batch * (f(f-1)/2)
        output = torch.sum(score_tensor, dim=1, keepdim=True) # batch * 1
        return output

class HyperBPRModel(torch.nn.Module):
    """
    A pytorch implementation of Lorentz Factorization Machine.

    Reference:
        Wu et al, Lorentz Factorization Machines, 2019.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        #self.inner_product = InnerProductNetwork() 
        self.poincdist = PoincareDistance()
        self.a = torch.nn.Parameter(torch.tensor(1.0))  # Изначальное значение 1
        self.b = torch.nn.Parameter(torch.tensor(1.0))  # Изначальное значение 0
        self.c = torch.nn.Parameter(torch.tensor(0.0))
        
        #self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x, items = False):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        feature_embedding = self.embedding(x, items)
        d = self.poincdist(feature_embedding)
        return torch.sigmoid(d.squeeze(1))


class TetrahedronFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Lorentz Factorization Machine.

    Reference:
        Wu et al, Lorentz Factorization Machines, 2019.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        #self.inner_product = InnerProductNetwork() 
        self.inner_product = FactorizationMachine(reduce_sum=True)
        self.a = torch.nn.Parameter(torch.tensor(1.0))  # Изначальное значение 1
        self.b = torch.nn.Parameter(torch.tensor(0.0))  # Изначальное значение 0
        self.c = torch.nn.Parameter(torch.tensor(0.1))
        self.d = torch.nn.Parameter(torch.tensor(0.0))
        #self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x, items = False):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        feature_embedding = self.embedding(x, items)
        #print("Feature_emb:", feature_embedding.shape)
        inner_product = self.inner_product(feature_embedding)
        #print("Inner_product:", inner_product.shape)
        zeroth_components = self.get_zeroth_components(feature_embedding)
        x = self.triangle_pooling(inner_product, zeroth_components)
        #print("X_shape:", x.shape)
        #print(x.shape)
        #return torch.sigmoid(x.squeeze(1) - 0.5), feature_embedding
        return x, inner_product, zeroth_components
    
    def get_zeroth_components(self, feature_emb):
        '''
        compute the 0th component
        '''
        sum_of_square = torch.sum(feature_emb ** 2, dim=-1) # batch * field
        zeroth_components = torch.sqrt(sum_of_square + 1) # beta = 1
        return zeroth_components # batch * field
    
    def triangle_pooling(self, inner_product, zeroth_components):
        '''
        T(u,v) = (1 - <u, v>L - u0 - v0) / (u0 * v0)
               = (1 + u0 * v0 - inner_product - u0 - v0) / (u0 * v0)
               = 1 + (1 - inner_product - u0 - v0) / (u0 * v0)
        '''
        num_fields = zeroth_components.size(1)
        p, q = zip(*list(combinations(range(num_fields), 2)))
        u0, v0 = zeroth_components[:, p], zeroth_components[:, q]  # batch * (f(f-1)/2)
        score_tensor = 1 + torch.div(1 - inner_product - u0 - v0, u0 * v0) # batch * (f(f-1)/2)
        output = torch.sum(score_tensor, dim=1, keepdim=True) # batch * 1
        return output


def bpr_loss(pos, neg):
    """
    BPR Loss
    """
    loss = -torch.mean(torch.log(torch.nn.functional.sigmoid(pos - neg)))    
    return loss

def triangle_loss(pos, neg, it):
    """
    Trianglr loss
    """
    loss = -torch.mean(torch.log(torch.nn.functional.sigmoid(pos - (neg + it))))    
    return loss

def tetrahedron_loss(pos, neg, it, pos_emb, neg_emb, it_emb, a, b, c, d):
    """
    Tetrahedron loss
    """
    pos_zeros = get_zeroth_components(pos_emb)
    neg_zeros = get_zeroth_components(neg_emb)
    u_0 = pos_zeros[:,0]
    v_p = pos_zeros[:,1]
    v_n = neg_zeros[:,1]
    loss = torch.div(a * (1 + u_0 * v_p - pos  - v_n) - a * (1 + u_0 * v_n - neg - v_p) - c * (1 + v_p * v_n - it - v_p - v_n - u_0) + d, u_0 * v_p + u_0 * v_n + v_p * v_n)
    
    loss = -torch.mean(torch.log(torch.nn.functional.sigmoid(loss)))
    return loss
    
    
def get_zeroth_components(feature_emb):
        '''
        compute the 0th component
        '''
        sum_of_square = torch.sum(feature_emb ** 2, dim=-1) # batch * field
        zeroth_components = torch.sqrt(sum_of_square + 1) # beta = 1
        return zeroth_components # batch * field
    




    

