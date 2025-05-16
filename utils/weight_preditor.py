import os
import torch
import json

from .activations import ActivationModule

class STEFunction(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for the backward pass.
    The forward pass multiplies the input by a mask, and the backward pass
    simply passes the gradient through without modification.
    """
    @staticmethod
    def forward(ctx, input, mask):
        # Forward pass: apply the mask to the input
        return input * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # Return gradient for input and None for mask


class WeightPredictor(object):
    def __init__(self, model_name='Meta-Llama-3-8B', dtype=torch.float32, device=torch.device("cuda:0"), D=1024):
        self.model_name = model_name
        self.dtype = dtype
        self.device = device
        self.sparsity_strategy = 'Dynamic'
        
        self.num_layers = 0
        self.num_weights = 0
        self.weight_counters = []
        self.weight_maps = {}

        self.DO_CAL_ACTIVATIONS = False
        self.activations = ActivationModule()

        self.sparse_infer = 1
        self.sparse_params = [0,0]
        
        self.reset()

    def set_sparse_infer(self, s=1) :
        self.sparse_infer = s
    def is_sparse_infer(self) -> bool:
        if os.environ.get('ENABLE_SPARSE_INFER','1') == '0' :
            return False
        return self.sparse_infer
    def reset(self) :
        print('Init Reset')
        self.attn_sp = 0.0
        self.mlp_sp = 0.0
        self.w_p = 0.0
        self.threshold = [[0.0] * self.weight_counters[_] for _ in range(self.num_layers)]
        self.do_pre_prediction = 0

        # CROSS_LAYER_TEST
        self.preds = [[[] for _ in range(self.weight_counters[layer_id])] for layer_id in range(self.num_layers)]
        self.wmetrics = [[None for _ in range(self.weight_counters[layer_id])] for layer_id in range(self.num_layers)]
        self.similarity_results = [[] * self.weight_counters[_] for _ in range(self.num_layers)]

    def set_cal_activations(self, file_path) :
        print('set activations', self.num_layers,self.num_weights)
        self.DO_CAL_ACTIVATIONS = True
        self.activations.set_init_dict(self.num_layers, self.num_weights, file_path)
        
    def to_fp16(self):
        self.dtype = torch.float16

    def to_bf16(self):
        self.dtype = torch.bfloat16
        
    def set_sparsity_threshold(self, file_path=None) :
        print('set_sparsity_threshold')
        if file_path == None :
            file_path = os.environ.get('THRESHOLD_PATH',None)
        if file_path == None : 
            print('there is none this file.')
        #     file_path = f'./threshold/{self.model_name}/{self.model_name}-{self.attn_sp}.txt'
        print('threshold_path', file_path)
        self.threshold = [[0.0] * self.num_weights for _ in range(self.num_layers)] 
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                sparsity_all_dict = json.load(f)
                
            for i in range(self.num_layers):
                layer_key = f"{i}"
                if layer_key in sparsity_all_dict:
                    layer_thresholds = sparsity_all_dict[layer_key]
                    for j in range(self.num_weights) :
                        self.threshold[i][j] = layer_thresholds.get(f"{j}", 0.0)
                        
            self.sparsity_strategy = 'Static'
            # print(self.threshold)
            # print(self.threshold)
        else:
            self.sparsity_strategy = 'Dynamic'
        print('sparsity_strategy : ', self.sparsity_strategy)
         
    def score_to_mask(self, x, sp, thres=0.0, ilayer=-1):
        # choose threshold
        if self.sparsity_strategy == 'Dynamic':
            if len(x.shape) == 2:
                thres = x.sort(dim=-1).values[:, int(x.shape[-1] * sp)].view(x.shape[0], 1)
            elif len(x.shape) == 3:
                thres = x.sort(dim=-1).values[:, :, int(x.shape[-1] * sp)].view(x.shape[0], x.shape[1], 1)
            else:
                raise ValueError("Length of x shape must be 2 or 3")
        else:
            pass

        # all activation in layer 0
        r = os.environ.get('ACTIVATE_LAYER' , '0') 
        if ilayer >= 0 and ilayer <= int(r): 
            # print('YES')
            mask = x >= 0 
        else :
            mask =  x >= thres
        mask = mask.to(torch.int64)

        # compute sparse param C
        if mask.sum() > 0:  
            sum_all = x.sum()
            sum_masked = (x * mask).sum() 
            C = sum_all / sum_masked
        else:
            C = 1.0 

        return mask, C

    def predict_by_x_thres(self, ilayer, iweight, x):
        # print('predict' , ilayer , iweight)
        sp = self.attn_sp
        # Prediction.
        x = x.abs()
        threshold = self.threshold[ilayer][iweight]
        preds, C = self.score_to_mask(x, sp, threshold, ilayer)
       
        # predictor
        if self.do_pre_prediction:
            self.preds[ilayer][iweight] = preds
            if ilayer > 0:
                prev_preds = self.preds[ilayer - 1][iweight]
                if prev_preds is not None:
                    device = preds.device
                    prev_preds = prev_preds.to(device)
                    current_ones = preds.sum().item()
                    if current_ones > 0:
                        common_ones = (preds & prev_preds).sum().item()
                        similarity = common_ones / current_ones
                        # print(f'ilayer {ilayer} iweight {iweight} similarity {similarity}')
                        self.similarity_results[ilayer][iweight].append(similarity)
                    self.preds[ilayer - 1][iweight] = None # Clear
                else :
                    pass

        return preds, C

    def apply_pred(self, x, pred=None):
        return x if pred is None else x * pred.to(x.dtype).to(x.device)
    
    def generate_pred(self, ilayer, iweight, x) :
        if self.DO_CAL_ACTIVATIONS == True:
            self.activations.grab_activations(x, ilayer, iweight)
        if self.is_sparse_infer() == False:
            return x
        else :
            # print('grab ', ilayer, iweight, x.size())
            pred, C = self.predict_by_x_thres(ilayer, iweight, x)
            if os.environ.get('DEBUG_CROSSLAYER','0') != '0' :
                # 统计 pred 中零值的数量和总元素数量
                total_elements = pred.numel()
                zero_elements = (pred == 0).sum().item()

                # update self.sparse_params
                self.sparse_params[0] += total_elements
                self.sparse_params[1] += zero_elements

                # TODO : 计算pred中1值出现的协方差，记录到self.debug_pred[ilayer][iweight]里
                # if ilayer == 1 and iweight == 0:
                    # print(pred.size(),pred,x.size(),x)
                
                

                # zero_ratio = zero_elements / total_elements
                # print(f"Layer {ilayer}, Weight {iweight}: Zero ratio in pred = {zero_ratio:.4f}")

            # print(os.environ.get('BACKWARD_STRATEGY'))
            # return self.apply_pred(x, pred)
            if os.environ.get('BACKWARD_STRATEGY','0') != '0' :
                # print('yes')
                return self.apply_pred(x, pred)
            else : 
                return STEFunction.apply(x, pred)

    def set_sp_config(self, attn_sp, mlp_sp, w_p):
        self.attn_sp = attn_sp
        self.mlp_sp = mlp_sp
        self.w_p = w_p
        print(f"Set sparsity: attn {self.attn_sp}, mlp {self.mlp_sp}, w {self.w_p}")

    def set_do_pre_prediction(self, do_pre_prediction):
        self.do_pre_prediction = do_pre_prediction
        print(f"Set pre-prediction: {self.do_pre_prediction}")

    def set_sparsity_strategy(self, method: str) :
        self.sparsity_strategy = method
        print('sparsity_strategy: ', method)

    def set_layers_and_weights(self, num_layers , weight_counters, weight_map) :
        self.num_layers = num_layers
        self.num_weights = weight_counters[0]
        self.weight_counters = weight_counters
        self.weight_map = weight_map
        print('num_layers', self.num_layers)
        print('weight_counters', self.weight_counters)
        self.reset()

    def get_sparsity(self) :
        return self.attn_sp


def _init_weight_predictor(model_name=None):
    
    dtype = torch.float32
    local_rank = os.environ.get("LOCAL_RANK","-1")

    if local_rank != "-1":
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")
    
    print("Create and load preditor...")
    print("Local device:", device)
    # print("Checkpoint dir:", checkpoint_dir)
    weight_preditor = WeightPredictor(model_name, dtype=dtype, device=device,)
    weight_preditor.to_bf16()
    return weight_preditor
