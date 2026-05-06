import torch
import torch.distributed as dist
from torch.optim import Optimizer
from collections import defaultdict
import logging
from typing import Optional, Callable, Dict, List, Tuple, Literal, Set
import torch.distributed as dist
import random
import re
import numpy as np

logger = logging.getLogger(__name__)

def _normalize_target_modules(target_modules) -> Tuple[str, ...]:
    normalized: List[str] = []

    def _collect(value) -> None:
        if value is None:
            return
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped:
                normalized.append(stripped)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _collect(item)
            return
        stripped = str(value).strip().lower()
        if stripped:
            normalized.append(stripped)

    _collect(target_modules)
    return tuple(normalized)
    
class BlockAdamW(Optimizer):
    """
    Block-wise AdamW optimizer that operates directly on model parameters.
    All block identification and manipulation is done through parameter names.

    Args:
        params: Iterable of parameters or dicts defining parameter groups
        model: The transformer model to identify blocks
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for running averages (default: (0.9, 0.999))
        eps: Numerical stability term (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        block_switch_frequency: Steps between block switches (default: 100)
        block_strategy: 'fixed_size' | 'random' | 'importance' | 'entropy_corr'
        block_size: Target number of transformer layers per block (default: 2)
        block_sequence: 'ascending' | 'descending' | 'random' | 'importance'
        enable_distributed: Enable distributed aggregation (default: True)
        reset_optimizer_on_switch: Reset m and v to zero when switching blocks (default: True)
    """

    def __init__(
        self,
        params,
        model: torch.nn.Module,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        mu: Tuple[float, float] = (0.2, 0.333),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        block_switch_frequency: int = 100,
        block_strategy: Literal['fixed_size', 'random', 'importance', 'entropy_corr'] = 'fixed_size',
        block_size: int = 1,
        block_sequence: Literal['ascending', 'descending', 'random', 'importance'] = 'ascending',
        enable_distributed: bool = True,
        reset_optimizer_on_switch: bool = True,
        target_modules: Optional[List[str]] = None,
        use_tsm: bool = False,
        use_mm: bool = False,
        clip=False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, mu=mu, eps=eps, weight_decay=weight_decay)
        super(BlockAdamW, self).__init__(params, defaults)

        self.model = model
        self.block_switch_frequency = block_switch_frequency
        self.block_strategy = block_strategy
        self.block_size = block_size
        self.block_sequence = block_sequence
        self.reset_optimizer_on_switch = reset_optimizer_on_switch
        self.storage_dtype = torch.bfloat16
        self.compute_dtype = torch.float32
        self.target_modules = _normalize_target_modules(target_modules)
        self.use_tsm = use_tsm
        self.use_mm = use_mm
        self.clip = clip

        self.enable_distributed = enable_distributed and dist.is_initialized()
        self.rank = dist.get_rank() if self.enable_distributed else 0
        self.world_size = dist.get_world_size() if self.enable_distributed else 1

        # Block management - only store metadata, not parameter references
        self.blk_meta = self._get_blk_meta()
        self.total_blocks = len(self.blk_meta)
        if self.block_sequence == 'descending':
            self.cur_blk_idx = self.total_blocks - 1
        else:
            self.cur_blk_idx = 0
        self.global_step = 0
        self.block_step_counts = [0] * self.total_blocks
        self.blk_rcd = []

        self._init()

    # =======================================================================
    # Frozen parameter detection
    # =======================================================================

    @property
    def embedding_layer(self):
        for n, p in self.model.named_parameters():
            if "embed" in n.lower():
                return p
        return None

    @property
    def lm_head_layer(self):
        for n, p in self.model.named_parameters():
            if "lm_head" in n.lower():
                return p
        return None

    def _is_frozen(self, name: str, param: torch.nn.Parameter) -> bool:
        if self.embedding_layer is not None and param is self.embedding_layer:
            return True
        if self.lm_head_layer is not None and param is self.lm_head_layer:
            return True
        return "embed" in name.lower() or "lm_head" in name.lower()
        
    def _matches_target_module(self, name: str) -> bool:
        if not self.target_modules:
            return True
        lower_name = name.lower()
        if not lower_name.endswith(".weight"):
            return False
        return any(f".{module}." in lower_name for module in self.target_modules)

    def _is_trainable_param(self, name: str, param: torch.nn.Parameter) -> bool:
        return (not self._is_frozen(name, param)) and self._matches_target_module(name)

    @property
    def block_params(self) -> List[Tuple[str, torch.nn.Parameter]]:
        names = self.blk_meta[self.cur_blk_idx]['param_names']
        return [
            (name, param)
            for name, param in self.model.named_parameters()
            if name in names and param.requires_grad
        ]

    def _init(self):
        for name, param in self.model.named_parameters():
            if self._is_frozen(name, param):
                param.requires_grad = False
        for block_idx in range(self.total_blocks):
            is_active = (block_idx == self.cur_blk_idx)
            self._set_blk_rg(block_idx, is_active)
        for _, param in self.block_params:
            self._pm_vert(param, self.compute_dtype)


    # =======================================================================
    # Block identification and grouping
    # =======================================================================

    def _dect_ly_ptn(self) -> Tuple[Optional[str], int]:
        """
        Detect the layer indexing pattern used by the model.
        Common patterns:
            - model.layers.0, model.layers.1, ...
            - transformer.h.0, transformer.h.1, ...
            - model.decoder.layers.0, ...
        """
        patterns = [
            r'model\.layers\.(\d+)',
            r'transformer\.h\.(\d+)',
            r'model\.decoder\.layers\.(\d+)',
            r'decoder\.layers\.(\d+)',
            r'encoder\.layers\.(\d+)',
            r'layers\.(\d+)',
        ]
        layer_indices = {}; matched_pattern = None
        for name, param in self.model.named_parameters():
            if not self._is_trainable_param(name, param):
                continue
            for pattern in patterns:
                match = re.search(pattern, name)
                if match:
                    layer_idx = int(match.group(1))
                    if matched_pattern is None:
                        matched_pattern = pattern
                    if matched_pattern == pattern:
                        layer_indices[layer_idx] = True
        if matched_pattern and layer_indices:
            num_layers = max(layer_indices.keys()) + 1
            return matched_pattern, num_layers
        return None, 0

    def _get_blk_meta(self) -> List[Dict]:
        """
        Compute block metadata. Returns list of dicts with block information
        (without storing parameter references, only names).
        """
        pattern, num_layers = self._dect_ly_ptn()
        if pattern is None or num_layers == 0:
            param_names = set()
            param_count = 0
            for name, param in self.model.named_parameters():
                if self._is_trainable_param(name, param):
                    param_names.add(name)
                    param_count += param.numel()
            return [{
                'block_id': 0,
                'layer_indices': [],
                'param_count': param_count,
                'param_names': param_names
            }]

        layer_info = {li: {'param_names': set(), 'param_count': 0} for li in range(num_layers)}
        for name, param in self.model.named_parameters():
            if not self._is_trainable_param(name, param):
                continue
            m = re.search(pattern, name)
            if m is not None:
                li = int(m.group(1))
                if li in layer_info:
                    layer_info[li]['param_names'].add(name)
                    layer_info[li]['param_count'] += param.numel()

        if self.block_strategy == 'random':
            return self._gp_rd(layer_info, num_layers)
        elif self.block_strategy == 'importance':
            return self._gp_ipt(layer_info, num_layers)
        else:
            return self._gp_fix(layer_info, num_layers)


    def _gp_fix(self, layer_info: Dict, num_layers: int) -> List[Dict]:
        """Fixed-size uniform block partitioning."""
        blocks = []
        for i in range(0, num_layers, self.block_size):
            end = min(i + self.block_size, num_layers)
            layer_indices = list(range(i, end))
            param_names: Set[str] = set()
            param_count = 0
            for li in layer_indices:
                if li in layer_info:
                    param_names.update(layer_info[li]['param_names'])
                    param_count += layer_info[li]['param_count']
            if param_names:
                blocks.append({
                    'block_id': len(blocks),
                    'layer_indices': layer_indices,
                    'param_count': param_count,
                    'param_names': param_names
                })
        return blocks

    def _gp_rd(self, layer_info: Dict, num_layers: int) -> List[Dict]:
        """Random block partitioning (shuffled layer indices)."""
        indices = list(range(num_layers))
        random.shuffle(indices)
        blocks = []
        for i in range(0, len(indices), self.block_size):
            layer_indices = sorted(indices[i:i + self.block_size])
            param_names: Set[str] = set()
            param_count = 0
            for li in layer_indices:
                if li in layer_info:
                    param_names.update(layer_info[li]['param_names'])
                    param_count += layer_info[li]['param_count']
            if param_names:
                blocks.append({
                    'block_id': len(blocks),
                    'layer_indices': layer_indices,
                    'param_count': param_count,
                    'param_names': param_names
                })
        return blocks

    def _gp_ipt(self, layer_info: Dict, num_layers: int) -> List[Dict]:
        """
        Importance-based block partitioning.
        Layers are sorted by total parameter L2-norm (proxy for importance),
        then grouped so the most important layers share a block and receive
        focused training together.  Layer indices within each block are kept
        in ascending order so the block still spans a contiguous sub-network.
        """
        # Per-layer importance: sum of L2 norms of all weight tensors
        importance: Dict[int, float] = {}
        for li in range(num_layers):
            if li not in layer_info or not layer_info[li]['param_names']:
                importance[li] = 0.0
                continue
            total_norm = 0.0
            for pname, param in self.model.named_parameters():
                if pname in layer_info[li]['param_names']:
                    total_norm += param.data.float().norm(2).item()
            importance[li] = total_norm

        # Sort by descending importance
        sorted_layers = sorted(range(num_layers),
                               key=lambda i: importance.get(i, 0.0), reverse=True)

        blocks: List[Dict] = []
        for i in range(0, len(sorted_layers), self.block_size):
            layer_indices = sorted(sorted_layers[i:i + self.block_size])
            param_names: Set[str] = set()
            param_count = 0
            for li in layer_indices:
                if li in layer_info:
                    param_names.update(layer_info[li]['param_names'])
                    param_count += layer_info[li]['param_count']
            if param_names:
                blocks.append({
                    'block_id': len(blocks),
                    'layer_indices': layer_indices,
                    'param_count': param_count,
                    'param_names': param_names,
                    'importance': sum(importance.get(li, 0.0) for li in layer_indices),
                })
        return blocks


    # =======================================================================
    # State management
    # =======================================================================

    def _set_blk_rg(self, block_idx: int, requires_grad: bool):
        block_param_names = self.blk_meta[block_idx]['param_names']
        for name, param in self.model.named_parameters():
            if name in block_param_names:
                param.requires_grad = requires_grad

    def _pm_vert(self, param: torch.nn.Parameter, dtype: torch.dtype):
        if param.data.dtype != dtype:
            param.data = param.data.to(dtype)

    def _ensure_state(self, param: torch.nn.Parameter):
        state = self.state[param]
        if 'step' not in state:
            state['step'] = 0
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(param.data, dtype=self.compute_dtype)
        if 'exp_avg_sq' not in state:
            state['exp_avg_sq'] = torch.zeros_like(param.data, dtype=self.compute_dtype)
        if 'shadow_param' not in state:
            state['shadow_param'] = param.data.detach().clone().to(self.compute_dtype)

    def _reset_adam_stt(self, param_names: Set[str]):
        for name, param in self.model.named_parameters():
            if name in param_names and param in self.state:
                self.state.pop(param, None)
            elif name in param_names and param not in self.state:
                self.state[param] = {
                    'step': 0,
                    'exp_avg': torch.zeros_like(param.data, dtype=self.compute_dtype),
                    'exp_avg_sq': torch.zeros_like(param.data, dtype=self.compute_dtype),
                    'shadow_param': param.data.detach().clone().to(self.compute_dtype),
                }

    def _init_adam_stt(self):
        """Initialise Adam state for the current active block (used at step 0)."""
        param_names = self.blk_meta[self.cur_blk_idx]['param_names']
        self._reset_adam_stt(param_names)

    def _fix_shadow(self):
        """
        Update shadow_param to match current parameter values (post-aggregation
        snapshot).  Called by the agent after each block-parameter exchange so
        that _qg_update can measure the delta introduced by aggregation.
        """
        for _, param in self.block_params:
            if param in self.state and 'shadow_param' in self.state[param]:
                # Ensure dtype consistency
                self.state[param]['shadow_param'].copy_(
                    param.data.to(self.compute_dtype)
                )

    def _blk_switch(self, bi: List[float] = None):
        """
        Freeze the current block, advance to the next block, and activate it.
        Optionally resets Adam moments (controlled by reset_optimizer_on_switch).
        """
        # 1. Convert current block to storage precision
        for _, param in self.block_params:
            self._pm_vert(param, self.storage_dtype)

        # 2. Optionally clear Adam state for the old block
        if self.reset_optimizer_on_switch:
            self._reset_adam_stt(self.blk_meta[self.cur_blk_idx]['param_names'])

        # 3. Freeze the old block
        self._set_blk_rg(self.cur_blk_idx, False)

        self.blk_rcd.append(self.cur_blk_idx)
        print(f"The blocks chosen up to now are: {self.blk_rcd}")

        # 4. Select next block index
        if self.block_sequence == 'ascending':
            next_block_idx = (self.cur_blk_idx + 1) % self.total_blocks
        elif self.block_sequence == 'descending':
            next_block_idx = (self.cur_blk_idx - 1 + self.total_blocks) % self.total_blocks
        elif self.block_sequence == 'random':
            next_block_idx = random.randint(0, self.total_blocks - 1)
        else:
            raise ValueError(f"Unknown block sequence: {self.block_sequence}")

        self.cur_blk_idx = next_block_idx

        # 5. Initialise / reset Adam state for the new block
        if self.reset_optimizer_on_switch:
            self._reset_adam_stt(self.blk_meta[self.cur_blk_idx]['param_names'])

        # 6. Unfreeze the new block and promote to compute precision
        self._set_blk_rg(self.cur_blk_idx, True)
        for _, param in self.block_params:
            self._pm_vert(param, self.compute_dtype)

        self.block_step_counts[self.cur_blk_idx] = 0

    @torch.no_grad()
    def _get_comm_param(self, with_error_feedback: bool = False) -> List[Tuple[str, torch.Tensor]]:
        """
        Return per-parameter delta tensors for communication:
            u = (θ_curr - θ_shadow) [+ comm_error if with_error_feedback]
        """
        comm_param: List[Tuple[str, torch.Tensor]] = []
        for name, param in self.block_params:
            shadow = self.state[param]['shadow_param']
            delta = param.data.to(self.compute_dtype) - shadow
            if with_error_feedback and 'comm_error' in self.state[param]:
                delta = delta + self.state[param]['comm_error']
            comm_param.append((name, delta.clone()))
        return comm_param

    def _agg_blk_params(
        self, 
        received_params: Dict[int, List[torch.Tensor]],
        neighbor, neighbor_rank_ns,
        diff: bool = False
    ):
        for i, (_, param) in enumerate(self.block_params):
            param.data.mul_(neighbor[self.rank])
            if diff:
                param.data.add_(self.state[param]['shadow_param'], alpha=1.0-neighbor[self.rank])
            for src_id in neighbor_rank_ns:
                param.data.add_(received_params[src_id][i], alpha=neighbor[src_id])

            
    # =======================================================================
    # Adam update
    # =======================================================================

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Standard Optimizer.step() entry point."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.global_step == 0 and self.reset_optimizer_on_switch:
            self._init_adam_stt()
        if self.use_tsm:
            self._update_current_block_tsm()
        elif self.use_mm:
            self._update_current_block_momentum()
        else:
            self._update_current_block()
        self.global_step += 1
        return loss

    @torch.no_grad()
    def _update_current_block(self):
        """
        AdamW update for the currently active block.

        Computes (in-place, using _foreach batch ops for efficiency):
            1. Decoupled weight decay:  θ ← θ · (1 - lr·λ)
            2. EMA first moment:        m ← β1·m + (1-β1)·g
            3. EMA second moment:       v ← β2·v + (1-β2)·g²
            4. Bias-corrected update:   θ ← θ - lr·m̂ / (√v̂ + ε)

        Sign-trick explanation
        ----------------------
        We compute step_size = lr / (β1^t - 1)  which is NEGATIVE (since β1^t < 1).
        Then denom = (√(v/bc2) + ε) / step_size  which is also NEGATIVE.
        So  addcdiv(θ, m, denom)  =  θ + m/denom  = θ - lr/(1-β1^t) · m̂/(√v̂+ε)  ✓
        Both signs cancel; the math is equivalent to the standard positive formula.
        """
        group = self.param_groups[0]
        lr = group['lr']
        beta1, beta2 = group['betas']
        eps = group['eps']
        weight_decay = group['weight_decay']

        # Collect active params that actually have a gradient
        active = [(name, p) for name, p in self.block_params
                  if p.grad is not None]
        if not active:
            return

        params = [p for _, p in active]
        grads  = [p.grad for p in params]

        # Ensure Adam state exists for every active parameter
        for p in params:
            self._ensure_state(p)

        self.block_step_counts[self.cur_blk_idx] += 1
        step = self.block_step_counts[self.cur_blk_idx]

        exp_avgs    = [self.state[p]['exp_avg']    for p in params]
        exp_avg_sqs = [self.state[p]['exp_avg_sq'] for p in params]
        device = params[0].device
        dtype  = params[0].dtype

        # ---- Decoupled weight decay ----
        if weight_decay != 0.0:
            torch._foreach_mul_(params, 1.0 - lr * weight_decay)

        # ---- EMA first moment:  m ← β1·m + (1-β1)·g ----
        torch._foreach_lerp_(exp_avgs, grads, 1.0 - beta1)

        # ---- EMA second moment:  v ← β2·v + (1-β2)·g² ----
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1.0 - beta2)

        # ---- Bias correction via sign trick ----
        # Create per-param tensors for foreach_pow (avoids in-place issues)
        bc1_list = [torch.tensor(beta1, device=device, dtype=dtype)] * len(params)
        bc2_list = [torch.tensor(beta2, device=device, dtype=dtype)] * len(params)
        bias_correction1 = torch._foreach_pow(bc1_list, step)   # β1^t
        bias_correction2 = torch._foreach_pow(bc2_list, step)   # β2^t

        # bias_correction1: β1^t - 1  (negative, NOT negated yet)
        torch._foreach_sub_(bias_correction1, 1.0)
        # bias_correction2: 1 - β2^t  (positive)
        torch._foreach_sub_(bias_correction2, 1.0)
        torch._foreach_neg_(bias_correction2)

        # step_size = lr / (β1^t - 1)  → negative
        torch._foreach_div_(bias_correction1, lr)
        torch._foreach_reciprocal_(bias_correction1)
        step_size = bias_correction1                             # negative

        # bc2_sqrt = √(1 - β2^t)
        torch._foreach_sqrt_(bias_correction2)
        bc2_sqrt = bias_correction2                              # positive

        # denom = (√(v_t) / √(1-β2^t) + ε) / step_size
        #       = (√v̂_t + ε) / step_size   → negative (since step_size < 0)
        exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(exp_avg_sq_sqrt, bc2_sqrt)
        torch._foreach_add_(exp_avg_sq_sqrt, eps)
        torch._foreach_div_(exp_avg_sq_sqrt, step_size)

        # θ += m / denom  =  θ - lr/(1-β1^t) · m̂/(√v̂+ε)   ✓
        torch._foreach_addcdiv_(params, exp_avgs, exp_avg_sq_sqrt)

    @torch.no_grad()
    def _qg_update(self):
        """
        Quasi-gradient (QG) update: use the aggregation-induced parameter
        delta  Δθ = θ_agg - θ_prev  as a synthetic gradient to warm-start
        the Adam moments after decentralised parameter aggregation.

        This lets the momentum buffers reflect the "direction of improvement"
        introduced by neighbours, reducing the cold-start instability that
        occurs when moments are reset to zero on each block switch.
        
        """
        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        mu1, mu2 = group["mu"]

        active = [(name, p) for name, p in self.block_params
                  if p in self.state and 'shadow_param' in self.state[p]]
        if not active:
            return
        params = [p for _, p in active]
        exp_avgs    = [self.state[p]['exp_avg']    for p in params]
        exp_avg_sqs = [self.state[p]['exp_avg_sq'] for p in params]

        # Δθ =  θ_prev - θ_agg
        deltas = [self.state[p]['shadow_param'] - p.data.to(self.compute_dtype) for p in params]
        # Normalise the quasi-gradient vector
        norm = torch.cat([d.reshape(-1) for d in deltas]).norm(p=2).clamp_min(1e-6)
        if self.clip:
            norm = torch.maximum(norm, torch.tensor(0.5, device=norm.device, dtype=norm.dtype))
        torch._foreach_div_(deltas, norm)
        
        # m ← β1·m + (1-β1)·Δθ̂
        torch._foreach_lerp_(exp_avgs, deltas, 1.0 - mu1)
        # v ← β2·v + (1-β2)·Δθ̂²
        torch._foreach_mul_(exp_avg_sqs, mu2)
        torch._foreach_addcmul_(exp_avg_sqs, deltas, deltas, 1.0 - mu2)
        

    @torch.no_grad()
    def _update_current_block_tsm(self):
        group = self.param_groups[0]
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        active = [(name, p) for name, p in self.block_params if p.grad is not None]
        if not active:
            return

        params = [p for _, p in active]
        for p in params:
            self._ensure_state(p)
        self.block_step_counts[self.cur_blk_idx] += 1
        step = self.block_step_counts[self.cur_blk_idx]
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        step_size = -lr / bias_correction1
        for p in params:
            grad = p.grad
            if grad is None:
                continue
            state = self.state[p]
            exp_avg = state["exp_avg"]          # 旧 m
            exp_avg_sq = state["exp_avg_sq"]    # 旧 v
            # ---- Decoupled weight decay ----
            if weight_decay != 0.0:
                p.data.mul_(1.0 - lr * weight_decay)
            # 为了和状态 dtype 对齐，必要时仅对当前参数做一次临时转换
            g = grad.detach()
            if g.dtype != exp_avg.dtype:
                g_work = g.to(dtype=exp_avg.dtype)
            else:
                g_work = g

            # denom = sqrt((beta2 * v + (1-beta2) * g^2) / (1-beta2^t)) + eps
            denom = torch.empty_like(exp_avg_sq)
            denom.copy_(g_work)
            denom.mul_(g_work)                                   # g^2
            denom.mul_(1.0 - beta2)
            denom.add_(exp_avg_sq, alpha=beta2)                  # beta2*v + (1-beta2)*g^2
            denom.div_(bias_correction2)
            denom.sqrt_()
            denom.add_(eps)

            # num = beta1 * m + (1-beta1) * g
            # 直接复用 g_work 作为当前参数级 scratch，避免单独的 m_half
            g_work.mul_(1.0 - beta1)
            g_work.add_(exp_avg, alpha=beta1)
            # x <- x - lr * num / ((1-beta1^t) * denom)
            #   = x + step_size * num / denom, where step_size = -lr / (1-beta1^t)
            if p.data.dtype == g_work.dtype:
                p.data.addcdiv_(g_work, denom, value=step_size)
            else:
                # 如果参数 dtype 和状态 dtype 不同，就只对当前参数做一次临时更新后拷回
                p_fp = p.data.to(dtype=g_work.dtype)
                p_fp.addcdiv_(g_work, denom, value=step_size)
                p.data.copy_(p_fp.to(dtype=p.data.dtype))


    @torch.no_grad()
    def _qg_update_tsm(self):
        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        mu1, mu2 = group["mu"]

        active = [(name, p) for name, p in self.block_params
                  if p in self.state and 'shadow_param' in self.state[p]]
        if not active:
            return
        params = [p for _, p in active]
        exp_avgs    = [self.state[p]['exp_avg']    for p in params]
        exp_avg_sqs = [self.state[p]['exp_avg_sq'] for p in params]

        # Δθ =  θ_prev - θ_agg
        deltas = [self.state[p]['shadow_param'] - p.data.to(self.compute_dtype) for p in params]
        # Normalise the quasi-gradient vector
        norm = torch.cat([d.reshape(-1) for d in deltas]).norm(p=2).clamp_min(1e-6)
        norm = torch.maximum(norm, torch.tensor(1.0, device=norm.device, dtype=norm.dtype))
        torch._foreach_div_(deltas, norm)
        
        # m ← β1·m + (1-β1)·Δθ̂
        torch._foreach_lerp_(exp_avgs, deltas, 1.0 - beta1)
        # v ← β2·v + (1-β2)·Δθ̂²
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, deltas, deltas, 1.0 - beta2)

    @torch.no_grad()
    def _update_current_block_momentum(self):
        """
        Momentum update for the currently active block.
        Computes:
            1. Decoupled weight decay: θ ← θ · (1 - lr·λ)
            2. First momentum:         m ← β1·m + g
            3. Parameter update:       θ ← θ - lr·m
        Here β1 is used as the standard momentum coefficient.
        No second moment, no bias correction, no Adam denominator.
        """
        group = self.param_groups[0]
        lr = group["lr"]
        beta1 = group["betas"][0]
        weight_decay = group["weight_decay"]
        active = [(name, p) for name, p in self.block_params
                  if p.grad is not None]
        if not active:
            return
    
        params = [p for _, p in active]
        grads = [p.grad for p in params]
        # Ensure momentum state exists for every active parameter.
        for p in params:
            self._ensure_state(p)
        self.block_step_counts[self.cur_blk_idx] += 1
        exp_avgs = [self.state[p]["exp_avg"] for p in params]
    
        # ---- Decoupled weight decay ----
        if weight_decay != 0.0:
            torch._foreach_mul_(params, 1.0 - lr * weight_decay)
        # ---- Momentum: m ← β1·m + g ----
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads)
        # ---- Parameter update: θ ← θ - lr·m ----
        torch._foreach_add_(params, exp_avgs, alpha=-lr)
    
    
    @torch.no_grad()
    def _qg_update_momentum(self):
        """
        Quasi-gradient update for ordinary Momentum.
        Only updates the first-order momentum buffer:
            Δθ = θ_prev - θ_agg
            m ← μ1·m + Δθ_hat
        No second moment is used.
        """
        group = self.param_groups[0]
        beta1 = group["betas"][0]
        mu = group.get("mu", None)
        mu1 = mu[0] if mu is not None else beta1
        active = [(name, p) for name, p in self.block_params
                  if p in self.state and "shadow_param" in self.state[p]]
        if not active:
            return
    
        params = [p for _, p in active]
        exp_avgs = [self.state[p]["exp_avg"] for p in params]
        # Δθ = θ_prev - θ_agg
        deltas = [
            self.state[p]["shadow_param"] - p.data.to(self.compute_dtype)
            for p in params
        ]
        # m ← μ1·m + (1-μ1)·Δθ
        torch._foreach_mul_(exp_avgs, mu1)
        torch._foreach_add_(exp_avgs, deltas, alpha=1.0-mu1)
    
    # =======================================================================
    # Checkpointing
    # =======================================================================

    def state_dict(self):
        sd = super().state_dict()
        sd['blk_meta'] = self.blk_meta
        sd['cur_blk_idx'] = self.cur_blk_idx
        sd['global_step'] = self.global_step
        sd['block_step_counts'] = self.block_step_counts
        return sd

    def load_state_dict(self, state_dict):
        self.blk_meta = state_dict.pop('blk_meta', self.blk_meta)
        self.cur_blk_idx = state_dict.pop('cur_blk_idx', 0)
        self.global_step = state_dict.pop('global_step', 0)
        self.block_step_counts = state_dict.pop(
            'block_step_counts', [0] * self.total_blocks)
        super().load_state_dict(state_dict)
        # Restore active block state after loading
        self._set_blk_rg(self.cur_blk_idx, True)
        for _, param in self.block_params:
            self._pm_vert(param, self.compute_dtype)


# ---------------------------------------------------------------------------
# Block-MEZO-optimizer, using ZOO
# ---------------------------------------------------------------------------

class MeZOBlockAdamW(Optimizer):
    """
    Block-wise AdamW optimizer whose gradients are estimated via the MeZO
    zero-order method instead of backpropagation.

    Args:
        params:                 Iterable of parameters or param groups.
        model:                  The transformer model.
        lr:                     Learning rate (default 1e-3).
        betas:                  AdamW beta coefficients (default (0.9, 0.999)).
        eps:                    AdamW epsilon (default 1e-8).
        weight_decay:           AdamW weight decay (default 0.01).
        zo_eps:                 MeZO finite-difference step size (default 1e-3).
        candidate_seeds:        Pool of integer seeds for z-sampling.
                                If None, uses range(10000).
        grad_clip:              If > 0, skip update when |loss+ - loss-| exceeds
                                this value (MeZO gradient clipping, default 1.0).
        block_switch_frequency: Steps between block switches (default 100).
        block_strategy:         'fixed_size' | 'random' | 'importance'
                                (default 'fixed_size').
        block_size:             Number of transformer layers per block
                                (default 2).
        block_sequence:         'ascending' | 'descending' | 'random'
                                (default 'ascending').
        enable_distributed:     Aggregate across ranks when dist is initialized
                                (default True).
        reset_optimizer_on_switch:
                                Zero m/v when switching blocks (default True).
        num_zo_steps:           MeZO perturbation steps per optimizer.step()
                                call (default 1). Increasing this averages
                                multiple gradient estimates before the Adam
                                update, which reduces variance at extra cost.
    """

    def __init__(
        self,
        params,
        model: torch.nn.Module,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        zo_eps: float = 3e-4,
        candidate_seeds: Optional[List[int]] = None,
        grad_clip: float = 1.0,
        block_switch_frequency: int = 100,
        block_strategy: Literal["fixed_size", "random", "importance"] = "fixed_size",
        block_size: int = 2,
        block_sequence: Literal["ascending", "descending", "random"] = "ascending",
        enable_distributed: bool = True,
        reset_optimizer_on_switch: bool = True,
        num_zo_steps: int = 1,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if zo_eps <= 0.0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # ---- model & core settings ----------------------------------------
        self.model = model
        self.zo_eps = zo_eps
        self.candidate_seeds = candidate_seeds if candidate_seeds is not None else list(range(10_000))
        self.grad_clip = grad_clip
        self.num_zo_steps = num_zo_steps

        # ---- block settings -----------------------------------------------
        self.block_switch_frequency = block_switch_frequency
        self.block_strategy = block_strategy
        self.block_size = block_size
        self.block_sequence = block_sequence
        self.reset_optimizer_on_switch = reset_optimizer_on_switch
        self.storage_dtype = torch.bfloat16
        self.compute_dtype = torch.float32

        # ---- distributed --------------------------------------------------
        self.enable_distributed = enable_distributed and dist.is_initialized()
        self.rank = dist.get_rank() if self.enable_distributed else 0
        self.world_size = dist.get_world_size() if self.enable_distributed else 1

        # ---- block management state ---------------------------------------
        self.blk_meta: List[Dict] = self._compute_blk_meta()
        self.total_blocks: int = len(self.blk_meta)
        self.cur_blk_idx: int = 0
        self.global_step: int = 0
        self.block_step_counts: List[int] = [0] * self.total_blocks
        self.prev_block_params: Dict = {}
        self.aggregated_params_buffer: Dict = {}

        # ---- MeZO working state -------------------------------------------
        self._zo_random_seed: Optional[int] = None
        self._projected_grad: float = 0.0

        # ---- init gradient flow / dtypes ----------------------------------
        self._initialize_training()

    # =======================================================================
    # Main API
    # =======================================================================

    def step(
        self,
        batch: Dict,
        local_seed_pool: Optional[Dict[int, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one full MeZO-BlockAdamW update step.
        Replaces the conventional ``loss.backward(); optimizer.step()`` pair.
        No computational graph is built.
        Args:
            batch:            Dict forwarded to ``model(**batch)``.
            local_seed_pool:  Optional dict accumulating projected gradients
                              per seed (for FedKSeed / gossip protocols).
        Returns:
            (logits, loss) from the first forward pass (f(θ + ε·z)).
        """
        # Accumulate projected-grad estimates over num_zo_steps
        logits, loss, proj_grad = self._single_zo_estimate(batch)
        # Apply AdamW update using averaged projected gradient
        if not (loss is None or (isinstance(loss, float) and loss == 0.0)):
            self._zo_adamw_update(proj_grad, local_seed_pool)
        # Block switching
        if self.global_step > 0 and (self.global_step + 1) % self.block_switch_frequency == 0:
            self._switch_to_next_block()
        self.global_step += 1
        return logits, loss


    # =======================================================================
    # MeZO core
    # =======================================================================

    def _single_zo_estimate(
        self, batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool]:
        """
        One two-point MeZO gradient estimate on the active block.
        Returns:
            logits, loss, projected_grad, skipped
        """
        self._zo_random_seed = int(np.random.choice(self.candidate_seeds, 1)[0])

        # f(θ + ε·z)
        self._zo_perturb_block(scaling_factor=1)
        logits1, loss1 = self._zo_forward(batch)
        # f(θ - ε·z)
        self._zo_perturb_block(scaling_factor=-2)
        logits2, loss2 = self._zo_forward(batch)
        # Restore θ
        self._zo_perturb_block(scaling_factor=1)

        # NaN guard
        if torch.is_tensor(loss1) and torch.isnan(loss1):
            return logits1, loss1, 0.0
        if torch.is_tensor(loss2) and torch.isnan(loss2):
            return logits2, loss2, 0.0
        # Gradient clipping (MeZO-style)
        l1 = loss1.item() if torch.is_tensor(loss1) else float(loss1)
        l2 = loss2.item() if torch.is_tensor(loss2) else float(loss2)
        # if self.grad_clip > 0.0 and abs(l1 - l2) > self.grad_clip:
        #     return logits1, 0.0, 0.0
        proj_grad = (l1 - l2) / (2.0 * self.zo_eps)
        return logits1, loss1, proj_grad

    def _zo_perturb_block(self, scaling_factor: float = 1.0):
        """
        Perturb only the parameters in the currently active block:
            θ_i ← θ_i + scaling_factor · ε · z_i
        z_i is deterministically derived from self._zo_random_seed.
        """
        torch.manual_seed(self._zo_random_seed)
        for _, param in self._get_block_params_tuple:
            z = torch.normal(
                mean=0, std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            param.data.add_(z, alpha=scaling_factor * self.zo_eps)
            del z

    def _zo_forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with no gradient tracking."""
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.compute_dtype):
                outputs = self.model(**batch)
                logits = outputs.logits.detach()
                loss = outputs.loss.detach()
        return logits, loss

    def _zo_adamw_update(
        self,
        projected_grad: float,
        local_seed_pool: Optional[Dict[int, float]] = None,
    ):
        """
        Expand projected_grad back to per-parameter synthetic gradients and
        apply an AdamW update on the active block.
        The synthetic gradient for parameter i is:
            g_i = projected_grad · z_i
        where z_i is the same perturbation vector used during estimation.
        """
        group = self.param_groups[0]
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps_adam = group["eps"]
        weight_decay = group["weight_decay"]
        self.block_step_counts[self.cur_blk_idx] += 1
        step = self.block_step_counts[self.cur_blk_idx]
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        step_size = lr / bc1
        bc2_sqrt = bc2 ** 0.5

        torch.manual_seed(self._zo_random_seed)

        for name, param in self._get_block_params_tuple:
            # Re-derive the same z used during perturbation
            z = torch.normal(
                mean=0, std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            # Synthetic gradient
            g = projected_grad * z
            # Lazy init optimizer state
            if param not in self.state:
                self.state[param] = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(param.data, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(param.data, dtype=torch.float32),
                }
            state = self.state[param]
            state["step"] = step
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # Weight decay (decoupled, AdamW style)
            is_bias_or_norm = (
                "bias" in name
                or "layer_norm" in name
                or "layernorm" in name
            )
            if weight_decay != 0.0 and not is_bias_or_norm:
                param.data.mul_(1.0 - lr * weight_decay)

            # m_t = β1·m_{t-1} + (1-β1)·g
            exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
            # v_t = β2·v_{t-1} + (1-β2)·g²
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            # θ ← θ - α · m̂ / (√v̂ + ε)
            denom = (exp_avg_sq.sqrt() / bc2_sqrt).add_(eps_adam)
            param.data.addcdiv_(exp_avg, denom, value=-step_size)

        # Accumulate projected grad in seed pool (FedKSeed protocol)
        if local_seed_pool is not None:
            local_seed_pool[self._zo_random_seed] = (
                local_seed_pool.get(self._zo_random_seed, 0.0) + projected_grad
            )

    @torch.no_grad()
    def _qg_update(self):
        """
        Quasi-gradient update: use Δθ = θ_agg - θ_prev as a synthetic gradient
        to warm-start the Adam moments after parameter aggregation.
        """
        prev_params = self.prev_block_params.get(self.cur_blk_idx, {})
        agg_params = self._get_block_params_dict
        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        for name, param in self.model.named_parameters():
            if name not in prev_params or name not in agg_params:
                continue
            if param not in self.state:
                continue
            state = self.state[param]
            delta = agg_params[name].sub(prev_params[name])
            state["exp_avg"].mul_(beta1).add_(delta, alpha=1 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(delta, delta, value=1 - beta2)

    def _set_aggregated_block_params(
        self,
        received_params: Dict[int, List[torch.Tensor]],
        neighbor: Dict,
        neighbor_rank: List,
    ):
        for i, (_, param) in enumerate(self._get_block_params_tuple):
            param.data.mul_(neighbor[self.rank])
            for src_id in neighbor_rank:
                param.data.add_(received_params[src_id][i], alpha=neighbor[src_id])
            
    # =======================================================================
    # Block management  (unchanged from BlockAdamW)
    # =======================================================================

    @property
    def embedding_layer(self):
        for n, p in self.model.named_parameters():
            if "embed" in n.lower():
                return p
        return None

    @property
    def lm_head_layer(self):
        for n, p in self.model.named_parameters():
            if "lm_head" in n.lower():
                return p
        return None

    def _is_frozen_param(self, name: str, param: torch.nn.Parameter) -> bool:
        if self.embedding_layer is not None and param is self.embedding_layer:
            return True
        if self.lm_head_layer is not None and param is self.lm_head_layer:
            return True
        return "embed" in name.lower() or "lm_head" in name.lower()

    def _get_layer_index(self, param_name: str, pattern: str) -> Optional[int]:
        match = re.search(pattern, param_name)
        return int(match.group(1)) if match else None

    @property
    def _get_block_params_tuple(self) -> List[Tuple[str, torch.nn.Parameter]]:
        names = self.blk_meta[self.cur_blk_idx]["param_names"]
        return [(n, p) for n, p in self.model.named_parameters() if n in names and p.requires_grad]

    @property
    def _get_block_params_list(self) -> List[torch.Tensor]:
        names = self.blk_meta[self.cur_blk_idx]["param_names"]
        return [p for n, p in self.model.named_parameters() if n in names and p.requires_grad]

    @property
    def _get_block_params_dict(self) -> Dict[str, torch.Tensor]:
        names = self.blk_meta[self.cur_blk_idx]["param_names"]
        return {n: p for n, p in self.model.named_parameters() if n in names and p.requires_grad}

    def _convert_param_dtype(self, param: torch.nn.Parameter, dtype: torch.dtype):
        if param.data.dtype != dtype:
            param.data = param.data.to(dtype)

    def _initialize_training(self):
        for name, param in self.model.named_parameters():
            if self._is_frozen_param(name, param):
                param.requires_grad = False
        self._initialize_block_requires_grad()
        for param in self._get_block_params_list:
            self._convert_param_dtype(param, self.compute_dtype)

    def _initialize_block_requires_grad(self):
        for block_idx in range(self.total_blocks):
            self._set_block_requires_grad(block_idx, block_idx == self.cur_blk_idx)

    def _set_block_requires_grad(self, block_idx: int, requires_grad: bool):
        names = self.blk_meta[block_idx]["param_names"]
        for name, param in self.model.named_parameters():
            if name in names:
                param.requires_grad = requires_grad

    def _reset_optimizer_states(self, param_names: Set[str]):
        for name, param in self.model.named_parameters():
            if name in param_names and param in self.state:
                s = self.state[param]
                if "exp_avg" in s:
                    s["exp_avg"].zero_()
                if "exp_avg_sq" in s:
                    s["exp_avg_sq"].zero_()
                if "step" in s:
                    s["step"] = 0

    def _switch_to_next_block(self):
        # Freeze & downcast current block
        self._set_block_requires_grad(self.cur_blk_idx, False)
        for param in self._get_block_params_list:
            self._convert_param_dtype(param, self.storage_dtype)
        if self.reset_optimizer_on_switch:
            self._reset_optimizer_states(self.blk_meta[self.cur_blk_idx]["param_names"])
        # Advance index
        if self.block_sequence == "ascending":
            next_idx = (self.cur_blk_idx + 1) % self.total_blocks
        elif self.block_sequence == "descending":
            next_idx = (self.cur_blk_idx - 1) % self.total_blocks
        else:
            next_idx = random.randint(0, self.total_blocks - 1)
        self.cur_blk_idx = next_idx
        # Activate new block
        if self.reset_optimizer_on_switch:
            self._reset_optimizer_states(self.blk_meta[self.cur_blk_idx]["param_names"])
        self._set_block_requires_grad(self.cur_blk_idx, True)
        for param in self._get_block_params_list:
            self._convert_param_dtype(param, self.compute_dtype)
        self._save_current_block_params()
        self.block_step_counts[self.cur_blk_idx] = 0

    def _save_current_block_params(self):
        self.prev_block_params.clear()
        self.prev_block_params[self.cur_blk_idx] = {
            name: param.detach().clone()
            for name, param in self._get_block_params_tuple
        }

    # =======================================================================
    # Block metadata computation  (unchanged from BlockAdamW)
    # =======================================================================

    def _detect_layer_pattern(self) -> Tuple[Optional[str], int]:
        patterns = [
            r"model\.layers\.(\d+)",
            r"transformer\.h\.(\d+)",
            r"model\.decoder\.layers\.(\d+)",
            r"decoder\.layers\.(\d+)",
            r"encoder\.layers\.(\d+)",
            r"layers\.(\d+)",
        ]
        layer_indices: Dict[int, bool] = {}
        matched_pattern = None
        for name, param in self.model.named_parameters():
            if self._is_frozen_param(name, param):
                continue
            for pattern in patterns:
                match = re.search(pattern, name)
                if match:
                    idx = int(match.group(1))
                    if matched_pattern is None:
                        matched_pattern = pattern
                    if matched_pattern == pattern:
                        layer_indices[idx] = True
        if matched_pattern and layer_indices:
            return matched_pattern, max(layer_indices.keys()) + 1
        return None, 0

    def _compute_blk_meta(self) -> List[Dict]:
        pattern, num_layers = self._detect_layer_pattern()
        if pattern is None or num_layers == 0:
            param_names: Set[str] = set()
            param_count = 0
            for name, param in self.model.named_parameters():
                if not self._is_frozen_param(name, param):
                    param_names.add(name)
                    param_count += param.numel()
            if not param_names:
                raise RuntimeError("No trainable parameters found in model")
            return [{"block_id": 0, "layer_indices": [], "param_count": param_count, "param_names": param_names}]

        layer_info: Dict[int, Dict] = {
            i: {"param_names": set(), "param_count": 0} for i in range(num_layers)
        }
        for name, param in self.model.named_parameters():
            if self._is_frozen_param(name, param):
                continue
            idx = self._get_layer_index(name, pattern)
            if idx is not None and idx in layer_info:
                layer_info[idx]["param_names"].add(name)
                layer_info[idx]["param_count"] += param.numel()

        if self.block_strategy == "random":
            return self._group_random(layer_info, num_layers)
        elif self.block_strategy == "importance":
            return self._group_importance(layer_info, num_layers)
        else:
            return self._group_fixed_size(layer_info, num_layers)

    def _group_fixed_size(self, layer_info: Dict, num_layers: int) -> List[Dict]:
        blocks = []
        for i in range(0, num_layers, self.block_size):
            end = min(i + self.block_size, num_layers)
            layer_indices = list(range(i, end))
            param_names: Set[str] = set()
            param_count = 0
            for li in layer_indices:
                if li in layer_info:
                    param_names.update(layer_info[li]["param_names"])
                    param_count += layer_info[li]["param_count"]
            if param_names:
                blocks.append({"block_id": len(blocks), "layer_indices": layer_indices,
                                "param_count": param_count, "param_names": param_names})
        return blocks

    def _group_random(self, layer_info: Dict, num_layers: int) -> List[Dict]:
        indices = list(range(num_layers))
        random.shuffle(indices)
        blocks = []
        for i in range(0, len(indices), self.block_size):
            layer_indices = sorted(indices[i: i + self.block_size])
            param_names: Set[str] = set()
            param_count = 0
            for li in layer_indices:
                if li in layer_info:
                    param_names.update(layer_info[li]["param_names"])
                    param_count += layer_info[li]["param_count"]
            if param_names:
                blocks.append({"block_id": len(blocks), "layer_indices": layer_indices,
                                "param_count": param_count, "param_names": param_names})
        return blocks

    def _group_importance(self, layer_info: Dict, num_layers: int) -> List[Dict]:
        sorted_layers = sorted(layer_info.items(), key=lambda x: x[1]["param_count"], reverse=True)
        blocks = []
        for i in range(0, len(sorted_layers), self.block_size):
            group = sorted_layers[i: i + self.block_size]
            layer_indices = sorted(li for li, _ in group)
            param_names: Set[str] = set()
            param_count = 0
            for _, info in group:
                param_names.update(info["param_names"])
                param_count += info["param_count"]
            if param_names:
                blocks.append({"block_id": len(blocks), "layer_indices": layer_indices,
                                "param_count": param_count, "param_names": param_names,
                                "importance": param_count})
        return blocks

    # =======================================================================
    # Checkpointing
    # =======================================================================

    def state_dict(self):
        sd = super().state_dict()
        sd["blk_meta"] = self.blk_meta
        sd["cur_blk_idx"] = self.cur_blk_idx
        sd["global_step"] = self.global_step
        sd["block_step_counts"] = self.block_step_counts
        return sd

    def load_state_dict(self, state_dict):
        self.blk_meta = state_dict.pop("blk_meta", self.blk_meta)
        self.cur_blk_idx = state_dict.pop("cur_blk_idx", 0)
        self.global_step = state_dict.pop("global_step", 0)
        self.block_step_counts = state_dict.pop("block_step_counts", [0] * self.total_blocks)
        super().load_state_dict(state_dict)
        self._set_block_requires_grad(self.cur_blk_idx, True)
        for param in self._get_block_params_list:
            self._convert_param_dtype(param, self.compute_dtype)

    # =======================================================================
    # Repr
    # =======================================================================

    def __repr__(self) -> str:
        g = self.param_groups[0]
        return (
            f"MeZOBlockAdamW("
            f"lr={g['lr']}, betas={g['betas']}, eps={g['eps']}, "
            f"weight_decay={g['weight_decay']}, zo_eps={self.zo_eps}, "
            f"block_strategy={self.block_strategy}, block_size={self.block_size}, "
            f"total_blocks={self.total_blocks}, "
            f"block_switch_frequency={self.block_switch_frequency})"
        )

