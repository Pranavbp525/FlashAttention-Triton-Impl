import torch
import triton
import triton.language as tl




class TritonAttention(torch.autograd.Function):


    @triton.jit
    def _attn_forward_inner(
        O_block,
        l_i,
        m_i,
        Q_block,
        K_block_ptr,
        V_block_ptr,
        block_index_q,
        softmax_scale,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
        STAGE: tl.constexpr,
        offs_q: tl.constexpr,
        offs_kv: tl.constexpr,
        SEQ_LEN: tl.constexpr
    ):
        if STAGE == 1:
            lo, hi = 0, block_index_q * BLOCK_SIZE_Q
        elif STAGE == 2:
            lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
            lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
        else:
            lo, hi = 0, SEQ_LEN

        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

        for start_kv in range(lo, hi, BLOCK_SIZE_KV):
            start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

            K_block = tl.load(K_block_ptr)
            QK_block = tl.dot(Q_block, K_block)

            if STAGE == 2:
                mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
                QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
                QK_block -= m_ij[:, None]
            else:
                m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
                QK_block = QK_block * softmax_scale - m_ij[:, None]

            P_block = tl.math.exp(QK_block)
            l_ij = tl.sum(P_block, 1)
            alpha = tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            V_block = tl.load(V_block_ptr)
            P_block = P_block.to(tl.float16)

            O_block = O_block * alpha[:, None]
            O_block = tl.dot(P_block, V_block, O_block)

            m_i = m_ij

            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))

        return O_block, l_i, m_i




    @triton.jit
    def _attn_forward(
        Q, 
        K, 
        V, 
        softmax_scale, 
        M, 
        O, 
        stride_Q_batch, 
        stride_Q_head, 
        stride_Q_seq, 
        stride_Q_dim,
        stride_K_batch, 
        stride_K_head, 
        stride_K_seq, 
        stride_K_dim,
        stride_V_batch, 
        stride_V_head, 
        stride_V_seq, 
        stride_V_dim,
        stride_O_batch, 
        stride_O_head, 
        stride_O_seq, 
        stride_O_dim,
        BATCH_SIZE,
        NUM_HEADS: tl.constexp,
        SEQ_LEN: tl.constexp,
        HEAD_DIM: tl.constexp,
        BLOCK_SIZE_Q: tl.constexp,
        BLOCK_SIZE_KV: tl.constexp,
        STAGE: tl.constexpr
    ):

        tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

        block_index_q = tl.program_id(0)
        index_batch_head = tl.program_id(1)
        index_batch = index_batch_head // NUM_HEADS
        index_head = index_batch_head % NUM_HEADS

        qkv_offset = (index_batch.to(tl.int64) * stride_Q_batch + index_head.to(tl.int64) * stride_Q_head)

        Q_block_ptr = tl.make_block_ptr(
            base=Q+qkv_offset,
            shape=(SEQ_LEN, HEAD_DIM),
            strides=(stride_Q_seq, stride_Q_dim),
            offsets=(block_index_q*BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
            order=(1,0)
        )

        V_block_ptr = tl.make_block_ptr(
            base=V+qkv_offset,
            shape=(SEQ_LEN, HEAD_DIM),
            strides=(stride_V_seq, stride_V_dim),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
            order=(1,0)
        )

        K_block_ptr = tl.make_block_ptr(
            base=K+qkv_offset,
            shape=(HEAD_DIM, SEQ_LEN),
            strides=(stride_K_dim, stride_K_seq),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
            order=(0,1)
        )

        O_block_ptr = tl.make_block_ptr(
            base=O+qkv_offset,
            shape=(SEQ_LEN, HEAD_DIM),
            strides=(stride_O_seq, stride_O_dim),
            offsets=(block_index_q*BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
            order=(1,0)
        )

        offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

        offs_kv = tl.arange(0, BLOCK_SIZE_KV)

        m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float('inf')

        l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1

        O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

        Q_block = tl.load(Q_block_ptr)

        if STAGE == 1 or STAGE == 3:
            O_block, l_i, m_i = self._attn_forward_inner(
                O_block,
                l_i,
                m_i,
                Q_block,
                K_block_ptr,
                V_block_ptr,
                block_index_q,
                softmax_scale,
                BLOCK_SIZE_Q,
                BLOCK_SIZE_KV,
                4 - STAGE,
                offs_q,
                offs_kv,
                SEQ_LEN
            )
        
        if STAGE == 3:
            O_block, l_i, m_i = self._attn_forward_inner(
                O_block,
                l_i,
                m_i,
                Q_block,
                K_block_ptr,
                V_block_ptr,
                block_index_q,
                softmax_scale,
                BLOCK_SIZE_Q,
                BLOCK_SIZE_KV,
                2,
                offs_q,
                offs_kv,
                SEQ_LEN
            )

        m_i += tl.math.log(l_i)

        O_block = O_block / l_i[:, None]
        m_ptrs = M + index_batch_head * SEQ_LEN + offs_q 
        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, O_block.to(O.type.element_ty))







        


    

    @static_method
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q = Q.shape[-1]
        HEAD_DIM_K = K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_K == HEAD_DIM_Q and HEAD_DIM_Q == HEAD_DIM_V

        O = torch.empty_like(Q)
        
        stage = 3 if causal else 1

        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args['BLOCK_SIZE_Q']),
            BATCH_SIZE*NUM_HEADS,
            1,
        )

        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        self._attn_forward[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage
        )
        
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal

        return 0
