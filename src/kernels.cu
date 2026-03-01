#include <vector>
#include <cuda_fp16.h>
#include <type_traits>
#include "../tester/utils.h"

// 为half类型特化的exp函数
#ifdef __CUDACC__
template <typename T>
__device__ __host__ inline T exp_fp16(T x)
{
  return exp(x);
}

template <>
__device__ __host__ inline __half exp_fp16(__half x)
{
#if __CUDA_ARCH__ >= 530
  return hexp(x); // CUDA 原生half指数函数
#else
  // 模拟实现：先转换为float，计算exp，再转换回half
  return __float2half(exp(__half2float(x)));
#endif
}
#else
template <typename T>
inline T exp_fp16(T x)
{
  return std::exp(x);
}

template <>
inline __half exp_fp16(__half x)
{
  // CPU端的fallback实现
  float f = __half2float(x);
  f = std::exp(f);
  return __float2half(f);
}
#endif

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
__global__ void trace_kernel_naive(T *input, T *outputV, size_t diagonal_length, size_t cols)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = 0;
  if (idx < diagonal_length)
  {
    index = idx * cols + idx;
    atomicAdd(outputV, input[index]);
  }
}

template <typename T>
__global__ void trace_kernel_smem(T *input, T *output, size_t diagonal_length, size_t cols)
{
  // extern __shared__ T sMem[];// compile error
  extern __shared__ char shared_memory[];
  T *sMem = reinterpret_cast<T *>(shared_memory);

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = 0;
  // load2smem
  size_t tid = threadIdx.x;
  sMem[tid] = T(0);

  if (idx < diagonal_length)
  {
    index = idx * cols + idx;
    sMem[tid] = input[index]; // idx lead2 illegal memory access
  }
  __syncthreads();
  // reduce
  for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1)
  {
    if (tid < i)
    {
      sMem[tid] += sMem[tid + i];
    }
    __syncthreads();
  }
  // if (tid == 0) {// cannot process multi block for large matrix
  //   *output = sMem[0];
  // }
  if (tid == 0)
  {
    atomicAdd(output, sMem[0]);
  }
}

template <typename T>
T trace(const std::vector<T> &h_input, size_t rows, size_t cols)
{
  // TODO: Implement the trace function
  int diagonal_length = min(rows, cols);

  T trace_value = T(0);
#if 0
  int index = 0;
  for (int i=0;i <diagonal_length;i++){
      index = i * cols + i;
      trace_value += h_input[index];
  }
#elif 1
  // mem alloc
  int matrix_length = rows * cols;
  T *d_input;
  cudaMalloc((void **)&d_input, matrix_length * sizeof(T));
  cudaMemcpy(d_input, h_input.data(), matrix_length * sizeof(T), cudaMemcpyHostToDevice);

  T *d_output;
  cudaMalloc((void **)&d_output, sizeof(T));
  T zero = T(0);
  cudaMemcpy(d_output, &zero, sizeof(T), cudaMemcpyHostToDevice);

  // kernel call
  dim3 block(1024, 1);
  dim3 grid((diagonal_length - 1) / block.x + 1, 1);

  size_t sMem_size = 1024 * sizeof(T);
#if 1
  trace_kernel_naive<<<grid, block>>>(d_input, d_output, diagonal_length, cols); // greater
#elif 1
  trace_kernel_smem<<<grid, block, sMem_size>>>(d_input, d_output, diagonal_length, cols);
#endif
  cudaDeviceSynchronize();

  // get result
  cudaMemcpy(&trace_value, d_output, sizeof(T), cudaMemcpyDeviceToHost);
#endif
  return T(trace_value);
}
template <int Br, int Bc, typename T>
__global__ void flash_fwd_kernel(const T *Q, const T *K, const T *V, T *O, int batch_size, int target_seq_len, int src_seq_len, int query_heads, int kv_heads,
                                 int head_dim, bool is_causal)
{
  int batch_idx = blockIdx.x;
  int query_head_idx = blockIdx.y;
  int tile_idx = blockIdx.z;
  int thread_idx = threadIdx.x;

  int kv_head_idx = query_head_idx / ((int) query_heads/kv_heads); // GQA

  extern __shared__ char shared_mem[];
  T *q_tile = reinterpret_cast<T *>(&shared_mem[0]);
  T *k_tile = &q_tile[Br * head_dim];
  T *v_tile = &k_tile[Bc * head_dim];

  // load Q and KV to smem
  // Q  : [batch_size, target_seq_len, query_heads, head_dim]
  const int stride_q_batch = target_seq_len * query_heads * head_dim;
  const int stride_q_seq = query_heads * head_dim;
  const int stride_q_head = head_dim;
  // KV : [batch_size, src_seq_len, kv_heads, head_dim]
  const int stride_kv_batch = src_seq_len * kv_heads * head_dim;
  const int stride_kv_seq = kv_heads * head_dim;
  const int stride_kv_head = head_dim;

  // batch bias
  const T* Q_batch = Q + batch_idx * stride_q_batch;
  T* O_batch = O + batch_idx * stride_q_batch;
  const T* K_batch = K + batch_idx * stride_kv_batch;
  const T* V_batch = V + batch_idx * stride_kv_batch;

  // 查询块起始行
  const int q_block_start_row = tile_idx * Br;
  int q_global_row = q_block_start_row + thread_idx;
  
  // o, l, m
  T o_i[256] = {static_cast<T>(0)};
  T m_i = static_cast<T>(-1e10);
  T l_i = static_cast<T>(0);

  // 加载查询块到共享内存
  for (int i = thread_idx; i < Br * head_dim; i += blockDim.x)
  {
    int row = i / head_dim;
    int col = i % head_dim;
    int global_row_q = q_block_start_row + row;
    if (row < Br && global_row_q < target_seq_len)
    {
      int offset = global_row_q * stride_q_seq + query_head_idx * stride_q_head + col;
      q_tile[row * head_dim + col] = Q_batch[offset];
    }
    else
    {
      q_tile[row * head_dim + col] = static_cast<T>(0);
    }
  }
  __syncthreads();

  // 检查当前线程是否有效的查询行
  if (thread_idx >= Br || q_global_row >= target_seq_len) {
    return;  // 无效线程提前返回
  }

  // 循环遍历键值块
  for (int col_block_start = 0; col_block_start < src_seq_len; col_block_start += Bc)
  {
    // 加载键值块到共享内存
    for (int i = thread_idx; i < Bc * head_dim; i += blockDim.x)
    {
      int row = i / head_dim;
      int col = i % head_dim;
      int global_row_kv = col_block_start + row;
      if (row < Bc && global_row_kv < src_seq_len)
      {
        int kv_offset = global_row_kv * stride_kv_seq + kv_head_idx * stride_kv_head + col;
        k_tile[row * head_dim + col] = K_batch[kv_offset];
        v_tile[row * head_dim + col] = V_batch[kv_offset];
      }
      else
      {
        k_tile[row * head_dim + col] = static_cast<T>(0);
        v_tile[row * head_dim + col] = static_cast<T>(0);
      }
    }
    __syncthreads();

    // 注意力分数 S_ij = Q_i @ K_j^T
    T s_score[Bc] = {static_cast<T>(0)};
    
    // 计算缩放因子 self-attention
    T scale = static_cast<T>(1.0);
    if constexpr (std::is_same_v<T, float>) {
        scale = rsqrtf(static_cast<float>(head_dim));
    } else if constexpr (std::is_same_v<T, half>) {
        scale = hrsqrt(__float2half(static_cast<float>(head_dim)));
    }

    // 计算查询与所有键的点积
    for (int k_local_col = 0; k_local_col < Bc; ++k_local_col) {
      T dot_sum = static_cast<T>(0);
      for (int d = 0; d < head_dim; ++d) {
        T q_val = q_tile[thread_idx * head_dim + d];
        T k_val = k_tile[k_local_col * head_dim + d];
        dot_sum += q_val * k_val;
      }
      s_score[k_local_col] = dot_sum * scale;

      // 应用mask
      int k_global_col = col_block_start + k_local_col;
      bool is_masked = false;
      
      if (k_global_col >= src_seq_len) {
        is_masked = true;
      } else if (is_causal && q_global_row < k_global_col) {
        is_masked = true;
      }
      
      if (is_masked) {
        s_score[k_local_col] = static_cast<T>(-1e10);
      }
    }

    // 在线softmax计算 P_ij = softmax(S_ij)
    // 1. 计算当前块的最大值
    T m_block = static_cast<T>(-1e10);
    for (int k = 0; k < Bc; ++k) {
      if (s_score[k] > m_block) {
        m_block = s_score[k];
      }
    }

    // 如果所有分数都被masked，跳过这个块
    if (m_block == static_cast<T>(-1e10)) {
      __syncthreads();
      continue;
    }

    // 2. 更新全局最大值
    T m_new = (m_block > m_i) ? m_block : m_i;

    // 3. 计算exp(s - m_new)和块内的sum
    T l_block = static_cast<T>(0);
    for (int k = 0; k < Bc; ++k) {
      if (s_score[k] > static_cast<T>(-1e9)) {  // 避免被masked的值
        T val = exp_fp16(s_score[k] - m_new);
        s_score[k] = val;  // 存储exp后的值
        l_block += val;
      } else {
        s_score[k] = static_cast<T>(0);
      }
    }

    // 4. 更新全局sum
    T exp_old = (l_i > static_cast<T>(0)) ? exp_fp16(m_i - m_new) : static_cast<T>(0);
    T l_new = l_i * exp_old + l_block;

    // 5. 重新缩放旧的输出累加
    if (l_new > static_cast<T>(0)) {
      T scale_factor = (l_i * exp_old) / l_new;
      for (int d = 0; d < head_dim; ++d) {
        o_i[d] *= scale_factor;
      }
    } else {
      for (int d = 0; d < head_dim; ++d) {
        o_i[d] = static_cast<T>(0);
      }
    }

    // 6. 添加当前块的贡献
    if (l_new > static_cast<T>(0)) {
      T inv_l_new = static_cast<T>(1.0) / l_new;
      for (int k = 0; k < Bc; ++k) {
        T p_val = s_score[k];
        if (p_val > static_cast<T>(0)) {
          for (int d = 0; d < head_dim; ++d) {
            o_i[d] += p_val * v_tile[k * head_dim + d] * inv_l_new;
          }
        }
      }
    }

    // 7. 更新全局状态
    l_i = l_new;
    m_i = m_new;
    
    __syncthreads();
  }

  // 写回全局内存
  if (q_global_row < target_seq_len) {
    int offset = q_global_row * stride_q_seq + query_head_idx * stride_q_head;
    for (int d = 0; d < head_dim; ++d) {
      O_batch[offset + d] = o_i[d];
    }
  }
}

template <typename T>
void flash_attention_fwd_cuda(const T *d_q, const T *d_k, const T *d_v, T *d_o, int batch_size, int target_seq_len,
                              int src_seq_len, int query_heads, int kv_heads, int head_dim, bool is_causal)
{
  constexpr int Br = 32; // Q_BLOCK_SIZE
  constexpr int Bc = 32; // KV_BLOCK_SIZE

  dim3 grid(batch_size, query_heads, (target_seq_len + Br - 1) / Br);
  dim3 block(Br);
  size_t shared_mem_size = (Br + 2 * Bc) * head_dim * sizeof(T);
  flash_fwd_kernel<Br, Bc, T><<<grid, block, shared_mem_size>>>(d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
                                                                query_heads, kv_heads, head_dim, is_causal);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());
  // const T NEG_INF = static_cast<T>(-1e10);
  // const T EPSILON = static_cast<T>(1e-10);
  // const T P_DROP = static_cast<T>(0.2);

  // int Tr = (target_seq_len + Q_BLOCK_SIZE - 1) / Q_BLOCK_SIZE;// 2
  // int Tc = (src_seq_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;// 2
}


/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len, // 1, 6, 6
                    int query_heads, int kv_heads, int head_dim, bool is_causal)
{ // 1, 1, 4
  // TODO: Implement the flash attention function
  // REF: https://pillumina.github.io/posts/aiinfra/11-flashattention/
#if 1
  // CUDA 写法
  // 0. check size
  int q_elems = batch_size * target_seq_len * query_heads * head_dim;
  int kv_elems = batch_size * src_seq_len * kv_heads * head_dim;
  if (h_q.size() != q_elems || h_k.size() != kv_elems || h_v.size() != kv_elems)
  {
    std::cerr << "Input vector sizes do not match the provided dimensions." << std::endl;
    return;
  }
  if (h_o.size() != q_elems)
  {
    h_o.resize(q_elems); // 24
    std::fill(h_o.begin(), h_o.end(), static_cast<T>(0));
  }
  // 1. malloc
  T *d_q, *d_k, *d_v, *d_o;
  RUNTIME_CHECK(cudaMalloc(&d_q, q_elems * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, kv_elems * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, kv_elems * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, q_elems * sizeof(T)));

  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_elems * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_elems * sizeof(T), cudaMemcpyHostToDevice));
  // 2. kernel
  flash_attention_fwd_cuda(d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
                           query_heads, kv_heads, head_dim, is_causal);

  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, q_elems * sizeof(T), cudaMemcpyDeviceToHost));
  // 3. free
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));

  return;
#elif 1
  // TODO:C++ Implement, not only pass test 1 and 2
  const int Q_BLOCK_SIZE = 32; // 可调整的块大小
  const int KV_BLOCK_SIZE = 32;
  const T NEG_INF = static_cast<T>(-1e10);
  const T EPSILON = static_cast<T>(1e-10);
  const T P_DROP = static_cast<T>(0.2);

  int Tr = (target_seq_len + Q_BLOCK_SIZE - 1) / Q_BLOCK_SIZE; // 2
  int Tc = (src_seq_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;  // 2

  // 确保输出向量大小正确
  h_o.resize(batch_size * target_seq_len * query_heads * head_dim); // 24
  std::fill(h_o.begin(), h_o.end(), static_cast<T>(0));

  // 初始化统计量
  std::vector<T> l(batch_size * target_seq_len * query_heads, static_cast<T>(0)); // 6
  std::vector<T> m(batch_size * target_seq_len * query_heads, NEG_INF);           // 6

  // 为每个batch和head并行处理
  // #pragma omp parallel for collapse(2)
  for (int b = 0; b < batch_size; ++b)
  { // 1
    for (int hq = 0; hq < query_heads; ++hq)
    {                                                  // 1
      int kv_head_idx = (hq * kv_heads) / query_heads; // GQA支持 0

      // 将输出、l、m分块
      std::vector<std::vector<T>> O_blocks(Tr);
      std::vector<std::vector<T>> l_blocks(Tr);
      std::vector<std::vector<T>> m_blocks(Tr);

      // 获取 block的 o, l ,m
      for (int i = 0; i < Tr; ++i)
      { // 2
        int q_start = i * Q_BLOCK_SIZE;
        int q_end = std::min(q_start + Q_BLOCK_SIZE, target_seq_len);
        int q_len = q_end - q_start; // 3

        O_blocks[i].resize(q_len * head_dim, static_cast<T>(0)); // 12

        for (int q_idx = 0; q_idx < q_len; ++q_idx)
        {
          int global_q_idx = q_start + q_idx;
          int l_idx = (b * target_seq_len + global_q_idx) * query_heads + hq;
          l_blocks[i].push_back(l[l_idx]);
          m_blocks[i].push_back(m[l_idx]);
        }
      }

      // 外层循环遍历Key/Value块
      for (int j = 0; j < Tc; ++j)
      {
        int kv_start = j * KV_BLOCK_SIZE;
        int kv_end = std::min(kv_start + KV_BLOCK_SIZE, src_seq_len);
        int kv_len = kv_end - kv_start; // 3

        // 内层循环遍历Query块
        for (int i = 0; i < Tr; ++i)
        {
          int q_start = i * Q_BLOCK_SIZE;
          int q_end = std::min(q_start + Q_BLOCK_SIZE, target_seq_len);
          int q_len = q_end - q_start; // 3

          // 计算S_ij = Q_i @ K_j^T
          std::vector<T> S_ij(q_len * kv_len, static_cast<T>(0));
          for (int qi = 0; qi < q_len; ++qi)
          {
            for (int kj = 0; kj < kv_len; ++kj)
            {
              T dot = static_cast<T>(0);
              for (int d = 0; d < head_dim; ++d)
              {
                int q_idx = ((b * target_seq_len + (q_start + qi)) * query_heads + hq) * head_dim + d;
                int k_idx = ((b * src_seq_len + (kv_start + kj)) * kv_heads + kv_head_idx) * head_dim + d;
                dot += h_q[q_idx] * h_k[k_idx];
              }

              // 应用因果mask
              if (is_causal && (q_start + qi) < (kv_start + kj))
              {
                dot = NEG_INF;
              }

              S_ij[qi * kv_len + kj] = dot;
            }
          }

          // 计算P_ij = softmax(S_ij)
          // 1. 计算每行最大值m_block_ij
          std::vector<T> m_block_ij(q_len, NEG_INF);
          for (int qi = 0; qi < q_len; ++qi)
          {
            for (int kj = 0; kj < kv_len; ++kj)
            {
              T val = S_ij[qi * kv_len + kj];
              if (val > m_block_ij[qi])
              {
                m_block_ij[qi] = val;
              }
            }
          }

          // 2. 计算exp(S_ij - m_block_ij)和l_block_ij
          std::vector<T> P_ij(q_len * kv_len, static_cast<T>(0));
          std::vector<T> l_block_ij(q_len, static_cast<T>(0));

          for (int qi = 0; qi < q_len; ++qi)
          {
            T sum_exp = static_cast<T>(0);
            for (int kj = 0; kj < kv_len; ++kj)
            {
              T val = S_ij[qi * kv_len + kj];
              T exp_val = exp_fp16(val - m_block_ij[qi]);
              P_ij[qi * kv_len + kj] = exp_val;
              sum_exp += exp_val;
            }
            l_block_ij[qi] = sum_exp + EPSILON;
          }

          // 3. 计算P_ij @ V_j
          std::vector<T> P_ij_Vj(q_len * head_dim, static_cast<T>(0));
          for (int qi = 0; qi < q_len; ++qi)
          {
            for (int d = 0; d < head_dim; ++d)
            {
              T sum = static_cast<T>(0);
              for (int kj = 0; kj < kv_len; ++kj)
              {
                int v_idx = ((b * src_seq_len + (kv_start + kj)) * kv_heads + kv_head_idx) * head_dim + d;
                sum += P_ij[qi * kv_len + kj] * h_v[v_idx];
              }
              P_ij_Vj[qi * head_dim + d] = sum;
            }
          }

          // 更新统计量和输出
          for (int qi = 0; qi < q_len; ++qi)
          {
            T mi_val = m_blocks[i][qi];
            T li_val = l_blocks[i][qi];
            T m_block_val = m_block_ij[qi];
            T l_block_val = l_block_ij[qi];

            // 计算新的mi和li
            T mi_new = std::max(mi_val, m_block_val);
            T li_new = exp_fp16(mi_val - mi_new) * li_val +
                       exp_fp16(m_block_val - mi_new) * l_block_val;

            // 更新输出块
            for (int d = 0; d < head_dim; ++d)
            {
              int idx = qi * head_dim + d;
              T oi_val = O_blocks[i][idx];
              T pv_val = P_ij_Vj[idx];

              O_blocks[i][idx] = (li_val / li_new) * exp_fp16(mi_val - mi_new) * oi_val +
                                 (exp_fp16(m_block_val - mi_new) / li_new) * pv_val;
            }

            // 保存新的统计量
            l_blocks[i][qi] = li_new;
            m_blocks[i][qi] = mi_new;
          }
        }
      }

      // 合并所有块到最终输出
      for (int i = 0; i < Tr; ++i)
      {
        int q_start = i * Q_BLOCK_SIZE;
        int q_end = std::min(q_start + Q_BLOCK_SIZE, target_seq_len);
        int q_len = q_end - q_start;

        for (int qi = 0; qi < q_len; ++qi)
        {
          int global_q_idx = q_start + qi;

          // 更新全局l和m
          int l_idx = (b * target_seq_len + global_q_idx) * query_heads + hq;
          l[l_idx] = l_blocks[i][qi];
          m[l_idx] = m_blocks[i][qi];

          // 复制输出
          for (int d = 0; d < head_dim; ++d)
          {
            int src_idx = qi * head_dim + d;
            int dst_idx = ((b * target_seq_len + global_q_idx) * query_heads + hq) * head_dim + d;
            h_o[dst_idx] = O_blocks[i][src_idx];
          }
        }
      }
    }
  }
#endif
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int> &, size_t, size_t);
template float trace<float>(const std::vector<float> &, size_t, size_t);
template void flashAttention<float>(const std::vector<float> &, const std::vector<float> &,
                                    const std::vector<float> &, std::vector<float> &,
                                    int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half> &, const std::vector<half> &,
                                   const std::vector<half> &, std::vector<half> &,
                                   int, int, int, int, int, int, bool);
