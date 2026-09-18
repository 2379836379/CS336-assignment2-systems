import torch
import triton
import triton.language as tl
from einops import rearrange


@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,  # 输入指针
    output_ptr,  # 输出指针
    x_stride_row,
    x_stride_dim,  # strides 用来描述在张量各个轴上移动一个元素时的步长
    weight_stride_dim,  # 通常为 1
    output_stride_row,  # 通常为 1
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,  # tile 的形状必须在编译期已知
):
    # 每个 kernel 实例负责处理若干行输入，也就是一个行方向的 tile。
    # 对于这个 tile 里的每一行，目标都是计算 `sum_j x[row, j] * weight[j]`。
    # `tl.program_id(0)` 返回当前实例在第 0 维 grid 中的编号，
    # 这里它直接对应“当前正在处理第几个行 tile”。
    row_tile_idx = tl.program_id(0)
    # 块指针让我们能够从多维内存区域中选取数据
    # 并在这个选区上移动。
    # 块指针需要知道：
    # - 张量首元素的指针
    # - 张量的整体形状，用于处理越界访问
    # - 每个维度的 stride，用于正确解释内存布局
    # - 起始块的多维坐标，也就是 `offsets`
    # - 每次 load/store 的块形状
    # - 维度在内存中的主次顺序
    # 维度顺序（= np.argsort(strides)）用于优化，也是
    # 在 >= Hopper 架构上支持 TMA 所必需的
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    # 初始化写入结果的缓冲区。
    # 这个向量长度等于一个行 tile 的行数，
    # output[k] 会累计当前 tile 中第 k 行的部分点积结果。
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        # 加载当前块指针对应的数据。
        # row 的形状是 (ROWS_TILE_SIZE, D_TILE_SIZE)，表示一个二维输入子块；
        # weight 的形状是 (D_TILE_SIZE,)，表示当前 D 子区间上的权重切片。
        # 由于 ROWS_TILE_SIZE 可能不能整除 NUM_ROWS，且 D_TILE_SIZE 可能不能整除 D，
        # 最后一个 tile 可能越界，因此两个维度都需要做边界检查。
        # 越界位置会被 padding 为 0，从而不影响最终求和结果。
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)
        # 计算当前 D 子区间对输出的贡献。
        # `weight[None, :]` 会把一维权重扩成 (1, D_TILE_SIZE)，
        # 从而和 row 做逐元素乘法；随后沿 axis=1 求和，
        # 得到这个 tile 中每一行在当前列块上的部分点积。
        output += tl.sum(row * weight[None, :], axis=1)
        # 将指针移动到下一个列方向 tile。
        # 这里行偏移保持为 0，表示仍然处理同一批行；
        # 只有列偏移增加 D_TILE_SIZE，继续覆盖下一段 embedding 维度。
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))  # 在最后一个维度上前进 D_TILE_SIZE
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))  # 权重向量同步前进 D_TILE_SIZE
    # 将输出写回输出块指针（每行对应一个标量）。
    # 由于 ROWS_TILE_SIZE 可能不能整除 NUM_ROWS，这里需要边界检查
    tl.store(output_block_ptr, output, boundary_check=(0,))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # 保存 x 和 weight，供反向传播阶段使用。
        # forward 结束后，autograd 在 backward 里只会把 `grad_out`
        # 传回来，因此我们必须缓存前向参与计算的输入，
        # 才能恢复出 `grad_x` 和 `grad_weight`。
        D, output_dims = x.shape[-1], x.shape[:-1]
        # 将输入张量重排为二维。
        # Triton kernel 这里按二维矩阵来做指针寻址：
        # 前面的所有 batch/序列维会先展平成 `NUM_ROWS`，最后一维保留为 D。
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        ctx.save_for_backward(x, weight)
        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16  # 大致让 embedding 维度循环 16 次
        ctx.ROWS_TILE_SIZE = 16  # 每个线程块一次处理 16 个 batch 元素
        ctx.input_shape = input_shape
        # 需要初始化一个空的结果张量。
        # 这里使用 `empty` 是因为 kernel 会覆盖每个输出位置，
        # 因此不需要提前清零。
        y = torch.empty(output_dims, device=x.device)
        # 在一维 grid 上启动 kernel。
        # 每个实例负责一个行 tile，所以 grid 大小等于
        # `ceil(n_rows / ROWS_TILE_SIZE)`。
        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x,
            weight,
            y,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            y.stride(0),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE  # 它们不一定需要相同
        n_rows, D = x.shape
        # 这里的策略是让每个线程块先写入一个局部缓冲区，
        # 然后再对这个缓冲区做归约，得到最终梯度。
        # 原因是 `grad_weight` 会被所有行共同累加，
        # 多个线程块如果直接写同一个位置会产生写冲突；
        # 先写局部结果再在 PyTorch 侧求和，实现上更简单也更安全。
        partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)
        weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x,
            weight,
            grad_out,
            grad_x,
            partial_grad_weight,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            partial_grad_weight.stride(0),
            partial_grad_weight.stride(1),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight


@triton.jit
def weighted_sum_backward(
    x_ptr,
    weight_ptr,  # 输入
    grad_output_ptr,  # 梯度输入
    grad_x_ptr,
    partial_grad_weight_ptr,  # 梯度输出
    stride_xr,
    stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr,
    stride_gxd,
    stride_gwb,
    stride_gwd,
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)
    # 当前 backward kernel 实例同样只负责一个行 tile。
    # `n_row_tiles` 表示总共有多少个行 tile，
    # 它同时决定了 partial_grad_weight 第一维的大小。
    # 下面这些块指针分别对应 grad_out、x、weight、grad_x 和局部 grad_weight。
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )
    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")  # (ROWS_TILE_SIZE,)
        # grad_output 是当前行 tile 上的输出梯度。
        # 对于 `y[row] = sum_j x[row, j] * weight[j]`，
        # 有 `grad_x[row, j] = grad_output[row] * weight[j]`，
        # 因此这里可以通过一个外积一次性得到整个子块的 grad_x。
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))
        # 计算 grad_weight 的局部贡献。
        # 根据链式法则，`grad_weight[j] = sum_row x[row, j] * grad_output[row]`。
        # 当前线程块只能看到自己负责的那些行，
        # 所以这里只对本 tile 内的行做归约，得到一个局部结果。
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))  # 第 0 维永远不会越界
        # 沿着 D 维把指针移动到下一个 tile。
        # 这样下一轮循环会处理同一批行在下一段列区间上的梯度计算。
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
