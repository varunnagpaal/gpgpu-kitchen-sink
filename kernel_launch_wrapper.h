#pragma once
#ifndef _KERNEL_LAUNCH_WRAPPER_H
#define _KERNEL_LAUNCH_WRAPPER_H

namespace cuda {
/**
 * CUDA's kernel launching mechanism cannot be compiled in C++ - both for syntactic and semantic reasons.
 * Thus, every kernel launch must at some point reach code compiled with CUDA's nvcc. Naively, every single
 * different kernel (perhaps up to template specialization) would require writing its own wrapper C++ function,
 * launching it. This function, however, constitutes a single minimal wrapper around the CUDA kernel launch,
 * which may be called from proper C++ code.
 *
 * <p>This function is similar to a beta-reduction in Lambda calculus: It applies a function to its arguments;
 * the difference is in the nature of the function (a CUDA kernel) and in that the function application
 * requires setting additional CUDA-related launch parameters, other than the function's own.
 *
 * <p>As kernels do not return values, neither does this function. It also contains
 * no hooks, logging commands etc. - if you want those, write your own wrapper (perhaps calling this one in 
 * turn).
 *
 * @param kernel_function the kernel to apply. Pass it just as-it-is, as though it were any other function. Note:
 * If the kernel is templated, you must pass it fully-instantiated.
 * @param grid_dimensions the number of CUDA execution grid blocks in each of upto 3 dimensions
 * @param block_dimensions the number of CUDA threads (a.k.a. hardware threads, or 'CUDA cores') in every
 * execution grid block, in each of upto 3 dimensions.
 * @param shared_memory_size the amount, in bytes, of shared memory to allocate for common use by each execution
 * block in the grid; limited by your specific GPU's capabilities and typically <= 48 Ki.
 * @param stream the CUDA hardware command queue on which to place the command to launch the kernel (affects
 * the scheduling of the launch and the execution)
 * @param parameters whatever parameters {@kernel_function} takes
 */
template<typename KernelFunction, typename... KernelParameters>
void launchKernel(
    const KernelFunction&   kernel_function,
    dim3                    grid_dimensions,
    dim3                    block_dimensions,
    unsigned                shared_memory_size,
    cudaStream_t            stream,
    KernelParameters...     parameters)
#ifndef __CUDACC__
    ;
#else
{
    kernel_function<<<grid_dimensions, block_dimensions, shared_memory_size, stream>>>(parameters...);
}
#endif

} /* namespace cuda */

#endif /* _KERNEL_LAUNCH_WRAPPER_H */
