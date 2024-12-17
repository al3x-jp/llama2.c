/*
**  Start of File
*/
#include <CL/sycl.hpp>
using namespace sycl;

/*
**  SYCL version of matmul(..) function
*/
extern "C" void sycl_matmul(float* xout, float* x, float* w, int n, int d) 
{
    static bool bFirstTime = false;
    static cl::sycl::queue q{ cl::sycl::gpu_selector{},  property::queue::enable_profiling{} };

    if (!bFirstTime)
    {
        // Work out th optimal tile size for this device
        auto device = q.get_device();

        // Report what the default device is
        std::cout 	<< "INFO(SYCL): Using Compute Device - "
                    << device.get_info<cl::sycl::info::device::name>() 
                    << std::endl << std::endl;

        bFirstTime = true;        
    }

    const unsigned long M = 1;
    const unsigned long K = n;
    const unsigned long N = d;

    const unsigned int tile_size = 16;

    // Set up workgroup sizes
	range global 	{M, N};
	range local 	{1, tile_size};

    { // Start of SYCL Buffer Scope

        sycl::buffer<float, 1> buf_x(x, K*M);
        sycl::buffer<float, 1> buf_w(w, N*K);
        sycl::buffer<float, 1> buf_xout(xout, N*M);

        q.submit([&](auto& h) 
        {
            sycl::accessor acc_x(buf_x, h, sycl::read_only);            // Input
            sycl::accessor acc_w(buf_w, h, sycl::read_only);            // Weights
            sycl::accessor acc_xout(buf_xout, h, sycl::write_only);     // Output

            sycl::local_accessor<float, 1> a_tile(range<1>(tile_size), h);

            h.parallel_for(nd_range<2>{global, local}, [=](nd_item<2> item) 
            {
                // Indices in the global index space:
                int m = item.get_global_id()[0];    // M
                int n = item.get_global_id()[1];    // N, Fast moving pointer
                
                // Index in the local index space:
                int i = item.get_local_id()[1];     // N, Fast moving pointer
                float sum = 0;
                
                for (int kk = 0; kk < K; kk += tile_size) 
                {
                    // Each workgroup item reads a tile of size n from the input along the row (fast-moving pointer) direction
                    // "m" = 1, so no collum indexing in the input            
                    a_tile[i] = acc_x[(m * K) + (kk + i)];
                    
                    // Sycnchonise so all workgroups have read a 32-size tile of input data
                    item.barrier();
                    
                    // Perform computation using the local memory tile, and
                    // matrix B in global memory.

                    for (int k = 0; k < tile_size; k++)
                    {
                        sum += a_tile[k] * acc_w[(kk + k) + (n * K)];
                    }

                    // After computation, synchronize again, to ensure all
                    // reads from the local memory tile are complete.
                    item.barrier();
                }
                
                // Write the final result to global memory.
                acc_xout[(m * N) + n] = sum;
            });

        });
    }
}
/*
**  End of File
*/
