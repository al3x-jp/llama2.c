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

    const unsigned long N_matrix_rows = n;
    const unsigned long D_matrix_cols_vec_rows = d;

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
    
    { // Start of SYCL Buffer Scope

        sycl::buffer<float, 1> buf_x(x, N_matrix_rows);
        sycl::buffer<float, 1> buf_w(w, D_matrix_cols_vec_rows*N_matrix_rows+N_matrix_rows);
        sycl::buffer<float, 1> buf_xout(xout, D_matrix_cols_vec_rows);

        q.submit([&](auto& h) 
        {
            sycl::accessor acc_x(buf_x, h, sycl::read_only);
            sycl::accessor acc_w(buf_w, h, sycl::read_only);
            sycl::accessor acc_xout(buf_xout, h, sycl::write_only);

#if DO_SINGLE_TASK

            h.single_task([=]() 
            {
                for (int i = 0; i < D_matrix_cols_vec_rows; i++) {
                    float val = 0.0f;
                    for (int j = 0; j < N_matrix_rows; j++) {
                        val += acc_w[i * N_matrix_rows + j] * acc_x[j];
                    }
                    acc_xout[i] = val;
                }
            });
            q.wait();

#else

            h.parallel_for(range{D_matrix_cols_vec_rows}, [=](id<1> idx) 
			{
				int i = idx[0]; // P

				float sum = 0;

				for (int j = 0; j < N_matrix_rows; j++) 
				{
					sum+= acc_w[i * N_matrix_rows + j] * acc_x[j];
				}

				acc_xout[i] = sum;
			});

#endif

        });

    } // End of SYCL Buffer Scope

}
/*
**  End of File
*/