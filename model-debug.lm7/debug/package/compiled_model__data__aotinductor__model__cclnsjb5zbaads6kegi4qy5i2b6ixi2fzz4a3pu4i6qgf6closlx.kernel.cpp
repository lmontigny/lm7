// Triton kernels are embedded as comments in /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.cpp

#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  cpp_fused_relu_0(float* in_out_ptr0,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(64L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    tmp1.store(in_out_ptr0 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0L) && x0 < static_cast<int64_t>(4L)))
                {
                    for (int64_t x0_tail = static_cast<int64_t>(0L);x0_tail < static_cast<int64_t>(4L); x0_tail++)
                    {
                        auto tmp0 = x0_tail;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(2);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<int64_t>(1);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = static_cast<float>(-0.031050194054841995);
                        auto tmp7 = static_cast<float>(0.05729067325592041);
                        auto tmp8 = tmp5 ? tmp6 : tmp7;
                        auto tmp9 = static_cast<int64_t>(3);
                        auto tmp10 = tmp1 < tmp9;
                        auto tmp11 = static_cast<float>(0.069354347884655);
                        auto tmp12 = static_cast<float>(0.13573212921619415);
                        auto tmp13 = tmp10 ? tmp11 : tmp12;
                        auto tmp14 = tmp3 ? tmp8 : tmp13;
                        out_ptr0[static_cast<int64_t>(x0_tail)] = tmp14;
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}

// Compile cmd
// g++ /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/cclnsjb5zbaads6kegi4qy5i2b6ixi2fzz4a3pu4i6qgf6closlx.kernel.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2  -O3 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=native -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -include /tmp/torchinductor_bernard/precompiled_headers/cl6erclal2xd46ads3oghhfycczy2niolj6wep34yfjhx7jf6ywj.h -I/usr/include/python3.12 -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include   -mavx2 -mfma -mf16c  -c -o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/cclnsjb5zbaads6kegi4qy5i2b6ixi2fzz4a3pu4i6qgf6closlx.kernel.o
// Link cmd
// g++ /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/cclnsjb5zbaads6kegi4qy5i2b6ixi2fzz4a3pu4i6qgf6closlx.kernel.o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi/ctrsxmmxgotwt2gcantv6qe5g4hugda73ji2v3ijip227636shte.weights.o -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2  -O3 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=native -shared -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -I/usr/include/python3.12 -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include   -mavx2 -mfma -mf16c  -o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.so  -ltorch -ltorch_cpu -lgomp  -L/usr/lib/x86_64-linux-gnu -L/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/lib 

// Compile cmd
// g++ /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/cclnsjb5zbaads6kegi4qy5i2b6ixi2fzz4a3pu4i6qgf6closlx.kernel.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2  -O3 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=native -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -include /tmp/torchinductor_bernard/precompiled_headers/cl6erclal2xd46ads3oghhfycczy2niolj6wep34yfjhx7jf6ywj.h -I/usr/include/python3.12 -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include   -mavx2 -mfma -mf16c  -c -o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/cclnsjb5zbaads6kegi4qy5i2b6ixi2fzz4a3pu4i6qgf6closlx.kernel.o
// Link cmd
// g++ /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/cclnsjb5zbaads6kegi4qy5i2b6ixi2fzz4a3pu4i6qgf6closlx.kernel.o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi/ctrsxmmxgotwt2gcantv6qe5g4hugda73ji2v3ijip227636shte.weights.o -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2  -O3 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=native -shared -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -I/usr/include/python3.12 -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include   -mavx2 -mfma -mf16c  -o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.so  -ltorch -ltorch_cpu -lgomp  -L/usr/lib/x86_64-linux-gnu -L/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/lib 

// Compile cmd
// g++ /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/cclnsjb5zbaads6kegi4qy5i2b6ixi2fzz4a3pu4i6qgf6closlx.kernel.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2  -O3 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=native -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -include /tmp/torchinductor_bernard/precompiled_headers/cl6erclal2xd46ads3oghhfycczy2niolj6wep34yfjhx7jf6ywj.h -I/usr/include/python3.12 -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include   -mavx2 -mfma -mf16c  -c -o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/cclnsjb5zbaads6kegi4qy5i2b6ixi2fzz4a3pu4i6qgf6closlx.kernel.o
// Link cmd
// g++ /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/cclnsjb5zbaads6kegi4qy5i2b6ixi2fzz4a3pu4i6qgf6closlx.kernel.o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi/ctrsxmmxgotwt2gcantv6qe5g4hugda73ji2v3ijip227636shte.weights.o -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2  -O3 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=native -shared -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -I/usr/include/python3.12 -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include   -mavx2 -mfma -mf16c  -o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.so  -ltorch -ltorch_cpu -lgomp  -L/usr/lib/x86_64-linux-gnu -L/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/lib 
