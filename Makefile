CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -march=native
NVCCFLAGS = -O3 -arch=sm_86

all: packer unpacker gpu_bench stream_verify

packer: src/zbf16_packer.cpp
	$(CXX) $(CXXFLAGS) src/zbf16_packer.cpp -o zbf16_packer

unpacker: src/zbf16_unpacker.cpp
	$(CXX) $(CXXFLAGS) src/zbf16_unpacker.cpp -o zbf16_unpacker

gpu_bench: src/gpu_benchmark.cu
	$(NVCC) $(NVCCFLAGS) src/gpu_benchmark.cu -o gpu_benchmark

stream_verify: src/full_model_stream_verify.cu
	$(NVCC) $(NVCCFLAGS) src/full_model_stream_verify.cu -o full_model_stream_verify

clean:
	rm -f zbf16_packer zbf16_unpacker gpu_benchmark full_model_stream_verify
