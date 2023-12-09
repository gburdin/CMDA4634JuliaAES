include("AES_GPU.jl")
using BenchmarkTools
using CUDA
using Test

# MAIN -------------------

# checking that the encryption works properly

println("testing for GPU below")

const key1 = "2b7e151628aed2a6abf7158809cf4f3c"
const plain1 = "3243f6a8885a308d313198a2e0370734"
const cipher1 = "3925841d02dc09fbdc118597196a0b32"

CUDA.allowscalar() do 
    println(bytes2hex(AES(plain1, key1, true)) == cipher1)
end


###### Speed Test
key2 = "2b7e151628aed2a6abf7158809cf4f3c" #use this for all speed tests
plaintext_base = "A"

plaintext_2 = plaintext_base
while length(plaintext_2) < 2^8 # 256 bytes
    global plaintext_2
    plaintext_2 = plaintext_2 * plaintext_2
end

plaintext_3 = plaintext_base
while length(plaintext_3) < 2^12 # 4096 bytes
    global plaintext_3
    plaintext_3 = plaintext_3 * plaintext_3
end

plaintext_4 = plaintext_base
while length(plaintext_4) < 2^16 # 65536 bytes
    global plaintext_4
    plaintext_4 = plaintext_4 * plaintext_4
end

plaintext_5 = plaintext_base
while length(plaintext_5) < 2^20 # 1048576 bytes
    global plaintext_5
    plaintext_5 = plaintext_5 * plaintext_5
end

plaintext_6 = plaintext_base
while length(plaintext_6) < 2^24 # 16777216 bytes
    global plaintext_6
    plaintext_6 = plaintext_6 * plaintext_6
end


CUDA.allowscalar(false)

function bench_gpu(plaintext, key, encryption)
    CUDA.@sync begin
		AES(plaintext, key, encryption)
	end
end

println("GPU timing for a plaintext of length 2^8 bytes: ")
@btime bench_gpu(plaintext_2, key2, true)
print("\n")

println("GPU timing for a plaintext of length 2^12 bytes: ")
@btime bench_gpu(plaintext_3, key2, true)
print("\n")

println("GPU timing for a plaintext of length 2^16 bytes: ")
@btime bench_gpu(plaintext_4, key2, true)
print("\n")

println("GPU timing for a plaintext of length 2^20 bytes: ")
@btime bench_gpu(plaintext_5, key2, true)
print("\n")

println("GPU timing for a plaintext of length 2^24 bytes: ")
@btime bench_gpu(plaintext_6, key2, true)
print("\n")
