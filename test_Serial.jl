using Test
using BenchmarkTools

include("AES_Serial_Modes.jl")

## Speed tests

key4 = "2b7e151628aed2a6abf7158809cf4f3c" #use this for all speed tests
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


function bench_cpu(plaintext, key)
    AESECB(plaintext, key4, true)
end

println("CPU timing for a plaintext of length 2^8 bytes: ")
@btime bench_cpu(plaintext_2, key4)
println("\n")

println("CPU timing for a plaintext of length 2^12 bytes: ")
@btime bench_cpu(plaintext_3, key4)
println("\n")

println("CPU timing for a plaintext of length 2^16 bytes: ")
@btime bench_cpu(plaintext_4, key4)
println("\n")

println("CPU timing for a plaintext of length 2^20 bytes: ")
@btime bench_cpu(plaintext_5, key4)
println("\n")

println("CPU timing for a plaintext of length 2^24 bytes: ")
@btime bench_cpu(plaintext_6, key4)
println("\n")

print("")



