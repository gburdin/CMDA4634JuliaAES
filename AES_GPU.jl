## Author: Garrett Burdin
## Class: CMDA3634
## Assignment: Final Project


using CUDA

include("Galois_GPU.jl")

# Code Begin ---------------------

## DECRYPTION CODE IS STILL IN PROGRESS**

global const sub_box = CuArray([ # lookup table for encrypt sub step
0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
])
global const sub_box_cuda = cudaconvert(sub_box) 

global const inv_sub_box = CuArray([ # lookup table for decrypt sub step
0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
])

global const mix_col_mat = CuArray([ # lookup table for encrypt mixcols step
0x02, 0x03, 0x01, 0x01,
0x01, 0x02, 0x03, 0x01,
0x01, 0x01, 0x02, 0x03,
0x03, 0x01, 0x01, 0x02
])

global const mix_col_mat_cuda = cudaconvert(mix_col_mat)

global const inv_mix_col_mat = cudaconvert(CuArray([ #lookup table for decrypt mixcols step
0x0e, 0x0b, 0x0d, 0x09,
0x09, 0x0e, 0x0b, 0x0d,
0x0d, 0x09, 0x0e, 0x0b,
0x0b, 0x0d, 0x09, 0x0e
]))

## AES will always be a 4x4 block. Set these global values
global const num_blocks = 4 # num blocks (for AES, not cuda)-- can think of as num of cols
global const block_rows = 4 # num of rows per block 

# Return (num_keys, num_rounds)
function AESParameters(key::CuArray{UInt8, 1})

    num_keys_lookup = CuArray([4, 6, 8]) #num keys
    num_rounds_lookup = CuArray([10, 12, 14]) #num rounds

	if mod(length(key), block_rows) != 0
		error("key length not a mult of 4")
	end

	num_keys = div(length(key), block_rows) # These lines find
	i = first(indexin([num_keys], num_keys_lookup)) # How many bits your AES key has (128, 196, 256)

	if i == 0 # checks to make sure our key size is proper bits (0 is "out of bounds")
		error("key length is not correct bit size")
	end

	num_rounds = num_rounds_lookup[i] # checking the globally defined arrays at the top of program
	return (num_keys, num_rounds)
end

# main encryption function
# will run all the substeps of AES in here, while calling other functions
function AESEncrypt(ct::CuDeviceVector{UInt8, 1}, plain::CuDeviceVector{UInt8, 1}, begin_ind::Int, end_ind::Int, num_rounds::Int, w::CuDeviceVector{UInt8, 1}, buffer::CuDeviceVector{UInt8, 1})
    # basic checks
	@assert((end_ind - begin_ind + 1) == (block_rows * num_blocks))
	@assert(length(w) == (block_rows * num_blocks * (num_rounds + 1)))

	# Copy
	for i=begin_ind:end_ind
		ct[i] = plain[i]
	end

	# AddRoundKey (ONLY for the 1st round do you add here)
	for i=begin_ind:end_ind
		ct[i] = gadd(ct[i], w[i - begin_ind + 1])
	end

	for round=1:(num_rounds-1)
		# SubBytes
		for i=begin_ind:end_ind
			ct[i] = sub_box_cuda[Int(ct[i]) + 1]
		end

		# ShiftRows
		for r=2:num_blocks
			step = r
			cnt = 0
			for index = r:num_blocks:num_blocks*num_blocks
				buffer[begin_ind + index - 1] = ct[begin_ind + index - 1]
			end
			for index=r:num_blocks:num_blocks*num_blocks
				p = mod(cnt + step - 1, num_blocks) + 1
				p_index = r + num_blocks * (p - 1)
				ct[begin_ind + index - 1] = buffer[begin_ind + p_index - 1]
				cnt += 1
			end
		end

 		# MixColumns
		for c=1:num_blocks
			for index=((c - 1) * num_blocks + 1):(c * num_blocks)
				buffer[begin_ind + index - 1] = ct[begin_ind + index - 1]
			end
			for r=1:num_blocks
				indices_r = ((c - 1) * num_blocks + 1) + r - 1
				for j=((r - 1) * num_blocks + 1):(r * num_blocks)
					mij = mix_col_mat_cuda[j]
					nth_element = j - ((r - 1) * num_blocks + 1) # 0-indexed
					aij = buffer[begin_ind + ((c - 1) * num_blocks + 1) - 1 + nth_element]
					res = gmul2(aij, mij)
					if j == ((r - 1) * num_blocks + 1)
						ct[begin_ind + indices_r - 1] = res
					else
						ct[begin_ind + indices_r - 1] = gadd(ct[begin_ind + indices_r - 1], res)
					end
				end
			end
		end

 		# AddRoundKey
		for i=begin_ind:end_ind
			ct[i] = gadd(ct[i], w[i - begin_ind + (round * num_blocks * block_rows + 1)])
		end

	end

 	# SubBytes(state)
	for i=begin_ind:end_ind
		ct[i] = sub_box_cuda[Int(ct[i]) + 1]
	end

 	# ShiftRows(state)
	for r=2:num_blocks
		step = r
		cnt = 0
		for index=r:num_blocks:num_blocks*num_blocks
			buffer[begin_ind + index - 1] = ct[begin_ind + index - 1]
		end
		for index=r:num_blocks:num_blocks*num_blocks
			p = mod(cnt + step - 1, num_blocks) + 1
			p_index = r + num_blocks * (p - 1)
			ct[begin_ind + index - 1] = buffer[begin_ind + p_index - 1]
			cnt += 1
		end
	end

 	# AddRoundKey(state, w[(num_rounds * num_blocks * block_rows + 1):((num_rounds + 1) * num_blocks * block_rows)])
	for i=begin_ind:end_ind
		ct[i] = gadd(ct[i], w[i - begin_ind + (num_rounds * num_blocks * block_rows + 1)])
	end

end

function AESDecrypt(ct::CuDeviceVector{UInt8, 1}, cipher::CuDeviceVector{UInt8, 1}, key::CuDeviceVector{UInt8, 1}, begin_ind::Int, end_ind::Int, num_keys::Int, num_rounds::Int, w::CuDeviceVector{UInt8, 1}, buffer::CuDeviceVector{UInt8, 1})
	# (w, num_rounds) = AEScrypt(cipher, key, num_keys, num_rounds)
	# return AESInvCipher(cipher, w, num_rounds)
	return cipher #UNFINISHED ---------------------
end

function KeyScheduling!(w::CuArray{UInt8, 1}, key::CuArray{UInt8, 1}, num_keys::Int, num_rounds::Int)
	@assert(length(key) == (block_rows * num_keys))

	w[1:(block_rows * num_keys)] = copy(key)
	i = num_keys

	while i < (num_blocks * (num_rounds + 1))
		temp = w[((i - 1) * block_rows + 1):(i * block_rows)]
		if mod(i, num_keys) == 0
			temp = xor.(SubWord(RotWord(temp)), Rcon(div(i, num_keys)))
		elseif (num_keys > 6) && (mod(i, num_keys) == num_blocks)
			temp = SubWord(temp)
		end
		w[(i * block_rows + 1):((i + 1) * block_rows)] = xor.(w[((i - num_keys) * block_rows + 1):((i - num_keys + 1) * block_rows)] , temp)
		i += 1
	end

	return nothing
end

function SubWord(w::CuArray{UInt8, 1})
	@assert(length(w) == block_rows)
	# map!(x -> sub_box[Int(x) + 1], w, w)
	# return w
	for i=1:length(w)
		w[i] = sub_box[Int(w[i]) + 1]
	end
	return w
end

function RotWord(w::CuArray{UInt8, 1})
	@assert(length(w) == block_rows)
	# permute!(w, [2, 3, 4, 1])
	temp = w[1]
	w[1] = w[2]
	w[2] = w[3]
	w[3] = w[4]
	w[4] = temp
    return w
end

function Rcon(i::Int)
	@assert(i > 0)
	x = 0x01
	for j=1:(i-1)
		x = gmul(x, 0x02)
	end
	return CuArray([x, 0x00, 0x00, 0x00])
end

# RUN AES IN ECB MODE (main mode for parallelization) ------------------
global const BLOCK_BYTES = (num_blocks * block_rows) # 4x4 = 16

function AES(blocks::String, key::String, encrypt::Bool)
    blocks_cuarray, key_cuarray = CUDA.allowscalar() do # Convert block (PT) and key from hex to byte (Uint8) format
        blocks_cuarray = CuArray(hex2bytes(blocks)) # then make both of them cuda arrays
        key_cuarray = CuArray(hex2bytes(key)) # will be of length 16, 24, 32 (for 128, 196, 256 bits)
        return blocks_cuarray, key_cuarray
    end
    cipher_cuarray = AES(blocks_cuarray, key_cuarray, encrypt) # pass cuda arrays into 
	return cipher_cuarray 
end

# Can override functions in Julia
function AES(blocks::CuArray{UInt8, 1}, key::CuArray{UInt8, 1}, encrypt::Bool)
    noBlocks, num_keys, num_rounds = CUDA.allowscalar() do # scalar indexing in Julia Cuda is disallowed by default
        return paddedCheck(blocks, key) #find num of blocks, number of keys, and number of rounds(?)
    end
	w = CuArray{UInt8, 1}(undef, block_rows * num_blocks * (num_rounds + 1)) # key scheduling initialization
	CUDA.allowscalar() do
		KeyScheduling!(w, key, num_keys, num_rounds) #convert given key to a set of round keys using key scheduling
	end
	
	ct = CuArray{UInt8, 1}(undef, length(blocks)) #undefined array of length blocks (PT)
	buffer = CuArray{UInt8, 1}(undef, length(blocks))

    # hardcode threads: find block number (for GPU)
	numblocks_cuda = ceil(Int, length(blocks) / BLOCK_BYTES / 256) # get number of blocks for CUDA (not for AES)
    @cuda threads=256 blocks=numblocks_cuda AES_run!(cudaconvert(ct), noBlocks, cudaconvert(blocks), cudaconvert(key), encrypt, num_keys, num_rounds, cudaconvert(w), cudaconvert(buffer))

	return ct
end

# ! in julia is a notation to denote that a value will be changed, typically the 1st one
function AES_run!(ct::CuDeviceVector{UInt8, 1}, noBlocks::Int, blocks::CuDeviceVector{UInt8, 1}, key::CuDeviceVector{UInt8, 1}, encrypt::Bool, num_keys::Int, num_rounds::Int, w::CuDeviceVector{UInt8, 1}, buffer::CuDeviceVector{UInt8, 1})
	# Julia GPU convention uses "index" and "stride" for each thread. This is just a convention. 
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x #use blockIdx().x-1 because julia starts indexing at 1
    stride = gridDim().x * blockDim().x # this is how much you'll "jump" each thread
    
    for i = index:stride:noBlocks # go from "index" to "noBlocks", iterating by "stride"
		indices_begin = (i - 1) * BLOCK_BYTES + 1 
		indices_end = min(i * BLOCK_BYTES, length(blocks)) #works out so that each thread will execute 1 AES Block
		@assert(indices_end - indices_begin <= 16)
		encrypt ? AESEncrypt(ct, blocks, indices_begin, indices_end, num_rounds, w, buffer) : AESDecrypt(ct, blocks, key, indices_begin, indices_end, num_keys, num_rounds, w, buffer)
	end
    return nothing
end

function paddedCheck(blocks::CuArray{UInt8, 1}, key::CuArray{UInt8, 1})
	noBlocks = div(length(blocks), BLOCK_BYTES) #number of blocks 
	if (noBlocks < 1) || ((noBlocks * BLOCK_BYTES) != length(blocks)) #checking that block length is mult of 16
		error("Number of blocks or length of blocks is not a multplile of 16!")
	end
	# Check if key is OK
	num_keys, num_rounds = AESParameters(key) #given the key, this will determine num_keys and num_rounds by the key length (bit size)
	return noBlocks, num_keys, num_rounds
end








