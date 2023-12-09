include("AES_Serial.jl")
using Base.Threads

global const BLOCK_BYTES = (num_blocks * block_rows)

## There are other ways to Run AES, but I didn't have time to run those
# converts to proper types for AES in ECB mode
function AESECB(blocks::String, key::String, encrypt::Bool)
	bytes2hex(AESECB(hex2bytes(blocks), hex2bytes(key), encrypt))
end

# executes AES in ECB mode 
function AESECB(blocks::Array{UInt8, 1}, key::Array{UInt8, 1}, encrypt::Bool)
	noBlocks = paddedCheck(blocks, key)
	o = Array{UInt8}(undef, length(blocks))

	gran = 100
	(num_keys, num_rounds) = AESParameters(key)
	w = KeyExpansion(key, num_keys, num_rounds)
	@threads for i in 1:convert(Int, ceil( noBlocks/ gran))
		for j in ((i-1) * gran + 1) : min(i * gran, noBlocks)
			local indices = blockIndices(blocks, j)
			o[indices] = encrypt ? AESEncrypt_modes(blocks[indices], w, num_rounds) : AESDecrypt_modes(blocks[indices], w, num_rounds)
		end
	end

	return o
end

# Returns the indices of the bytes of a block 
function blockIndices(blocks::Array{UInt8, 1}, blockNumber::Int)
	@assert(blockNumber >= 1)
	((blockNumber - 1) * BLOCK_BYTES + 1):(min(blockNumber * BLOCK_BYTES, length(blocks)))
end

# Checks whether the parameters are OK for ECB
function paddedCheck(blocks::Array{UInt8, 1}, key::Array{UInt8, 1})
	noBlocks = div(length(blocks), BLOCK_BYTES)
	if (noBlocks < 1) || ((noBlocks * BLOCK_BYTES) != length(blocks))
		error("No blocks or length of blocks is not a multplile of ",
		"16!")
	end
	# Check if key is OK
	AESParameters(key)
	return noBlocks
end