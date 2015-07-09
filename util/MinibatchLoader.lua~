
-- Modified from https://github.com/karpathy/char-rnn.git  util/MinibatchLoader.lua
		-- Originally modified from https://github.com/oxford-cs-ml-2015/practical6
		-- the modification included support for train/val/test splits

-- Everything involving vocabulary for text generation is commented out since it is not needed for our tracking model

local MinibatchLoader = {}
MinibatchLoader.__index = MinibatchLoader
	-- I don't quite understand this line.  Apparently it sets the lookup table for class MinibatchLoader - Shawn Rigdon

function MinibatchLoader.create(data_dir, batch_size, seq_length, split_fractions, input_size)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, MinibatchLoader)

    local input_file = path.join(data_dir, 'input.txt')
    --local vocab_file = path.join(data_dir, 'vocab.t7')
	--local tensor_file = path.join(data_dir, 'data.t7')
	local data_file = path.join(data_dir, 'data.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (--[[path.exists(vocab_file) or path.exists(tensor_file))]] path.exists(data_file)) then
        -- prepro files do not exist, generate them
        --print('vocab.t7 and data.t7 do not exist. Running preprocessing...')
		print('data.t7 does not exist. Running preprocessing...')
		run_prepro = true
    else
        -- check if the input file was modified since last time we 
        -- ran the prepro. if so, we have to rerun the preprocessing
        local input_attr = lfs.attributes(input_file)
        --local vocab_attr = lfs.attributes(vocab_file)
        --local tensor_attr = lfs.attributes(tensor_file)
		local data_attr = lfs.attributes(data_file)
        if --[[input_attr.modification > vocab_attr.modification or]] input_attr.modification > --[[tensor]]data_attr.modification then
            --print('vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
			print('data.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end
    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing input text file ' .. input_file .. '...')
        --MinibatchLoader.text_to_tensor(input_file, vocab_file, tensor_file)
		MinibatchLoader.text_to_tensor(input_file, data_file, seq_length, input_size)
    end

    print('loading data files...')
    --local data = torch.load(tensor_file)
	local data = torch.load(data_file)
    --self.vocab_mapping = torch.load(vocab_file)

    -- cut off the end so that it divides evenly
    --local len = data:size(1)
	local len = data:size(2)
	--local len = #data		--UNCOMMENT IF USING TABLES FOR DATA
    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        --[[data = data:sub(1, batch_size * seq_length
                    * math.floor(len / (batch_size * seq_length)))]]
		data = data:sub(1, -1, 1, batch_size * seq_length
                    * math.floor(len / (batch_size * seq_length)))
		--[[uncomment if data is a table
		data = { unpack(data, 1, batch_size * seq_length
                    * math.floor(len / (batch_size * seq_length)) }]]
    end

	--[[    
	-- count vocab
    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do 
        self.vocab_size = self.vocab_size + 1 
    end
	--]]

    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    local ydata = data:clone()
	--ydata:sub(1,-2):copy(data:sub(2,-1))
    --ydata[-1] = data[1]		--why train the last character using the first?
	ydata:sub(1,-1,1,-2):copy(data:sub(1,-1,2,-1))
	ydata:sub(1,-1,-1,-1):copy(data:sub(1,-1,1,1))
	--[[uncomment if data is a table
	local ydata = { unpack(data,2,#data) }
	table.insert(ydata,data[1])]]

    --self.x_batches = data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
	self.x_batches = data:view(input_size, batch_size, -1):split(seq_length, 3)  -- #rows = #batches (3 for split along 3rd dim)
    self.nbatches = #self.x_batches
    --self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
	self.y_batches = ydata:view(input_size, batch_size, -1):split(seq_length, 3)  -- #rows = #batches
    assert(#self.x_batches == #self.y_batches)
	--[[
	I'm not sure exactly what is going on here with creating the batches.  A couple of questions:
		1) Should a batch be a sequential segment of the data?  Using this method, a batch contains some sequential group of characters 
		   followed by another block later on in the sequence.
		2) What is the purpose of shifting the data (and wrapping the first element) to obtain y_batches?
			-- The purpose appears to be to use y_batches as the label set

		!!Something is likely going to have to change with the way the batches are segmented
	--]]

    -- lets try to be helpful here
    if self.nbatches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
    if split_fractions[3] == 0 then 
        -- catch a common special case where the user might not want a test set
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = self.nbatches - self.ntrain
        self.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = math.floor(self.nbatches * split_fractions[2])
        self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()
    return self
end

function MinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function MinibatchLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    return self.x_batches[ix], self.y_batches[ix]
end

-- *** STATIC method ***																					!!Redo this function to generate tensor from txt files containing box locations!!
function MinibatchLoader.text_to_tensor(in_textfile, --[[out_vocabfile,]] out_datafile, seq_length, input_size) --out_tensorfile)
    local timer = torch.Timer()

    print('loading text file...')
    local file = torch.DiskFile(in_textfile)
    --local rawdata = f:readString('*a') -- NOTE: this reads the whole file at once
	
	-- record data in object hierarchy
	if not file:isQuiet() then file:quiet() end		--puts file in quiet mode so when last line is read program won't crash	

	local rawdata, f
	local objects, frames, frame_data = {},{},{}
	local end_of_seq = false
		
	while true do	-- break when done parsing input file

		rawdata = file:readString('*l')	-- read input one line at a time
		if file:hasError() then break end	-- last line has been read or file is empty

		-- put char sequences between spaces in a table of inputs 
		for num in rawdata:gmatch'%S+' do table.insert(frame_data, num) end -- '%S+' gets the complement of the set of space chars
		table.insert(frames, frame_data)
		frame_data = {}	-- clear data tabe for next frame
	
		f = #frames		-- frame number
		if f > 1 then
			--define end of sequence as a break in the frame count (end of object in sequence)
			if frames[f][1] + 0 ~= frames[f-1][1] + 1 then	--add zero to force char to double conversion
				
				if f < seq_length then	-- discard that object's data
					frames = {}
				else					-- record object data
					table.insert(objects, frames)
					frames = {}	-- clear frames for next object
				end

			elseif f == seq_length then
				table.insert(objects, frames)
				frames = {}

			end
		end

	end
	
	file:clearError()	--clear error generated when last readString is performed	
	file:close()

	
	--[[
    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')
    -- record all characters to a set
    local unordered = {}
    for char in rawdata:gmatch'.' do
        if not unordered[char] then unordered[char] = true end
    end
    -- sort into a table (i.e. keys become 1..N)
    local ordered = {}
    for char in pairs(unordered) do ordered[#ordered + 1] = char end
    table.sort(ordered)
    -- invert `ordered` to create the char->int mapping
    local vocab_mapping = {}
    for i, char in ipairs(ordered) do
        vocab_mapping[char] = i
    end
	--]]

	--[[
    -- construct a tensor with all the data
    print('putting data into tensor...')
    local data = torch.ByteTensor(#rawdata) -- store it into 1D first, then rearrange
    for i=1, #rawdata do
        data[i] = vocab_mapping[rawdata:sub(i, i)] -- lua has no string indexing using []
    end
	--]]


	--[[Unrolling into tensor for now
	-- UNROLLING OBJECTS INTO A TABLE OF BOUNDING BOXES FOR INPUTS (NO LONGER SAVING TENSOR)
	print('putting data into table...')
	local data = {}
	-- reformat frame data in data table
	for obj,_ in objects do
		for _,fData in ipairs(objects[obj]) do table.insert(data, fData) end
	end
	--]]

	--UNROLLING OBJECTS INTO 3D TENSOR
	print('putting data into tensor...')
	local data_size = #objects * seq_length
	local data = torch.zeros(input_size, data_size)	--bounding box data is rows, cols are all frames
	-- reformat frame data into tensor
	for obj=1, #objects do
		for f,fData in ipairs(objects[obj]) do
			data[{{},(obj-1)*seq_length + f}] = torch.Tensor({unpack(fData,7,10)})
		end
	end	

    -- save output preprocessed files
    --print('saving ' .. out_vocabfile)
    --torch.save(out_vocabfile, vocab_mapping)
    --print('saving ' .. out_tensorfile)
    --torch.save(out_tensorfile, data)
	print('saving ' .. out_datafile)
	torch.save(out_datafile, data)

end

--[[
-- create function to copy tables without having them linked in memory
local function deepCopy(original)
    local copy = {}
    for k, v in pairs(original) do
        -- if we find a table, make sure we copy that too
        if type(v) == 'table' then
            v = deepCopy(v)
        end
        copy[k] = v
    end
    return copy
end
--]]

return MinibatchLoader

