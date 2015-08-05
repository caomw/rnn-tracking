INPUT_FILE = '../data/kitti/input.txt'
OUTPUT_FILE = '../output/input_kitti_format.txt'

input = torch.DiskFile(INPUT_FILE)
output = torch.DiskFile(OUTPUT_FILE, 'w')

if not input:isQuiet() then input:quiet() end

while true do

	data = input:readString('*l')
	if input:hasError() then break end

	output:writeString(data:gsub('kitti 19 ', '')..'\n')

end

input:clearError()
input:close()
output:close()
