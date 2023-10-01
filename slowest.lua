while true do
    emu.frameadvance()
    local slowest = 0
    for i =  1 , 30000000 do
        slowest = slowest + i
    end
end