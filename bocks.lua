--test

while true do
  
    local defilement  = memory.readbyte(0x071C)
    defilement = (defilement % 256) % 16

    emu.frameadvance()

    gui.drawRectangle(0, 0, 16-defilement, 8, "red")
    gui.drawRectangle(256-defilement, 0, defilement-1, 8, "red")

    gui.drawRectangle(0, 216, 16-defilement, 7, "red")
    gui.drawRectangle(256-defilement, 216, defilement-1, 7, "red")

    
    for y = 1 , 13 do
        gui.drawRectangle(256-defilement, 216-y*16, defilement-1, 16, "red")
        gui.drawRectangle(0, 216-y*16, 16-defilement, 16, "red")
        for i = 1 , 15 do
            gui.drawRectangle(256-defilement-i*16, 0, 16, 8, "red")
            gui.drawRectangle(256-defilement-i*16, 216-y*16, 16, 16, "red")
            gui.drawRectangle(256-defilement-i*16, 216, 16, 7, "red")
        end
    end


end