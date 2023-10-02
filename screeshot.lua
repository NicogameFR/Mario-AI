-- racio: 1024:768
gui.clearGraphics()
marioOldY = 100000

function deleteOldFiles()
	local command_1 = "cd ./database/images/ & del *.png"
	os.execute(command_1)
	local command_2 = "cd ./database/labels/ & del *.txt"
	os.execute(command_2)
end

function saveScreenshot(nom)
    local command_1 = "nircmd.exe savescreenshot ./database/images/" .. nom .. ".png 80 46 863 659"
    local command_2 = 'nircmd.exe win hide foreground & ' .. command_1 .. '& exit'
    os.execute(command_2)
    attendreCreationFichier(nom .. ".png")
end

function attendreCreationFichier(nom)
    local chemin = "./database/images/" .. nom
    local timeouts = 0
    while not io.open(chemin, "r") do
        timeouts = timeouts + 1
        print("attente")
        if timeouts > 100 then
            return
        end
    end
end

-- fonction d'ecriture pour n'importe quel objet
function ecritureFichier(Type, marioX, marioY, marioX_width, marioY_height)
    local fichier = io.open("./database/labels/frame_".. nb+1 ..".txt", "w")
    fichier:write(Type .. " " .. marioX .. " " .. marioY .. " " .. marioX_width .. " " .. marioY_height)
    fichier:close()
end

-- écriture dans le fichier pour l'objet mario
function ecritureMarioFichier(Type, mariox1, marioy1, mariox2, marioy2)
	local marioX = ((mariox1 + mariox2)/2)/255
    local marioY = ((marioy1 + marioy2)/2)/223
    local marioX_width = (mariox2 - mariox1)/255
    local marioY_height= (marioy2 - marioy1)/223
	ecritureFichier(Type, marioX, marioY, marioX_width, marioY_height)
end

function dessin(mariox1, marioy1, mariox2, marioy2)
    local marioX_width = (mariox2 - mariox1)
    local marioY_height= (marioy2 - marioy1)
    gui.drawRectangle(mariox1, marioy1, marioX_width, marioY_height, "red")
end


nb = 0
function main()
	for i = 1 , 100 do
        
        emu.frameadvance()
        if nb ~= 0 or nb ~= 1 then
            saveScreenshot("frame_" .. nb)
        end
        nb = nb + 1
		local mariox1 = memory.readbyte(0x04AC)
		local mariox2 = memory.readbyte(0x04AE)
		local marioy1 = memory.readbyte(0x04AD)-9
		local marioy2 = memory.readbyte(0x04AF)-9
		local marioType = memory.readbyte(0x0756)
		-- Pour mario petit
		if marioType == 0 then
			-- Pour mario petit en saut
            --dessin(mariox1-4, marioy1-3, mariox2+3, marioy2+2)
			ecritureMarioFichier(marioType, mariox1-4, marioy1-3, mariox2+3, marioy2+2)
			end
		-- Pour mario grand
		if marioType ~= 0 then
			if marioType ~= 1 then
				marioType = 2
			end
			--dessin(mariox1-3, marioy1-7, mariox2+2, marioy2+2)
			ecritureMarioFichier(marioType, mariox1-3, marioy1-7, mariox2+2, marioy2+2)
		end
	end
end

deleteOldFiles()
main()