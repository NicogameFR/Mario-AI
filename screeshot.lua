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
function Mario(Type, mariox1, marioy1, mariox2, marioy2)
	local marioX = ((mariox1 + mariox2)/2)/255
    local marioY = ((marioy1 + marioy2)/2)/223
    local marioX_width = (mariox2 - mariox1)/255
    local marioY_height= (marioy2 - marioy1)/223
	ecritureFichier(Type, marioX, marioY, marioX_width, marioY_height)
end

function dessin(x1, y1, x2, y2)
    local X_width = (x2 - x1)
    local Y_height= (y2 - y1)
    gui.drawRectangle(x1, y1, X_width, Y_height, "white")
end


nb = 0



function main()
    while true do
    
	--for i = 1 , 100 do
    --    
    --    emu.frameadvance()
    --    if nb ~= 0 or nb ~= 1 then
    --        saveScreenshot("frame_" .. nb)
    --    end
        nb = nb + 1

		local mariox1 = memory.readbyte(0x04AC)
		local mariox2 = memory.readbyte(0x04AE)
		local marioy1 = memory.readbyte(0x04AD)-9
		local marioy2 = memory.readbyte(0x04AF)-9
		local marioType = memory.readbyte(0x0756)
		
        local enemy1 = memory.readbyte(0x000F)
        local enemy2 = memory.readbyte(0x0010)
        local enemy3 = memory.readbyte(0x0011)
        local enemy4 = memory.readbyte(0x0012)
        local enemy5 = memory.readbyte(0x0013)

        local fireball1 = memory.readbyte(0x0024)
        local fireball2 = memory.readbyte(0x0025)

        if enemy1 == 1 then
            enemy1x1 = memory.readbyte(0x04B0)
            enemy1x2 = memory.readbyte(0x04B2)
            enemy1y1 = memory.readbyte(0x04B1)-9
            enemy1y2 = memory.readbyte(0x04B3)-9
            enemy1Type = memory.readbyte(0x0016)
        end

        if enemy2 == 1 then
            enemy2x1 = memory.readbyte(0x04B4)
            enemy2x2 = memory.readbyte(0x04B6)
            enemy2y1 = memory.readbyte(0x04B5)-9
            enemy2y2 = memory.readbyte(0x04B7)-9
            enemy2Type = memory.readbyte(0x0017)
            
        end

        if enemy3 == 1 then
            enemy3x1 = memory.readbyte(0x04B8)
            enemy3x2 = memory.readbyte(0x04BA)
            enemy3y1 = memory.readbyte(0x04B9)-9
            enemy3y2 = memory.readbyte(0x04BB)-9
            enemy3Type = memory.readbyte(0x0018)
        end

        if enemy4 == 1 then
            enemy4x1 = memory.readbyte(0x04BC)
            enemy4x2 = memory.readbyte(0x04BE)
            enemy4y1 = memory.readbyte(0x04BD)-9
            enemy4y2 = memory.readbyte(0x04BF)-9
            enemy4Type = memory.readbyte(0x0019)
        end

        if enemy5 == 1 then
            enemy5x1 = memory.readbyte(0x04C0)
            enemy5x2 = memory.readbyte(0x04C2)
            enemy5y1 = memory.readbyte(0x04C1)-9
            enemy5y2 = memory.readbyte(0x04C3)-9
            enemy5Type = memory.readbyte(0x001A)
        end
        
        if fireball1 == 1 then
            fireball1x1 = memory.readbyte(0x04C8)
            fireball1x2 = memory.readbyte(0x04CA)
            fireball1y1 = memory.readbyte(0x04C9)-9
            fireball1y2 = memory.readbyte(0x04CB)-9
        end

        if fireball2 == 1 then
            fireball2x1 = memory.readbyte(0x04CC)
            fireball2x2 = memory.readbyte(0x04CE)
            fireball2y1 = memory.readbyte(0x04CD)-9
            fireball2y2 = memory.readbyte(0x04CF)-9
        end
        emu.frameadvance()
		if marioType == 0 then
            dessin(mariox1-4, marioy1-3, mariox2+3, marioy2+2)
			--Mario(marioType, mariox1-4, marioy1-3, mariox2+3, marioy2+2)
		end
		if marioType ~= 0 then
			if marioType ~= 1 then
				marioType = 2
		    end
			dessin(mariox1-3, marioy1-7, mariox2+2, marioy2+2)
			--Mario(marioType, mariox1-3, marioy1-7, mariox2+2, marioy2+2)
		end

        if enemy1 == 1 then
            if enemy1Type == 0 then
                koopa(enemy1Type, enemy1x1, enemy1y1, enemy1x2, enemy1y2)
            end
            if enemy1Type == 6 then
                goomba(enemy1Type, enemy1x1, enemy1y1, enemy1x2, enemy1y2)
            end
        end
        if enemy2 == 1 then
            if enemy2Type == 0 then
                koopa(enemy2Type, enemy2x1, enemy2y1, enemy2x2, enemy2y2)
            end
            if enemy2Type == 6 then
                goomba(enemy2Type, enemy2x1, enemy2y1, enemy2x2, enemy2y2)
            end
        end
        if enemy3 == 1 then
            if enemy3Type == 0 then
                koopa(enemy3Type, enemy3x1, enemy3y1, enemy3x2, enemy3y2)
            end
            if enemy3Type == 6 then
                goomba(enemy3Type, enemy3x1, enemy3y1, enemy3x2, enemy3y2)
            end
        end
        if enemy4 == 1 then
            if enemy4Type == 0 then
                koopa(enemy4Type, enemy4x1, enemy4y1, enemy4x2, enemy4y2)
            end
            if enemy4Type == 6 then
                goomba(enemy4Type, enemy4x1, enemy4y1, enemy4x2, enemy4y2)
            end
        end
        if enemy5 == 1 then
            if enemy5Type == 0 then
                koopa(enemy5Type, enemy5x1, enemy5y1, enemy5x2, enemy5y2)
            end
            if enemy5Type == 6 then
                goomba(enemy5Type, enemy5x1, enemy5y1, enemy5x2, enemy5y2)
            end
        end
        if fireball1 == 1 then
            fireball(fireball1x1, fireball1y1, fireball1x2, fireball1y2)
        end

        if fireball2 == 1 then
            fireball(fireball2x1, fireball2y1, fireball2x2, fireball2y2)
        end
    end
end

function koopa(enemyType, enemyx1, enemyy1, enemyx2, enemyy2)
    dessin(enemyx1-3, enemyy1-7, enemyx2+2, enemyy2+5)
end

function goomba(enemyType, enemyx1, enemyy1, enemyx2, enemyy2)
    dessin(enemyx1-4, enemyy1-5, enemyx2+3, enemyy2+6)
end

function fireball(fireballx1, firebally1, fireballx2, firebally2)
    dessin(fireballx1, firebally1+1, fireballx2, firebally2+2)

end


deleteOldFiles()
main()