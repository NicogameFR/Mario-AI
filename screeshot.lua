-- racio: 1024:768


forecolor = nil -- Par défaut, la couleur du texte est blanche
backcolor = nil -- Par défaut, l'arrière-plan est transparent
fontfamily = nil -- Par défaut, la famille de police est monospace générique
fontstyle = "regular" -- Par défaut, le style de police est "regular"
horizalign = "left" -- Par défaut, l'alignement horizontal est "left"
vertalign = "bottom" -- Par défaut, l'alignement vertical est "bottom"
surfacename = nil -- Le nom de la surface si nécess


function deleteOldFiles()
    local command_1 = "cd ./database/images/ & del *.png"
    os.execute(command_1)
    local command_2 = "cd ./database/labels/ & del *.txt"
    os.execute(command_2)
end

function saveScreenshot(nom)
    local commande = "nircmd.exe win hide foreground & " .. "nircmd.exe savescreenshot ./database/images/" .. nom ..
                         ".png 80 46 863 659 " .. '& exit'
    os.execute(commande)
    local chemin = "./database/images/" .. nom .. ".png"
    local timeouts = 0
    while not io.open(chemin, "r") do
        timeouts = timeouts + 1
        print("attente")
        if timeouts > 50 then
            return
        end
    end
end

function ecritureFichier(Type, X, Y, X_width, Y_height)
    if nb + 1 > nb_image - 2 then
        return
    end
    local fichier = io.open("./database/labels/frame_" .. nb + 1 .. ".txt", "a+")
    fichier:write(Type .. " " .. X .. " " .. Y .. " " .. X_width .. " " .. Y_height .. "\n")
    fichier:close()
end

function dessin(x1, y1, x2, y2)
    local X_width = (x2 - x1)
    local Y_height = (y2 - y1)
    gui.drawRectangle(x1, y1, X_width, Y_height, "white")
end

fontsize = 9 -- Par défaut, la taille de la police est de 12




function quadrillage()
    
    for i = 0, 14 do
        if i == 0 then
            gui.drawRectangle(0, 0, 15 - defilement, 7, "red")
        elseif i == 14  then
            gui.drawRectangle(0,216,15 - defilement, 7, "red")
        else
            gui.drawRectangle(0, 8+16*(i-1), 15 - defilement, 15, "red")
        end
    end

    for i = 1 , 15 do 
        gui.drawRectangle(0+16*i-defilement,0,15,7,"red")
        gui.drawRectangle(0+16*i-defilement,216,15,7,"red")
    end

    for i = 0, 12 do
        for y = 1, 15 do
            gui.drawRectangle(0+16*y-defilement,8+16*i,15,15,"red")
        end
    end

    for i = 0 , 14 do 
        if i == 0 then
            gui.drawRectangle(256-defilement, 0, defilement-1, 7, "red")
        elseif i == 14  then
            gui.drawRectangle(256-defilement,216, defilement-1, 7, "red")
        else
            gui.drawRectangle(256-defilement, 8+16*(i-1), defilement-1, 15, "red")
        end
    end


end







function choix()
    if affichage == 1 then
        while true do
            main()
        end
    elseif affichage == 0 then
        nb = -1
        for i = 1, nb_image do --
            emu.frameadvance() --
            if nb > 0 then --
                saveScreenshot("frame_" .. nb) --
            end
            nb = nb + 1
            main()
        end
    end
end

function main()
    defilement = (memory.readbyte(0x071C)) % 16

    local mariox1 = memory.readbyte(0x04AC)
    local mariox2 = memory.readbyte(0x04AE)
    local marioy1 = memory.readbyte(0x04AD) - 9
    local marioy2 = memory.readbyte(0x04AF) - 9
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
        enemy1y1 = memory.readbyte(0x04B1) - 9
        enemy1y2 = memory.readbyte(0x04B3) - 9
        enemy1Type = memory.readbyte(0x0016)
    end

    if enemy2 == 1 then
        enemy2x1 = memory.readbyte(0x04B4)
        enemy2x2 = memory.readbyte(0x04B6)
        enemy2y1 = memory.readbyte(0x04B5) - 9
        enemy2y2 = memory.readbyte(0x04B7) - 9
        enemy2Type = memory.readbyte(0x0017)

    end

    if enemy3 == 1 then
        enemy3x1 = memory.readbyte(0x04B8)
        enemy3x2 = memory.readbyte(0x04BA)
        enemy3y1 = memory.readbyte(0x04B9) - 9
        enemy3y2 = memory.readbyte(0x04BB) - 9
        enemy3Type = memory.readbyte(0x0018)
    end

    if enemy4 == 1 then
        enemy4x1 = memory.readbyte(0x04BC)
        enemy4x2 = memory.readbyte(0x04BE)
        enemy4y1 = memory.readbyte(0x04BD) - 9
        enemy4y2 = memory.readbyte(0x04BF) - 9
        enemy4Type = memory.readbyte(0x0019)
    end

    if enemy5 == 1 then
        enemy5x1 = memory.readbyte(0x04C0)
        enemy5x2 = memory.readbyte(0x04C2)
        enemy5y1 = memory.readbyte(0x04C1) - 9
        enemy5y2 = memory.readbyte(0x04C3) - 9
        enemy5Type = memory.readbyte(0x001A)
    end

    if fireball1 == 1 then
        fireball1x1 = memory.readbyte(0x04C8)
        fireball1x2 = memory.readbyte(0x04CA)
        fireball1y1 = memory.readbyte(0x04C9) - 9
        fireball1y2 = memory.readbyte(0x04CB) - 9
    end

    if fireball2 == 1 then
        fireball2x1 = memory.readbyte(0x04CC)
        fireball2x2 = memory.readbyte(0x04CE)
        fireball2y1 = memory.readbyte(0x04CD) - 9
        fireball2y2 = memory.readbyte(0x04CF) - 9
    end

    if affichage == 1 then
        
        emu.frameadvance()
        quadrillage()
        if memory.readbyte(0x06A0)%2 == 0 then
            if (memory.readbyte(0x06A0)-24) >= 0 then
                dep = memory.readbyte(0x06A0)-24
            else
                dep = memory.readbyte(0x06A0)+8
            end
            gui.drawString(100, 100, dep, forecolor, backcolor, fontsize, fontfamily, fontstyle, horizalign, vertalign, surfacename)
        end
    end


    Mario(marioType, mariox1, marioy1, mariox2, marioy2)

    if enemy1 == 1 then
        if enemy1Type == 0 then
            koopa(enemy1x1, enemy1y1, enemy1x2, enemy1y2)
        end
        if enemy1Type == 6 then
            goomba(enemy1x1, enemy1y1, enemy1x2, enemy1y2)
        end
    end
    if enemy2 == 1 then
        if enemy2Type == 0 then
            koopa(enemy2x1, enemy2y1, enemy2x2, enemy2y2)
        end
        if enemy2Type == 6 then
            goomba(enemy2x1, enemy2y1, enemy2x2, enemy2y2)
        end
    end
    if enemy3 == 1 then
        if enemy3Type == 0 then
            koopa(enemy3x1, enemy3y1, enemy3x2, enemy3y2)
        end
        if enemy3Type == 6 then
            goomba(enemy3x1, enemy3y1, enemy3x2, enemy3y2)
        end
    end
    if enemy4 == 1 then
        if enemy4Type == 0 then
            koopa(enemy4x1, enemy4y1, enemy4x2, enemy4y2)
        end
        if enemy4Type == 6 then
            goomba(enemy4x1, enemy4y1, enemy4x2, enemy4y2)
        end
    end
    if enemy5 == 1 then
        if enemy5Type == 0 then
            koopa(enemy5x1, enemy5y1, enemy5x2, enemy5y2)
        end
        if enemy5Type == 6 then
            goomba(enemy5x1, enemy5y1, enemy5x2, enemy5y2)
        end
    end
    if fireball1 == 1 then
        fireball(fireball1x1, fireball1y1, fireball1x2, fireball1y2)
    end

    if fireball2 == 1 then
        fireball(fireball2x1, fireball2y1, fireball2x2, fireball2y2)
    end
end

function coordonees(x1, y1, x2, y2)
    local X = ((x1 + x2) / 2) / 256
    local Y = ((y1 + y2) / 2) / 224
    local X_width = (x2 - x1) / 256
    local Y_height = (y2 - y1) / 224
    return {X, Y, X_width, Y_height}
end

function Mario(Type, x1, y1, x2, y2)
    if Type == 0 then
        local coordonees = coordonees(x1 - 4, y1 - 3, x2 + 3, y2 + 2)
        local X = coordonees[1]
        local Y = coordonees[2]
        local X_width = coordonees[3]
        local Y_height = coordonees[4]
        if affichage == 1 then
            dessin(x1 - 4, y1 - 3, x2 + 3, y2 + 2) --
        else
            ecritureFichier(Type, X, Y, X_width, Y_height)  --
        end
    end
    if Type ~= 0 then
        local coordonees = coordonees(x1 - 3, y1 - 7, x2 + 2, y2 + 2)
        local X = coordonees[1]
        local Y = coordonees[2]
        local X_width = coordonees[3]
        local Y_height = coordonees[4]
        if affichage == 1 then
            dessin(x1 - 3, y1 - 7, x2 + 2, y2 + 2) --
        else
            ecritureFichier(Type, X, Y, X_width, Y_height) --
        end
    end
end

function koopa(x1, y1, x2, y2)
    local coordonees = coordonees(x1 - 3, y1 - 7, x2 + 2, y2 + 5)
    local X = coordonees[1]
    local Y = coordonees[2]
    local X_width = coordonees[3]
    local Y_height = coordonees[4]
    if affichage == 1 then
        dessin(x1 - 3, y1 - 7, x2 + 2, y2 + 5) --
    else
        ecritureFichier(2, X, Y, X_width, Y_height) --
    end
end

function goomba(x1, y1, x2, y2)
    local coordonees = coordonees(x1 - 4, y1 - 5, x2 + 3, y2 + 6)
    local X = coordonees[1]
    local Y = coordonees[2]
    local X_width = coordonees[3]
    local Y_height = coordonees[4]
    if affichage == 1 then
        dessin(x1 - 4, y1 - 5, x2 + 3, y2 + 6) --
    else
        ecritureFichier(4, X, Y, X_width, Y_height) --
    end
end

function fireball(x1, y1, x2, y2)
    local coordonees = coordonees(x1, y1 + 1, x2, y2 + 2)
    local X = coordonees[1]
    local Y = coordonees[2]
    local X_width = coordonees[3]
    local Y_height = coordonees[4]
    if affichage == 1 then
        dessin(x1, y1 + 1, x2, y2 + 2) --
    else
        ecritureFichier(3, X, Y, X_width, Y_height) --
    end

end




affichage = 1
nb_image = 200





nb_image = nb_image + 2
gui.clearGraphics()
gui.cleartext()
deleteOldFiles()
choix()