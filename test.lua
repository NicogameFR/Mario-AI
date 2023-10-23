-- racio: 1024:768

affichage = 1
nb_image = 20

function deleteOldFiles()
    local command_1 = "cd ./database/images/ & del *.png"
    os.execute(command_1)
    local command_2 = "cd ./database/labels/ & del *.txt"
    os.execute(command_2)
end

function saveScreenshot(nom)
    local commande = "nircmd.exe win hide foreground & " .. "nircmd.exe savescreenshot ./database/images/" .. nom ..
                         ".png 1513 54 814 623 " .. '& exit'
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
    if X_width == 0 or Y_height == 0 then
        return
    end
    local fichier = io.open("./database/labels/frame_" .. nb + 1 .. ".txt", "a+")
    fichier:write(Type .. " " .. X .. " " .. Y .. " " .. X_width .. " " .. Y_height .. "\n")
    fichier:close()
end

function dessin(x1, y1, x2, y2, color)
    local X_width = (x2 - x1)
    local Y_height = (y2 - y1)
    gui.drawRectangle(x1, y1, X_width, Y_height, color)
end

function detection(m1, m2, decalage_niveau, decalage_block)
    for y = decalage_niveau + 1, 15 do
        for i = 1, 13 do
            local val = memory.readbyte(m1 + y + (i - 1) * 16)
            local color = color_choix(val)
            if color ~= nil then
                if val ~= 0 then
                    if i == 13 then
                        gui.drawRectangle(0 + 16 * y - decalage_block - decalage_niveau * 16, 216, 15, 7, color)
                    else
                        gui.drawRectangle(0 + 16 * y - decalage_block - decalage_niveau * 16, 8 + 16 * i, 15, 15, color)
                    end
                end
            end
        end
    end
    for y = 0, decalage_niveau - 1 do
        for i = 1, 13 do
            local val = memory.readbyte(m2 + y + (i - 1) * 16)
            local color = color_choix(val)
            if color ~= nil then
                if val ~= 0 then
                    if i == 13 then
                        gui.drawRectangle(256 - (decalage_niveau * 16 + decalage_block) + 16 * y, 216, 15, 7, color)
                    else
                        gui.drawRectangle(256 - (decalage_niveau * 16 + decalage_block) + 16 * y, 8 + 16 * i, 15, 15,
                            color)
                    end
                end
            end
        end
    end
end

function detection_cote(m1, m2, decalage_niveau, decalage_block)
    for i = 2, 14 do
        local val = memory.readbyte(m1 + (i - 2) * 16 + decalage_niveau)
        local color = color_choix(val)
        if color ~= nil then
            if i == 14 then
                if val ~= 0 then
                    gui.drawRectangle(0, 216, 15 - decalage_block, 7, color)
                end
            else
                if val ~= 0 then
                    gui.drawRectangle(0, 8 + 16 * (i - 1), 15 - decalage_block, 15, color)
                end
            end
        end
        local val = memory.readbyte(m2 + (i - 2) * 16 + decalage_niveau)
        local color = color_choix(val)
        if color ~= nil then
            if i == 14 then
                if val ~= 0 then
                    gui.drawRectangle(256 - decalage_block, 216, decalage_block - 1, 7, color)
                end
            else
                if val ~= 0 then
                    gui.drawRectangle(256 - decalage_block, 8 + 16 * (i - 1), decalage_block - 1, 15, color)
                end
            end
        end
    end
end

function quadrillage(decalage_block, decalage_niveau)
    if decalage_niveau <= 15 then
        detection_cote(0x500, 0x5D0, decalage_niveau, decalage_block)
    else
        local decalage = decalage_niveau - 16
        detection_cote(0x5D0, 0x500, decalage, decalage_block)
    end

    if decalage_niveau <= 15 then
        detection(0x500, 0x5D0, decalage_niveau, decalage_block)
    else
        local decalage = decalage_niveau - 16
        detection(0x5D0, 0x500, decalage, decalage_block)
    end

end

function choix()
    if affichage == 1 then
        while true do
            main()
        end
    elseif affichage == 0 then
        nb_image = nb_image + 2
        deleteOldFiles()
        nb = -1
        for i = 1, nb_image do
            local decalage_block = memory.readbyte(0x071C) % 16
            local decalage_niveau = memory.readbyte(0x071C) // 16
            local ecran_actuel = memory.readbyte(0x071A)

            if ecran_actuel % 2 ~= 0 then
                decalage_niveau = decalage_niveau + 16
            end
            emu.frameadvance()
            if nb > 0 then
                saveScreenshot("frame_" .. nb)
            end
            nb = nb + 1
            quadrillage(decalage_block, decalage_niveau)
            main()
        end
    end
end

function color_choix(val)
    local color = nil
    if val == 84 then
        color = "black"
    end
    if val == 81 or val == 88 or val == 87 or val == 82 then
        color = "blue"
    end
    if val == 192 or val == 193 then
        color = "red"
    end
    if val == 16 or val == 17 or val == 18 or val == 19 or val == 20 or val == 21 then
        color = "green"
    end
    if val == 97 then
        color = "yellow"
    end
    if val == 36 or val == 37 then
        color = "purple"
    end
    if val == 196 then
        color = "pink"
    end
    if val == 28 then
        color = 'grey'
    end
    return color
end

function main()

    local mariox1 = memory.readbyte(0x04AC)
    local mariox2 = memory.readbyte(0x04AE)
    local marioy1 = memory.readbyte(0x04AD) - 9
    local marioy2 = memory.readbyte(0x04AF) - 9
    local marioType = memory.readbyte(0x0756)
    local marioState = memory.readbyte(0x000E)

    local enemy1
    local enemy2
    local enemy3
    local enemy4
    local enemy5

    enemy1 = memory.readbyte(0x000F)
    enemy2 = memory.readbyte(0x0010)
    enemy3 = memory.readbyte(0x0011)
    enemy4 = memory.readbyte(0x0012)
    enemy5 = memory.readbyte(0x0013)

    local fireball1
    local fireball2

    if marioType == 2 then
        fireball1 = memory.readbyte(0x0024)
        fireball2 = memory.readbyte(0x0025)
    end

    if enemy1 == 1 then
        enemy1x1 = memory.readbyte(0x04B0)
        enemy1x2 = memory.readbyte(0x04B2)
        enemy1y1 = memory.readbyte(0x04B1) - 9
        enemy1y2 = memory.readbyte(0x04B3) - 9
        enemy1Type = memory.readbyte(0x0016)
    else
        memory.writebyte(0x04B0, 255)
        memory.writebyte(0x04B1, 255)
        memory.writebyte(0x04B2, 255)
        memory.writebyte(0x04B3, 255)
    end

    if enemy2 == 1 then
        enemy2x1 = memory.readbyte(0x04B4)
        enemy2x2 = memory.readbyte(0x04B6)
        enemy2y1 = memory.readbyte(0x04B5) - 9
        enemy2y2 = memory.readbyte(0x04B7) - 9
        enemy2Type = memory.readbyte(0x0017)
    else
        memory.writebyte(0x04B4, 255)
        memory.writebyte(0x04B6, 255)
        memory.writebyte(0x04B5, 255)
        memory.writebyte(0x04B7, 255)

    end

    if enemy3 == 1 then
        enemy3x1 = memory.readbyte(0x04B8)
        enemy3x2 = memory.readbyte(0x04BA)
        enemy3y1 = memory.readbyte(0x04B9) - 9
        enemy3y2 = memory.readbyte(0x04BB) - 9
        enemy3Type = memory.readbyte(0x0018)
    else
        memory.writebyte(0x04B8, 255)
        memory.writebyte(0x04BA, 255)
        memory.writebyte(0x04B9, 255)
        memory.writebyte(0x04BB, 255)
    end

    if enemy4 == 1 then
        enemy4x1 = memory.readbyte(0x04BC)
        enemy4x2 = memory.readbyte(0x04BE)
        enemy4y1 = memory.readbyte(0x04BD) - 9
        enemy4y2 = memory.readbyte(0x04BF) - 9
        enemy4Type = memory.readbyte(0x0019)
    else
        memory.writebyte(0x04BC, 255)
        memory.writebyte(0x04BE, 255)
        memory.writebyte(0x04BD, 255)
        memory.writebyte(0x04BF, 255)
    end

    if enemy5 == 1 then
        enemy5x1 = memory.readbyte(0x04C0)
        enemy5x2 = memory.readbyte(0x04C2)
        enemy5y1 = memory.readbyte(0x04C1) - 9
        enemy5y2 = memory.readbyte(0x04C3) - 9
        enemy5Type = memory.readbyte(0x001A)
    else
        memory.writebyte(0x04C0, 255)
        memory.writebyte(0x04C2, 255)
        memory.writebyte(0x04C1, 255)
        memory.writebyte(0x04C3, 255)
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
        local decalage_block = memory.readbyte(0x071C) % 16
        local decalage_niveau = memory.readbyte(0x071C) // 16
        local ecran_actuel = memory.readbyte(0x071A)

        if ecran_actuel % 2 ~= 0 then
            decalage_niveau = decalage_niveau + 16
        end
        emu.frameadvance()

        quadrillage(decalage_block, decalage_niveau)
    end

    if marioState ~= 3 and marioState ~= 6 and marioState ~= 11 then
        Mario(marioType, mariox1, marioy1, mariox2, marioy2)
    end

    if enemy1 == 1 then
        enemyType(enemy1Type, enemy1x1, enemy1y1, enemy1x2, enemy1y2)
    end
    if enemy2 == 1 then
        enemyType(enemy2Type, enemy2x1, enemy2y1, enemy2x2, enemy2y2)
    end
    if enemy3 == 1 then
        enemyType(enemy3Type, enemy3x1, enemy3y1, enemy3x2, enemy3y2)
    end
    if enemy4 == 1 then
        enemyType(enemy4Type, enemy4x1, enemy4y1, enemy4x2, enemy4y2)
    end
    if enemy5 == 1 then
        enemyType(enemy5Type, enemy5x1, enemy5y1, enemy5x2, enemy5y2)
    end
    if fireball1 == 1 then
        fireball(fireball1x1, fireball1y1, fireball1x2, fireball1y2)
    end
    if fireball2 == 1 then
        fireball(fireball2x1, fireball2y1, fireball2x2, fireball2y2)
    end
end

function enemyType(type, x1, y1, x2, y2)
    if type == 0 then
        koopa(x1, y1, x2, y2)
    end
    if type == 6 then
        goomba(x1, y1, x2, y2)
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
            dessin(x1 - 4, y1 - 3, x2 + 3, y2 + 2, "black") --
        else
            ecritureFichier(Type, X, Y, X_width, Y_height) --
        end
    end
    if Type ~= 0 then
        local coordonees = coordonees(x1 - 3, y1 - 7, x2 + 2, y2 + 2)
        local X = coordonees[1]
        local Y = coordonees[2]
        local X_width = coordonees[3]
        local Y_height = coordonees[4]
        if affichage == 1 then
            if Type == 1 then
                dessin(x1 - 3, y1 - 7, x2 + 2, y2 + 2, "blue") --
            else
                dessin(x1 - 3, y1 - 7, x2 + 2, y2 + 2, "red")
            end --
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
        dessin(x1 - 3, y1 - 7, x2 + 2, y2 + 5, "red") --
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
        dessin(x1 - 4, y1 - 5, x2 + 3, y2 + 6, "pink") --
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
        dessin(x1, y1 + 1, x2, y2 + 2, "yellow") --
    else
        ecritureFichier(3, X, Y, X_width, Y_height) --
    end

end

gui.clearGraphics()
gui.cleartext()

choix()
