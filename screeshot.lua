-- racio: 1024:768
affichage = 0
nb_image = 10

function deleteOldFiles()
    local command_1 = "cd ./database/images/ & del *.png"
    os.execute(command_1)
    local command_2 = "cd ./database/labels/ & del *.txt"
    os.execute(command_2)
end

function saveScreenshot(nom)
    local commande = "nircmd.exe savescreenshot ./database/images/" .. nom ..
                         ".png 1515 54 809 619 " .. '& exit'
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
    if X_width == 0 or Y_height == 0 then
        local fichier = io.open("./database/labels/frame_" .. nb .. ".txt", "a+")
        fichier:close()
        return
    end
    local fichier = io.open("./database/labels/frame_" .. nb .. ".txt", "a+")
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
            local color = choix_block(val)
            if color ~= nil then
                if val ~= 0 then
                    if i == 13 then
                        if affichage == 0 then
                            block(color, 0 + 16 * y - decalage_block - decalage_niveau * 16, 216, 15, 7)
                        else
                            gui.drawRectangle(0 + 16 * y - decalage_block - decalage_niveau * 16, 216, 15, 7, color)
                        end
                    else
                        if affichage == 0 then
                            block(color, 0 + 16 * y - decalage_block - decalage_niveau * 16, 8 + 16 * i, 15, 15)
                        else
                            gui.drawRectangle(0 + 16 * y - decalage_block - decalage_niveau * 16, 8 + 16 * i, 15, 15,
                                color)
                        end
                    end
                end
            end
        end
    end
    for y = 0, decalage_niveau - 1 do
        for i = 1, 13 do
            local val = memory.readbyte(m2 + y + (i - 1) * 16)
            local color = choix_block(val)
            if color ~= nil then
                if val ~= 0 then
                    if i == 13 then
                        if affichage == 0 then
                            block(color, 256 - (decalage_niveau * 16 + decalage_block) + 16 * y, 216, 15, 7)
                        else
                            gui.drawRectangle(256 - (decalage_niveau * 16 + decalage_block) + 16 * y, 216, 15, 7,
                                color)
                        end
                    else
                        if affichage == 0 then
                            block(color, 256 - (decalage_niveau * 16 + decalage_block) + 16 * y, 8 + 16 * i, 15, 15)
                        else
                            gui.drawRectangle(256 - (decalage_niveau * 16 + decalage_block) + 16 * y, 8 + 16 * i, 15,
                                15, color)
                        end
                    end
                end
            end
        end
    end
end

function detection_cote(m1, m2, decalage_niveau, decalage_block)
    for i = 2, 14 do
        local val = memory.readbyte(m1 + (i - 2) * 16 + decalage_niveau)
        local color = choix_block(val)
        if color ~= nil then
            if i == 14 then
                if val ~= 0 then
                    if affichage == 0 then
                        block(color, 0, 216, 15 - decalage_block, 7)
                    else
                        gui.drawRectangle(0, 216, 15 - decalage_block, 7, color)
                    end
                end
            else
                if val ~= 0 then
                    if affichage == 0 then
                        block(color, 0, 8 + 16 * (i - 1), 15 - decalage_block, 15)
                    else
                        gui.drawRectangle(0, 8 + 16 * (i - 1), 15 - decalage_block, 15, color)
                    end
                end
            end
        end
        local val = memory.readbyte(m2 + (i - 2) * 16 + decalage_niveau)
        local color = choix_block(val)
        if color ~= nil then
            if i == 14 then
                if val ~= 0 then
                    if affichage == 0 then
                        block(color, 256 - decalage_block, 216, decalage_block - 1, 7)
                    else
                        gui.drawRectangle(256 - decalage_block, 216, decalage_block - 1, 7, color)
                    end
                end
            else
                if val ~= 0 then
                    if affichage == 0 then
                        block(color, 256 - decalage_block, 8 + 16 * (i - 1), decalage_block - 1, 15)
                    else
                        gui.drawRectangle(256 - decalage_block, 8 + 16 * (i - 1), decalage_block - 1, 15, color)
                    end
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
        deleteOldFiles()
        nb = 0
        for i = 1, nb_image do
            local decalage_block = memory.readbyte(0x071C) % 16
            local decalage_niveau = memory.readbyte(0x071C) // 16
            local ecran_actuel = memory.readbyte(0x071A)

            if ecran_actuel % 2 ~= 0 then
                decalage_niveau = decalage_niveau + 16
            end
            quadrillage(decalage_block, decalage_niveau)
            main()
            emu.frameadvance()
            emu.frameadvance()
            saveScreenshot("frame_" .. nb)
            nb = nb + 1
            frame_pass(100)
        end
    end
end

function frame_pass(x)
    for i = 0, x do
        emu.frameadvance()
    end
end

function choix_block(val)
    local color = nil
    local num = nil
    if val == 84 then
        color = "black"
        num = 10
    end
    if val == 81 or val == 88 or val == 87 or val == 82 or val == 90 or val == 93 then
        color = "blue"
        num = 9
    end
    if val == 192 or val == 193 then
        color = "red"
        num = 11
    end
    if val == 20 or val == 21 or val == 32 or val == 30 or val == 29 or val == 32 or val == 33 then
        color = "green"
        num = 12
    end
    if val == 16 or val == 17 or val == 18 or val == 19 or val == 28 or val == 31 then
        color = "purple"
        num = 13
    end
    if val == 97 then
        color = "yellow"
        num = 20
    end
    if val == 36 or val == 37 then
        color = "purple"
        num = 15
    end
    if val == 196 then
        color = "pink"
        num = 8
    end
    if val == 194 then
        color = 'yellow'
        num = 14
    end
    if affichage == 0 then
        return num
    else
        return color
    end
end

function main()

    local mariox1 = memory.readbyte(0x04AC)
    local mariox2 = memory.readbyte(0x04AE)
    local marioy1 = memory.readbyte(0x04AD) - 9
    local marioy2 = memory.readbyte(0x04AF) - 9
    local marioType = memory.readbyte(0x0756)
    local marioState = memory.readbyte(0x000E)

    --if marioState == 0 then
    --    if affichage == 0 then
    --        local fichier = io.open("./database/labels/frame_" .. nb .. ".txt", "a+")
    --        fichier:close()
    --        return
    --    else
    --        gui.clearGraphics()
    --        emu.frameadvance()
    --        return
    --    end
    --end
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

    local powerup1

    powerup1 = memory.readbyte(0x0014)

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
        enemy1State = memory.readbyte(0x001E)
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
        enemy2State = memory.readbyte(0x001F)
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
        enemy3State = memory.readbyte(0x0020)
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
        enemy4State = memory.readbyte(0x0021)
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
        enemy5State = memory.readbyte(0x0022)
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

    if powerup1 == 1 then
        powerupx1 = memory.readbyte(0x04C4)
        powerupx2 = memory.readbyte(0x04C6)
        powerupy1 = memory.readbyte(0x04C5) - 9
        if memory.readbyte(0x0023) ~= 128 and memory.readbyte(0x0023) ~= 192 and memory.readbyte(0x0023) ~= 194  and memory.readbyte(0x0023) ~= 130 then
            powerupy2 = powerupy1 + (memory.readbyte(0x0023) - 5)
        else
            powerupy2 = memory.readbyte(0x04C7) - 9
        end
        powerup1Type = memory.readbyte(0x0039)
    else
        memory.writebyte(0x04C4, 255)
        memory.writebyte(0x04C6, 255)
        memory.writebyte(0x04C5, 255)
        memory.writebyte(0x04C7, 255)
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

    if marioState ~= 3 and marioState ~= 6 and marioState ~= 11 and marioState ~= 5 and marioState ~= 7 and marioState ~=
        2 and marioState ~= 9 and marioState ~= 10 and marioState ~= 13 then
        Mario(marioType, mariox1, marioy1, mariox2, marioy2)
    end

    if enemy1 == 1 and (enemy1State == 0 or ((enemy1State == 132 or enemy1State == 4) and (enemy1Type == 0))) then
        enemyType(enemy1Type, enemy1x1, enemy1y1, enemy1x2, enemy1y2, enemy1State)
    end
    if enemy2 == 1 and (enemy2State == 0 or ((enemy2State == 132 or enemy2State == 4) and (enemy2Type == 0))) then
        enemyType(enemy2Type, enemy2x1, enemy2y1, enemy2x2, enemy2y2, enemy2State)
    end
    if enemy3 == 1 and (enemy3State == 0 or ((enemy3State == 132 or enemy3State == 4) and (enemy3Type == 0))) then
        enemyType(enemy3Type, enemy3x1, enemy3y1, enemy3x2, enemy3y2, enemy3State)
    end
    if enemy4 == 1 and (enemy4State == 0 or ((enemy4State == 132 or enemy4State == 4) and (enemy4Type == 0))) then
        enemyType(enemy4Type, enemy4x1, enemy4y1, enemy4x2, enemy4y2, enemy4State)
    end
    if enemy5 == 1 and (enemy5State == 0 or ((enemy5State == 132 or enemy5State == 4) and (enemy5Type == 0))) then
        enemyType(enemy5Type, enemy5x1, enemy5y1, enemy5x2, enemy5y2, enemy5State)
    end
    if fireball1 == 1 then
        fireball(fireball1x1, fireball1y1, fireball1x2, fireball1y2)
    end
    if fireball2 == 1 then
        fireball(fireball2x1, fireball2y1, fireball2x2, fireball2y2)
    end
    if powerup1 == 1 then
        powerup(powerupx1, powerupy1, powerupx2, powerupy2)
    end
end

function enemyType(type, x1, y1, x2, y2, State)
    if type == 0 or type == 3 then
        koopa(x1, y1, x2, y2, State)
    elseif type == 6 then
        goomba(x1, y1, x2, y2)
    elseif type == 13 then
        piranha(x1, y1, x2, y2)
    elseif type == 38 or type == 39 then
        plateforme(x1, y1, x2, y2)
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

function koopa(x1, y1, x2, y2, State)
    local infos
    local type
    if State == 132 or State == 4 then
        infos = coordonees(x1 - 3, y1 +1 , x2 + 2, y2 + 5)
        type = "21"
    else
        infos = coordonees(x1 - 3, y1 - 7, x2 + 2, y2 + 5)
        type = "4"
    end
    local X = infos[1]
    local Y = infos[2]
    local X_width = infos[3]
    local Y_height = infos[4]
    if affichage == 1 then
        if State == 132 or State == 4 then
            dessin(x1 - 3, y1+1, x2 + 2, y2 + 5, "red") --
        else
            dessin(x1 - 3, y1 - 7, x2 + 2, y2 + 5, "red") --
        end
    else
        ecritureFichier(type, X, Y, X_width, Y_height) --
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
        ecritureFichier(3, X, Y, X_width, Y_height) --
    end
end

function fireball(x1, y1, x2, y2)
    local coordonees = coordonees(x1, y1 + 1, x2, y2 + 2)
    local X = coordonees[1]
    local Y = coordonees[2]
    local X_width = coordonees[3]
    local Y_height = coordonees[4]
    if affichage == 1 then
        dessin(x1, y1 + 1, x2, y2 + 2, "yellow")
    else
        ecritureFichier(5, X, Y, X_width, Y_height)
    end
end

function piranha(x1, y1, x2, y2)
    local coordonees = coordonees(x1 - 5, y1 - 13, x2 + 4, y2 + 5)
    local X = coordonees[1]
    local Y = coordonees[2]
    local X_width = coordonees[3]
    local Y_height = coordonees[4]
    if affichage == 1 then
        dessin(x1 - 5, y1 - 13, x2 + 4, y2 + 5, "pink") --
    else
        ecritureFichier(6, X, Y, X_width, Y_height) --
    end
end

function plateforme(x1, y1, x2, y2)
    local coordonees = coordonees(x1 - 2, y1 - 2, x2 + 1, y2 - 1)
    local X = coordonees[1]
    local Y = coordonees[2]
    local X_width = coordonees[3]
    local Y_height = coordonees[4]
    if affichage == 1 then
        dessin(x1 - 2, y1 - 2, x2 + 1, y2 - 1, "red") --
    else
        ecritureFichier(7, X, Y, X_width, Y_height) --
    end
end

function powerup(x1, y1, x2, y2)
    local coordonees = coordonees(x1 - 3, y1, x2 + 2, y2 + 5)
    local X = coordonees[1]
    local Y = coordonees[2]
    local X_width = coordonees[3]
    local Y_height = coordonees[4]
    local type
    if powerup1Type == 0 then
        type = 16
    elseif powerup1Type == 1 then
        type = 17
    elseif powerup1Type == 2 then
        type = 18
    else
        type = 19
    end

    if affichage == 1 then
        dessin(x1 - 3, y1, x2 + 2, y2 + 5, "pink") --
    else
        ecritureFichier(type, X, Y, X_width, Y_height) --
    end
end

function block(type, x, y, X_width, Y_height)
    local X = ((x + X_width / 2)+0.5) / 256
    local Y = ((y + Y_height / 2)+0.5) / 224
    local X_width = (X_width+1) / 256
    local Y_height = (Y_height+1) / 224
    ecritureFichier(type, X, Y, X_width, Y_height)
end

gui.clearGraphics()
gui.cleartext()

choix()
