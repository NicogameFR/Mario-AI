

variable = 0x05D0-17


print(string.format("%X", variable))

print(memory.readbyte(variable))
--local number = "05C0"

--number = tonumber(number,16)

--number = number + 1

--print(number)

--number = "0".. number

--print(number)

--number = string.format("%X",number)




-- Coordonnées x et y
local x = 100
local y = 200

-- Texte à afficher
local message = "Hello, World!"

-- Options facultatives avec les mêmes valeurs par défaut que dans votre description
local forecolor = nil -- Par défaut, la couleur du texte est blanche
local backcolor = nil -- Par défaut, l'arrière-plan est transparent
local fontsize = 6 -- Par défaut, la taille de la police est de 12
local fontfamily = nil -- Par défaut, la famille de police est monospace générique
local fontstyle = "regular" -- Par défaut, le style de police est "regular"
local horizalign = "left" -- Par défaut, l'alignement horizontal est "left"
local vertalign = "bottom" -- Par défaut, l'alignement vertical est "bottom"
local surfacename = nil -- Le nom de la surface si nécessaire

-- Appel de la fonction gui.drawString avec les mêmes valeurs que dans votre description
gui.drawString(x, y, message, forecolor, backcolor, fontsize, fontfamily, fontstyle, horizalign, vertalign, surfacename)
