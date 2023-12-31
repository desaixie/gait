# please note Replica is a right-handed coordinate system!

from locale import normalize
import numpy as np
from pyparsing import original_text_for
# from sklearn.preprocessing import normalize

# corner1 = np.array([-0.679394, -1.36736, -3.11225])
# corner2 = np.array([6.57061, -1.29736, -2.96225])
# corner3 = np.array([6.57061, -1.43736, 0.954633])
# corner4 = np.array([-0.729394, -1.22736, 0.937749])
# corner5 = np.array([-0.729394, 1.01264, 0.937749])

def LoadCorners(fpath):
    print("========================= loading scene file: ",fpath)
    with open(fpath,'r') as rf:
        content = rf.readlines()
        print("input file content", content)
        cn1, cn2, cn3, cn4, cn5 = np.array([0.,0.,0.]),np.array([0.,0.,0.]),np.array([0.,0.,0.]),np.array([0.,0.,0.]),np.array([0.,0.,0.])
        i = 0
        for variable in [cn1, cn2, cn3, cn4, cn5]:
            print("line content", content[i])
            part = content[i].split('[')
            numbers = part[1].split(']')
            numbers = numbers[0].split(',')
            variable[0] = float(numbers[0])
            variable[1] = float(numbers[1])
            variable[2] = float(numbers[2])
            i = i+1
    rf.close()
    print("corners: ", cn1, cn2, cn3, cn4, cn5)
    return cn1, cn2, cn3, cn4, cn5

def Builder(corner1, corner2, corner3, corner4, corner5):
    maxFloor = max([corner1[1], corner2[1], corner3[1], corner4[1]])
    corner1[1] = maxFloor
    corner2[1] = maxFloor
    corner3[1] = maxFloor
    corner4[1] = maxFloor
    v12 = corner2 - corner1
    v43 = corner3 - corner4
    v23 = corner3 - corner2
    v41 = corner1 - corner4

    fourV = np.array([v12,v43,v23,v41])
    print(fourV)

    # with i in range(4):
    #     v = fourV[i,:]
    #     nv = v/np.linalg.norm(v)
    #     xdir = np.array([1,0,0])
    #     ydir = np.array([0,1,0])
    #     np.dot(v, xdir)


    #yAxis = corner5- corner4
    #yAxis = yAxis/np.linalg.norm(yAxis)
    yAxis = np.array([0.,1.0,0.])
    
    nv43 = v43/np.linalg.norm(v43)
    est_v41 = np.cross(yAxis, nv43)
    est_nv41 = est_v41/np.linalg.norm(est_v41)
    print("est_nv41", est_nv41)

    nv12 = v12/np.linalg.norm(v12)
    est_v14 = np.cross(nv12, yAxis)
    est_nv14 = est_v14/np.linalg.norm(est_v14)
    print("est_nv14", est_nv14)

    nv41 = v41/np.linalg.norm(v41)

    if(np.dot(est_nv41, nv41)<np.dot(est_nv14, -nv41)):
        # choose 4 as the primary
        minCorner = corner4
        zAxis = nv43
        xAxis = est_nv41
        roomY = corner5[1] - corner4[1]
        roomX = min(np.dot(corner1-minCorner, xAxis), np.dot(corner2-minCorner, xAxis))
        roomZ = min(np.dot(corner2-minCorner, zAxis), np.dot(corner3-minCorner, zAxis))
        
    else:
        # choose 1 as the primary
        minCorner = corner1
        xAxis = nv12
        zAxis = est_nv14
        roomY = corner5[1] - corner4[1]
        roomX = min(np.dot(corner2-minCorner, xAxis), np.dot(corner3-minCorner, xAxis))
        roomZ = min(np.dot(corner3-minCorner, zAxis), np.dot(corner4-minCorner, zAxis))
        
    roomSize = np.array([roomX, roomY, roomZ])
    return minCorner, roomSize, xAxis, yAxis, zAxis

def WriteToFile(fpath, varlist):
    wf = open(fpath,'w+')
    for i in range(4):
        wf.write(np.array2string(varlist[i])+"\n")
    wf.write(np.array2string(varlist[4]))
    wf.close()

if __name__ == "__main__":
    cn1, cn2, cn3, cn4, cn5 = LoadCorners("./denseSampling/scene_corners_office_0.txt")
    origin, roomSize, xAxis, yAxis, zAxis = Builder(cn1, cn2, cn3, cn4, cn5)
    print(f"origin {origin}, roomSize {roomSize}, xAxis {xAxis}, yAxis {yAxis}, zAxis {zAxis}")
    # f = open("boundingbox_room_0.txt",'w+')
    # f.write(np.array2string(origin)+"\n")
    # f.write(np.array2string(roomSize)+"\n")
    # f.write(np.array2string(xAxis)+"\n")
    # f.write(np.array2string(yAxis)+"\n")
    # f.write(np.array2string(zAxis))
    # f.close()
    WriteToFile("./denseSampling/boundingbox_office_0.txt", [origin, roomSize, xAxis, yAxis, zAxis])

