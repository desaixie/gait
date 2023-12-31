from locale import normalize
import numpy as np
import os
from pathlib import Path
from pyparsing import original_text_for

def loadBoundingBox(fpath="/home/peggy/Research/Aesthetic/aestheticview/denseSampling", sceneName="room_0"):
    
    fpath = os.path.join(fpath, "boundingbox_" + sceneName+".txt")
    # print("------------------------------load boudingbox func", fpath)
    print(f"Loading bounding box from path {fpath}")
    origin = np.array([0.0,0.0,0.0])
    roomSize = np.array([0.,0.,0.])
    xAxis = np.array([0.,0.,0.])
    yAxis = np.array([0.,0.,0.])
    zAxis = np.array([0.,0.,0.])
    maxmin_scores = np.array([0.,0.,0.])
    print(Path.cwd())
    with open(fpath, 'r') as rf:
        content = rf.readlines()
        i = 0
        for vec in [origin, roomSize, xAxis, yAxis, zAxis, maxmin_scores]:
            numbers = content[i].split("[")
            # print("numbers", numbers[0], numbers[1])
            numbers = numbers[1].split("]")
            numbers = numbers[0].split()
            # print(numbers)

            vec[0] = float(numbers[0])
            vec[1] = float(numbers[1])
            vec[2] = float(numbers[2])
            i = i+1
    maxmin_scores = maxmin_scores[:2]
    # rf.close()
    print(f"room origin {origin}, room size {roomSize}, xaxis {xAxis}, yaxis {yAxis}, zAxis {zAxis}"
          f", maxmin_scores {maxmin_scores}")
    return origin, roomSize, xAxis, yAxis, zAxis, maxmin_scores

if __name__ == "__main__":
    fpath = "./denseSampling/"
    origin, roomSize, xAxis, yAxis, zAxis = loadBoundingBox(fpath, None)
    print(roomSize)
    print(origin, roomSize, xAxis, yAxis, zAxis)
