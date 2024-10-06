import numpy as np
import json

# Use:

# data = createCustomDataset(output_path='customDataset2.json')
# data = readCustomDataset()['points']


# Create custom dataset
def createCustomDataset(num_turbines = 12, min_distance = 0.25, output_path = None):
    data = []
    while len(data) < num_turbines:
        append = True
        new_point = np.random.rand(2)
        for point in data:
            if (np.linalg.norm(np.array(point) - new_point) < min_distance):
                append = False
                continue
        if append:
            data.append(list(new_point))

    if output_path:
        dictionary = {"data": {"points": data}}
        json_object = json.dumps(dictionary)
        with open(output_path, "w+") as outfile:
            outfile.write(json_object)

    return data


# Reads dataset, returned as a 2D python list (of points between [0.0, 1.0])
def readCustomDataset(fileName = 'Datasets/customDataset.json'):
    with open(fileName, 'r') as openfile:
        json_object = json.load(openfile)
    return json_object['data']
