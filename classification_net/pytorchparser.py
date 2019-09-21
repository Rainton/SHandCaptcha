# coding=utf-8

import torch
import numpy as np
from collections import OrderedDict


class FcnPreProcessMark(torch.autograd.Function):
    def __init__(self):
        super(FcnPreProcessMark, self).__init__()

    def forward(self, X):
        h = X
        return h


layerDict = {'ConvNd': 'Convolution',
             'Threshold': 'ReLU',
             'MaxPool2d': 'Pooling',
             'MaxPool2D': 'Pooling',
             'AvgPool2d': 'Pooling',
             'AvgPool2D': 'Pooling',
             'Dropout': 'Dropout',
             'Addmm': 'InnerProduct',
             'BatchNorm': 'TorchBnFixedParam',
             'Add': 'Eltwise',
             'View': 'Reshape',
             'Concat': 'Concat',
             'UpsamplingNearest2d': 'NearestUnsampling',
             'UpsamplingBilinear2d': 'Bilinearupsampling',
             'Sigmoid': 'Sigmoid',
             'LeakyRelu': 'ReLU',
             'Negate': 'Power',
             'Mul': 'Eltwise',
             'SpatialCrossMapLRNFunc': 'LRN',
             'ImagewarpFunc': 'Stn',
             'ImagefuseFunc': 'Imagefuse',
             'Sub': 'Eltwise',
             'PReLU': 'PReLU',
             'Tanh': 'TanH',
             'Cat': 'Concat',
             'FcnPreProcessMark': 'FcnPreProcessMark',
             'Type': None
             }

layerId = 0
paramIdx = 0;

SUPERPARAMSDESCKEY = "superParams";
import string


def pytorch2NcnnInfo(output_var):
    global layerId;
    global paramIdx;
    netInfo = OrderedDict();

    layers = [];

    seen = set()
    topNames = dict()

    def addLayer(func):
        global layerId
        global paramIdx
        parentType = str(type(func).__name__).rstrip(string.digits)
        parentBottoms = []
        if parentType.endswith('Backward'):
            parentType = parentType[:-8]
        if hasattr(func, 'next_functions'):
            for u in func.next_functions:
                if u[0] is not None:
                    childType = str(type(u[0]).__name__).rstrip(string.digits)
                    if childType.endswith('Backward'):
                        childType = childType[:-8]
                    childName = childType + str(layerId);
                    print(parentType, childType, childName)

                    if childType != 'AccumulateGrad' and (
                            parentType != 'Addmm' or childType != 'Transpose') and childType != 'Expand' and childType != 'CudaTransfer' and childType != 'T' and childType != 'ExpandBackward':
                        if u[0] not in seen:
                            topName = addLayer(u[0])
                            parentBottoms.append(topName)
                            seen.add(u[0])
                        else:
                            topName = topNames[u[0]]
                            parentBottoms.append(topName)
                        if childType != "View" and childType != "Expand":
                            layerId = layerId + 1

        parentName = parentType + str(layerId);
        layer = OrderedDict()

        layer['name'] = parentName
        layer['type'] = layerDict[parentType];

        parentTop = parentName;
        if len(parentBottoms) > 0:
            layer['bottom'] = parentBottoms
        else:
            layer['bottom'] = ['data']
        layer['top'] = parentTop

        # params
        if parentType == "Mul":
            eltwiseParam = {
                "operation": "PROD"
            }
            layer['eltwiseParam'] = eltwiseParam
            layer[SUPERPARAMSDESCKEY] = " %d %d" % (0, 0)
        elif parentType == "FcnPreProcessMark":
            layer["type"] = "Power"
            layer[SUPERPARAMSDESCKEY] = " %.6f %.6f %.6f" % (1, 0.003921, -0.5)

            tempTop = parentTop
            parentTop = parentTop + "_prePower"
            prePoolLayer = OrderedDict()
            prePoolLayer['name'] = parentName + "_prePool"
            prePoolLayer['type'] = 'Pooling'
            prePoolLayer['bottom'] = [parentTop]
            prePoolLayer['top'] = tempTop
            prePoolLayer[SUPERPARAMSDESCKEY] = " %d %d %d %d %d" % (1, 2, 2, 0, 0)

        elif parentType == "Negate" or parentType == 'Neg':
            powerParam = {
                "power": 1,
                "scale": -1,
                "shift": 0
            }
            layer["powerParam"] = powerParam
            layer[SUPERPARAMSDESCKEY] = "%.2f %.2f %.2f" % (
                powerParam["power"], powerParam["scale"], powerParam["shift"])
        elif parentType == 'Add' or parentType == 'Add':
            eltwiseParam = OrderedDict()
            eltwiseParam['operation'] = 'SUM'
            layer['eltwiseParam'] = eltwiseParam
            layer[SUPERPARAMSDESCKEY] = "%d %d" % (1, 0)
        elif parentType == "Sub":
            eltwiseParam = OrderedDict()
            eltwiseParam['operation'] = 'SUM'
            layer['eltwiseParam'] = eltwiseParam
            layer[SUPERPARAMSDESCKEY] = "%d %d" % (1, -1)
        elif parentType == "LeakyRelu":
            slope = 0.01  # func.additional_args[0]              #to do
            layer["reluParam"] = {"slope": slope}
            layer[SUPERPARAMSDESCKEY] = "%f" % (slope)
        elif parentType == "PReLU":
            slope = 0.01  # func.additional_args[0]#0.2         #to do
            layer["preluParam"] = {"slope": slope}
            layer[SUPERPARAMSDESCKEY] = "%f" % (slope)
        elif parentType == "UpsamplingNearest2d" or parentType == "UpsamplingBilinear2d":
            convParam = OrderedDict();
            convParam['stride'] = func.scale_factor[0];
            layer['convParam'] = convParam;
            layer[SUPERPARAMSDESCKEY] = "%d" % (func.scale_factor[0])
        elif parentType == 'ConvNd':
            if func.transposed is True and func.next_functions[1][0] is None:
                layer['type'] = layerDict['UpsamplingBilinear2d']
                convParam = OrderedDict()
                factor = func.stride[0]
                convParam['num_output'] = func.next_functions[0][0].saved_tensors[0].size(1)
                convParam['group'] = convParam['num_output']
                convParam['kernel_size'] = (2 * factor - factor % 2)
                convParam['stride'] = factor
                convParam['pad'] = int(np.ceil((factor - 1) / 2.))
                convParam['weight_filler'] = {'type': 'bilinear'}
                convParam['bias_term'] = 'false'
                layer[SUPERPARAMSDESCKEY] = ""
            else:
                convParam = OrderedDict()
                convParam['pad'] = func.padding[0]
                convParam['group'] = func.groups
                convParam['stride'] = func.stride[0]
                dilation = func.dilation[0]
                convParam['dilation'] = dilation
                if dilation > 1:
                    layer['type'] = "ConvolutionDilated"
                biasTerm = 0
                if func.next_functions[1][0] is not None:
                    weights = func.next_functions[1][0].variable.data
                    layer['weights'] = weights;
                    convParam['num_output'] = weights.size(0)
                    convParam['num_input'] = weights.size(1)
                    convParam['kernel_size'] = weights.size(2)
                    if func.next_functions[2][0]:
                        biases = func.next_functions[2][0].variable.data
                        biasTerm = 1
                    else:
                        biases = None
                        biasTerm = 0
                    layer['biases'] = biases
                    layer['biasTerm'] = biasTerm
                group = func.groups
                weightDataSize = weights.shape[0] * weights.shape[1] * weights.shape[2] * weights.shape[3]
                layer['convolutionParam'] = convParam
                if dilation == 1:
                    layer[SUPERPARAMSDESCKEY] = "%d %d %d %d %d %d %d" % (
                        convParam['num_output'], convParam['kernel_size'], convParam['stride'], convParam['pad'],
                        biasTerm,
                        convParam['group'], weightDataSize)
                else:
                    layer[SUPERPARAMSDESCKEY] = "%d %d %d %d %d %d %d %d" % (
                        convParam['num_output'], convParam['kernel_size'], convParam['stride'], convParam['pad'],
                        biasTerm,
                        convParam['group'], dilation, weightDataSize)
        elif parentType == 'Threshold':
            layer[SUPERPARAMSDESCKEY] = "0.0"
        elif parentType == 'AvgPool2d':
            poolingParam = OrderedDict()
            poolingParam['pool'] = 'AVE'
            poolingParam['kernelSize'] = func.kernel_size[0]
            poolingParam['stride'] = func.stride[0]
            poolingParam['pad'] = func.padding[0]
            layer['poolingParam'] = poolingParam
            layer[SUPERPARAMSDESCKEY] = "%d %d %d %d %d" % (
                1, poolingParam['kernelSize'], poolingParam['stride'], poolingParam['pad'], 0)
        elif parentType == 'AvgPool2D':
            poolingParam = OrderedDict()
            poolingParam['pool'] = 'AVE'
            poolingParam['kernelSize'] = 2
            poolingParam['stride'] = 2
            poolingParam['pad'] = 0
            layer['poolingParam'] = poolingParam
            layer[SUPERPARAMSDESCKEY] = "%d %d %d %d %d" % (
                1, poolingParam['kernelSize'], poolingParam['stride'], poolingParam['pad'], 0)
        elif parentType == 'MaxPool2D':
            poolingParam = OrderedDict()
            poolingParam['pool'] = 'MAX'
            poolingParam['kernelSize'] = 2
            poolingParam['stride'] = 2
            poolingParam['pad'] = 0
            layer['poolingParam'] = poolingParam
            layer[SUPERPARAMSDESCKEY] = "%d %d %d %d %d" % (
                0, poolingParam['kernelSize'], poolingParam['stride'], poolingParam['pad'], 0)
        elif parentType == 'MaxPool2d':
            poolingParam = OrderedDict()
            poolingParam['pool'] = 'MAX'
            poolingParam['kernelSize'] = func.kernel_size[0]
            poolingParam['stride'] = func.stride[0]
            padding = func.padding[0]
            poolingParam['pad'] = padding
            layer['poolingParam'] = poolingParam
            layer[SUPERPARAMSDESCKEY] = "%d %d %d %d %d" % (
                0, poolingParam['kernelSize'], poolingParam['stride'], poolingParam['pad'], 0)
        elif parentType == 'Dropout':
            parentTop = parentBottoms[0]
            layer[SUPERPARAMSDESCKEY] = ""
        elif parentType == 'Addmm':
            innerProductParam = OrderedDict()
            innerProductParam['num_output'] = func.next_functions[0][0].variable.size(0)
            biases = func.next_functions[0][0].variable.data
            weights = func.next_functions[2][0].next_functions[0][0].variable.data
            layer['innerProductParam'] = innerProductParam
            layer['weights'] = weights
            layer['biases'] = biases
            layer[SUPERPARAMSDESCKEY] = "%d %d %d" % (innerProductParam['num_output'], 1, weights.size())
        elif parentType == 'View':
            parentTop = parentBottoms[0]
        elif parentType == 'SpatialCrossMapLRNFunc':
            layer['lrnParam'] = {
                'local_size': func.size,
                'alpha': func.alpha,
                'beta': func.beta,
            }
            layer[SUPERPARAMSDESCKEY] = "%d %d %f %f" % (0, func.size, func.alpha, func.beta)
        elif parentType == 'BatchNorm':
            runningMean = func.running_mean
            runningVar = func.running_var
            layer['runningMean'] = runningMean
            layer['runningVar'] = runningVar
            layer[SUPERPARAMSDESCKEY] = "%d %d" % (func.running_mean.size(0), 0)
            affine = func.next_functions[1][0] is not None

            if affine:
                tempTop = parentTop
                parentTop = parentTop + "_bn"
                scaleLayer = OrderedDict()
                scaleLayer['name'] = parentName + "_scale"
                scaleLayer['type'] = 'Scale'
                scaleLayer['bottom'] = [parentTop]
                scaleLayer['top'] = tempTop
                scaleLayer[SUPERPARAMSDESCKEY] = "%d %d" % (func.running_mean.size(0), 1)
                scaleLayer["weights"] = func.next_functions[1][0].variable.data;
                scaleLayer["biases"] = func.next_functions[2][0].variable.data;
            else:
                scale_layer = None

        layer['top'] = parentTop  # reset layer['top'] as parent_top may change
        if parentType != 'View':
            if parentType == "BatchNorm":
                layers.append(layer)
                if scaleLayer is not None:
                    layers.append(scaleLayer)
            elif parentType == "FcnPreProcessMark":
                layers.append(layer)
                layers.append(prePoolLayer)
            else:
                layers.append(layer)

        if ((parentType != "BatchNorm") or ((parentType == "BatchNorm") and (not affine))) and (
                parentType != "FcnPreProcessMark"):
            topNames[func] = parentTop
            return parentTop
        else:
            topNames[func] = tempTop;
            return tempTop

    addLayer(output_var.grad_fn)

    firstLayer = OrderedDict()
    firstLayer['type'] = "Input"
    firstLayer['name'] = "Input"
    firstLayer['bottom'] = []
    firstLayer['top'] = 'data'
    firstLayer[SUPERPARAMSDESCKEY] = "0 0 0"

    layers = [firstLayer] + layers
    netInfo['layers'] = layers
    return netInfo


def pytorch2ncnn(output_var, paramFilePath, binFilePath):
    net = pytorch2NcnnInfo(output_var)
    saveNetToFile(net, paramFilePath, binFilePath);


def write2File(txt, file, changeLine=True):
    suffix = ""
    if changeLine:
        suffix = "\n"

    if (file != None):
        file.write(txt + suffix)
    else:
        print(txt + suffix)
    pass


def saveNetToFile(net, paramFilePath, binFilePath):
    paramFile = None
    binFile = None
    if paramFilePath != "":
        paramFile = open(paramFilePath, 'w')
    if binFilePath != "":
        binFile = open(binFilePath, 'w')
    layers = net['layers'];

    botRefCount = {}
    for layer in layers:
        top = layer['top']
        botRefCount[top] = 0
        for l in layers:
            bottoms = l['bottom']
            for t in bottoms:
                if t == top:
                    botRefCount[top] = botRefCount[top] + 1

    for key in botRefCount.keys():
        refCnt = botRefCount[key];
        if refCnt > 1:
            idx = 1
            for layer in layers:
                bottoms = layer['bottom']
                for i in range(0, len(bottoms)):
                    if bottoms[i] == key:
                        bottoms[i] = key + "_split_%d" % (idx)
                        idx = idx + 1

    layerDescripitions = []
    totalBlobCnt = 0
    totalLayerCnt = 0
    for layer in layers:
        layerType = layer['type']
        layerName = layer['name']
        bottoms = layer['bottom']
        top = layer['top']
        totalBlobCnt = totalBlobCnt + 1
        print(layerName)
        if layer.get(SUPERPARAMSDESCKEY) != None:  # to do
            superParams = layer[SUPERPARAMSDESCKEY]
            print('superParams:', superParams)
        else:
            superParams = ""
            print('none superParams')
        bottomsDesc = ' '
        bottomsDesc = bottomsDesc.join(bottoms)
        layerDescStr = '%s %s %d %d %s %s %s' % (layerType, layerName, len(bottoms), 1, bottomsDesc, top, superParams)
        layerDescripitions.append(layerDescStr)
        totalLayerCnt += 1

        if layerType == "Convolution" or layerType == "ConvolutionDilated":
            weights = layer['weights'].cpu()
            weightSize = weights.shape[0] * weights.shape[1] * weights.shape[2] * weights.shape[3]

            write2File("%d" % (weightSize), binFile, True)

            for oc in weights.numpy():
                for ic in oc:
                    for row in ic:
                        for v in row:
                            write2File("%.4f " % (v), binFile, False)
            write2File("\n", binFile, False)

            if layer['biasTerm'] == 1:
                biases = layer['biases'].cpu()
                biasesSize = biases.shape[0]
                write2File("%d" % (biasesSize), binFile, True)
                for v in biases.numpy():
                    write2File("%.4f " % (v), binFile, False)
                write2File("\n", binFile, False)

        elif layerType == "TorchBnFixedParam":
            means = layer['runningMean']
            write2File("%d" % (means.shape[0]), binFile, True)
            for v in means:
                write2File("%.4f " % (v), binFile, False)
            write2File("\n", binFile, False)

            vars = layer['runningVar']
            write2File("%d" % (vars.shape[0]), binFile, True)
            for v in vars:
                write2File("%.4f " % (v), binFile, False)
            write2File("\n", binFile, False)
        elif layerType == "Scale":
            weights = layer['weights'].cpu()
            weightSize = weights.shape[0]

            write2File("%d" % (weightSize), binFile, True)

            for v in weights.numpy():
                write2File("%.4f " % (v), binFile, False)
            write2File("\n", binFile, False)

            biases = layer['biases'].cpu()
            biasesSize = biases.shape[0]
            write2File("%d" % (biasesSize), binFile, True)
            for v in biases.numpy():
                write2File("%.4f " % (v), binFile, False)
            write2File("\n", binFile, False)

        if botRefCount[top] > 1:
            totalLayerCnt += 1
            idx = 1
            splitTopBlobs = "%s_split_%d" % (top, idx)
            idx = idx + 1
            totalBlobCnt = totalBlobCnt + 1
            for i in range(1, botRefCount[top]):
                splitTopBlobs = splitTopBlobs + " %s_split_%d" % (top, idx)
                idx = idx + 1
                totalBlobCnt = totalBlobCnt + 1
            splitLayerDesc = "Split split_%s 1 %d %s %s" % (top, botRefCount[top], top, splitTopBlobs)
            layerDescripitions.append(splitLayerDesc)

    write2File("%d %d" % (totalLayerCnt, totalBlobCnt), paramFile)
    for layerDesc in layerDescripitions:
        write2File(layerDesc, paramFile)

    if paramFile is not None:
        paramFile.close()

    if binFile is not None:
        binFile.close()
