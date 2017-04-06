import numpy

def SeparateCSV(D, w_idx, sch_idx):  # D is dataset, feats is the number of features

    feats = w_idx + sch_idx
    row, col = D.shape
    X = D[:, 0:(feats)]
    Y = D[:, col-2]
    idx = D[:, col-1]
    W = D[:, 0:w_idx]
    Sch = D[:, w_idx: (w_idx + sch_idx)]

    # setting time
    t24 = numpy.arange(1,25)
    t = numpy.tile(t24, (row/24))
    return X, Y, idx, W, Sch, t

#setting up data for data conversion




