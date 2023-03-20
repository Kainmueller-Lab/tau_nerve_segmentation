# tau_nerve_segmentation
This a started project for segmenting cells and proteins which was created during my work at the Kainmüller Lab at the MDC Berlin.

## Structure
The repository is composed by a data folder (which should contain the data and labelled masks), a segmentation pipeline which was forked and adapted from https://github.com/Kainmueller-Lab/lizard_challenge and an analysis part for getting an idea of the data and some analysis on the masked data. The instance_cell.ipynb is the first try of training a network for segmenting the cells in the data.

## Data
The data originates from 5 .tiff files which contains a time series of light microscope data which each was seperated to 30 pages. The color channel of each of those pages shows {blue: cells + cell_plasma, red: tau proteins, green: ApoE proteins}
