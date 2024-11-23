# Degree Matrix Comparison

## Datasets
1. COSNET: https://www.aminer.cn/cosnet. Specifically, look for datasets for Flickr, Lastfm, and Myspace.
2. SNAP: Download "com-Youtube" and "ego-Facebook" from the SNAP database. For the baseline experiments, the PPI dataset we are using is taken from here: https://snap.stanford.edu/biodata/datasets/10013/10013-PPT-Ohmnet.html.
3. Network Repository: search in the repository with network name as listed in the paper, then download the files. One would need to perform the graph sampling from the start.

## Code Content
We provide code scripts that can execute with the graphml files posted, but the synthetic data creation process can differ significantly depending on the input graphs. Once the graph extraction and sampling processes are done, the Degree Matrix Comparison process is the same and can be directly used.

## Considerations
1. Graphml files are sampled graphs ($G_{s}$) that can be used for further sampling and graph alignment.
2. To run the code files, code scripts would need to be modified depending on which network we are running the experiment on. Ideally, everything should run as long as you put in the right paths for downloaded files and datasets.
3. The DMC_Facebook should be easier to modify when applied to Facebook or Youtube networks for alignment. The other DMC script is easier to modify for Flickr, Lastfm, and Myspace.






