# Degree Matrix Comparison

## Datasets
1. COSNET: https://www.aminer.cn/cosnet. Specifically, look for datasets Flickr, Lastfm, and Myspace.
2. SNAP: Download "com-Youtube" and "ego-Facebook" from the SNAP database. Also go to roadmaps section to look for the California and Pennsylvania roadmaps. For the baseline experiments, the PPI dataset we are using is taken from here: https://snap.stanford.edu/biodata/datasets/10013/10013-PPT-Ohmnet.html.
3. Network Repository: search in the repository with network name as listed in the paper, then download the files. One would need to perform the graph sampling from the start.

## File and Code Content
We provide code scripts that can execute with the graphml files posted (five on the main page and the combined_ppi.graphml file is in the releases), but the synthetic data creation process can differ significantly depending on the input graphs the user desires to use. Once the graph extraction and sampling processes are done, the Degree Matrix Comparison process is the same and can be directly used. If one has difficulty running other scripts, it is most important to get the PPI DMC to work, so that one could borrow most of the code for the method itself for future experiments.

In the Alignment Methods file, there are many methods. All of them are degree based methods, but only the ones labeled with "DMC" or involve "degree matrices" in the title are the ones eventually proposed in the paper. The other degree based methods have exhibited insatisfactory performance, but they inspired the creation of DMC or Greedy DMC.

## Considerations
1. Graphml files are sampled graphs ($G_{s}$) that can be used for further sampling and graph alignment.
2. To run the code files, code scripts would need to be modified depending on which network we are running the experiment on. 
3. The DMC_Facebook should be easier to modify when applied to Facebook or Youtube networks for alignment. The flickr_degree_matrices.py DMC script is easier to modify for Flickr, Lastfm, and Myspace. Moreover, we post the DMC and Greedy DMC scripts for the PPI network used for comparing DMC and Greedy DMC to baselines.







