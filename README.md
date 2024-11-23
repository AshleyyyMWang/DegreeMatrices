# Degree Matrix Comparison

## Datasets
1. **COSNET**: Access datasets Flickr, Lastfm, and Myspace at [COSNET](https://www.aminer.cn/cosnet).
2. **SNAP**: 
   - Download **"com-Youtube"** and **"ego-Facebook"** from the [SNAP database](https://snap.stanford.edu/data/).
   - For roadmaps, refer to the **California** and **Pennsylvania** roadmaps section.
   - For baseline experiments, the **PPI dataset** used is available at [PPI-Ohmnet](https://snap.stanford.edu/biodata/datasets/10013/10013-PPT-Ohmnet.html).
3. **Network Repository**: Search the repository by the network name listed in the paper and download the files. Note that graph sampling needs to be performed from scratch.

## File and Code Content
The provided scripts are designed to work with the GraphML files shared in this repository (five files on the main page and `combined_ppi.graphml` available in the releases). However, the synthetic data creation process can vary significantly depending on the input graphs the user wishes to analyze. After the graph extraction and sampling steps are completed, the Degree Matrix Comparison (DMC) process remains consistent and can be applied directly.

If issues arise while running other scripts, prioritize getting the **PPI DMC** script to work first. This script can serve as a foundation, with much of its code being adaptable for future experiments involving the method.

The **Alignment Methods** file includes a variety of degree-based methods. However, only those labeled with **"DMC"** or mentioning **"degree matrices"** in their title are the ones proposed in the paper. While the other degree-based methods demonstrated suboptimal performance, they played an instrumental role in inspiring the creation of **DMC** and **Greedy DMC**.

## Considerations
1. **GraphML Files**: 
   - These represent graphs ($G_{s}$) that can be used for further sampling and graph alignment. 
   - The exception is **PPI**, which is treated as $G_{r}$ instead of $G_{s}$.

2. **Running DMC Scripts**:
   - Code scripts must be tailored to the specific network being analyzed.

3. **Script Suggestions**:
   - Use **DMC_Facebook** as a starting point for alignment experiments on **Facebook** or **Youtube** networks.
   - For **Flickr**, **Lastfm**, and **Myspace**, the `flickr_degree_matrices.py` script is easier to adapt.
   - Both **DMC** and **Greedy DMC** scripts for the **PPI** network are included, enabling a comparison between these methods and baseline approaches.








