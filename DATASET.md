## DATASET
Our main 20 predicate category VG settings follows split from [Limited-Labels](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Scene_Graph_Prediction_With_Limited_Labels_ICCV_2019_paper.pdf).

### Download:
1. Download the VG images [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`. If you want to use other directory, please link it in `DATASETS['VG_stanford_filtered']['img_dir']` of `maskrcnn_benchmark/config/paths_catelog.py`. 
2. Download the 20 settings from [GoogleDrive](https://drive.google.com/file/d/1PvY-bDN0mJUF2fIi_1iTU9ys56h-OuGT/view?usp=sharing). Extract these files to `datasets/` and forms the file structure `datasets/vg/20`.
