# interpret-genetics

## Usage guide
Every file involving a network has 2 global variables at the top, right after all the imports. The first is the "TRAIN" flag, which is set to either true or false. The second is the "PREFIX" which determines the name of the model, following the form of "models/someprefix." If the TRAIN flag is set to True, a new model will be trained, and saved in the folder models under PREFIX\_model. Furthermore metrics for the model will also be saved in the same folder. The file called PREFIX\_all\_metrics contains lists of all the metrics taken every epoch. The file called PREFIX\_metrics contains only the best result for each metric. If the TRAIN flag is False, an already existing model matching the PREFIX will be loaded, and something testing the model will be ran. The following files contain networks, with a check denoting whether or not the model works well:
- [ ] discrete.py
- [x] fc\_sumstats.py
- [ ] everything in experiments/
- [x] run\_example.py
- [x] schrider.py
- [ ] static.py
- [x] transfer.py

## Plotters
real\_data.py and plotter.py create the graphs. As the names suggest, real\_data uses real data, while plotter uses simulated data. Currently, the networks they compare are hardcoded, with the best networks currently being used in those files. 

## Support 
sample\_files.py used pyvcf to create txt files that can then be taken in by bcftools to parse an entire chromosome for a specific population. msms\_keras/MSMS\_Generator.py contains all the different simulators used by the networks. The name of each class in the file is fairly self explanatory, and the correct simulator is currently being called in each of the network files. 
